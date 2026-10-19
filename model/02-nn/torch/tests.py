"""PyTorch 版的測試：預訓練 / 微調的各個元件、MPS 相關的相容性檢查，以及模型與擴散模型的基本行為。"""

import contextlib
import io
import os
import sys
import tempfile
import traceback
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import nnlib  # 先於其他 nnlib 模組匯入：會設定 PYTORCH_ENABLE_MPS_FALLBACK
from nnlib import checkpoint, data, diffusion, models, qa, train
from nnlib.models import CausalSelfAttention

tests = []
rng = np.random.default_rng(0)


def test(fn):
    tests.append(fn)
    return fn


def project_corpus():
    pre, fine = data.read_docs('pretrain.txt'), data.read_docs('finetune.txt')
    return data.CharCorpus(docs=pre, block_size=32, vocab=data.build_vocab(pre, fine))


def small_gpt(corpus, width=8, n_layer=2):
    torch.manual_seed(0)
    return models.GPT(corpus.vocab_size, 32, width=width, n_layer=n_layer, n_head=2)


# ───────── 1. 資料 ─────────

@test
def corpus_docs_vocab_and_split():
    corpus = data.CharCorpus(docs=['小貓'], vocab=list('小貓狗'))   # 詞彙表可以比文件大（沿用預訓練的字元表）
    assert corpus.chars == list('小貓狗') and corpus.BOS == 3 and corpus.vocab_size == 4
    try:
        data.CharCorpus(docs=['小魚'], vocab=list('小貓'))
        raise SystemExit('應該要丟出 ValueError')
    except ValueError:
        pass
    assert data.build_vocab(['ba'], ['ac']) == ['a', 'b', 'c']
    docs = [f'句子{i}' for i in range(20)]
    train_docs, val_docs = data.split_docs(docs, 0.1, seed=1)
    assert len(val_docs) == 2 and not set(train_docs) & set(val_docs) and sorted(train_docs + val_docs) == sorted(docs)


@test
def finetune_file_parsing_and_folds():
    assert qa.parse_line('<Q>火星有幾顆小衛星？<A>兩顆') == ('火星有幾顆小衛星？', '兩顆')
    for bad in ['火星有幾顆小衛星？兩顆', '<Q>問題<A>', '<Q><A>答案', '<Q>a<A>b<A>c']:
        try:
            qa.parse_line(bad)
            raise SystemExit(f'應該要丟出 ValueError：{bad}')
        except ValueError:
            pass
    lines = qa.read_pairs('finetune.txt')
    unique = qa.unique_pairs(lines)
    assert len(lines) == 1350 and len(unique) == 50
    assert len({q for q, _ in unique}) == 50                                  # 每個問題只有一個答案
    assert max(len(qa.prompt(q) + a) for q, a in unique) + 1 <= 32            # 放得進 block_size=32
    sentences = data.read_docs('pretrain.txt')
    missing = [(q, a) for q, a in unique if not any(a in s for s in sentences)]
    assert missing == [('火星有幾顆小衛星？', '兩顆')], missing                 # 49 題的答案逐字出現在預訓練句子裡；這題的句子寫「兩個」
    folds = qa.make_folds(unique, k=5, seed=1)
    assert [len(f) for f in folds] == [10] * 5 and sorted(sum(folds, [])) == unique
    assert qa.make_folds(unique, k=5, seed=1) == folds and qa.make_folds(unique, k=5, seed=2) != folds


@test
def cross_validation_has_no_leakage():
    """驗證題目的所有重複行都必須從訓練行裡拿掉，否則驗證題目會和訓練行完全相同。"""
    lines = qa.read_pairs('finetune.txt')
    for held in qa.make_folds(qa.unique_pairs(lines), k=5, seed=42):
        remaining = qa.without(lines, held)
        assert not set(remaining) & set(held) and len(remaining) == 1350 - 10 * 27
        assert set(remaining) | set(held) == set(lines)


# ───────── 2. 遮罩、損失、檢查點、凍結 ─────────

@test
def answer_only_masking_and_loss():
    corpus = project_corpus()
    question, answer = qa.unique_pairs(qa.read_pairs('finetune.txt'))[0]
    tokens, targets = qa.encode_pair(corpus, question, answer)
    n = len(qa.prompt(question))
    assert (targets[:n] == -100).all() and len(targets) == n + len(answer) + 1
    assert targets[n:].tolist() == [corpus.token_of[c] for c in answer] + [corpus.BOS]   # 答案與結尾 BOS 才計損失
    model = small_gpt(corpus)
    logits = model(torch.as_tensor(tokens)[None])[0]
    logp = logits.log_softmax(-1)
    expected = -sum(logp[n + i, targets[n + i]].item() for i in range(len(answer) + 1)) / (len(answer) + 1)
    assert abs(F.cross_entropy(logits, torch.as_tensor(targets), ignore_index=-100).item() - expected) < 1e-6


@test
def checkpoint_roundtrip():
    corpus = project_corpus()
    model = small_gpt(corpus)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.pt')
        checkpoint.save_gpt(path, model, corpus.chars)
        clone, chars = checkpoint.load_gpt(path, 'cpu')
    ids = torch.as_tensor(rng.integers(0, corpus.vocab_size, (2, 32)))
    assert chars == corpus.chars and clone.config == model.config            # 字元表含 < > Q A ？ 等符號，順序不變
    with torch.no_grad():
        assert torch.allclose(clone.eval()(ids), model.eval()(ids), atol=1e-6)


@test
def freeze_blocks_updates():
    corpus = project_corpus()
    model = small_gpt(corpus)
    total = train.count_parameters(model)
    frozen = train.freeze(model, 'wte', 'blocks.0.')
    assert frozen > 0 and train.count_parameters(model, trainable_only=True) == total - frozen
    before = {k: v.clone() for k, v in model.state_dict().items()}
    examples = qa.encode_pairs(corpus, qa.unique_pairs(qa.read_pairs('finetune.txt'))[:8])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        train.train_language_model(model, examples, epochs=3, lr=0.05, batch_size=4)   # log_every=0：不應印任何東西
    assert out.getvalue() == ''
    for name, value in model.state_dict().items():
        same = torch.equal(before[name], value)
        assert same == name.startswith(('wte', 'blocks.0.')), name           # 只有凍結的沒變
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith('wte'))


# ───────── 3. 評估與取樣 ─────────

@test
def evaluate_loss_complete_and_ranking():
    corpus = project_corpus()
    model = small_gpt(corpus, n_layer=1)
    unique = qa.unique_pairs(qa.read_pairs('finetune.txt'))
    examples = qa.encode_pairs(corpus, unique[:5])
    tokens, targets = train._pad_collate(examples)
    with torch.no_grad():
        summed = F.cross_entropy(model.eval()(tokens).reshape(-1, corpus.vocab_size), targets.reshape(-1),
                                 ignore_index=-100, reduction='sum').item()
    assert abs(train.evaluate_loss(model, examples, batch_size=2) - summed / (targets != -100).sum().item()) < 1e-6
    assert abs(train.sequence_nll(model, examples, batch_size=2).sum() - summed) < 1e-4     # 每個樣本的總 NLL 加起來 = 總損失
    model.train()
    prompt = qa.prompt(unique[0][0])
    first = train.complete(model, corpus, prompt)
    assert first == train.complete(model, corpus, prompt) and model.training      # 貪婪解碼固定、不改變 train/eval 狀態
    assert len(train.complete(model, corpus, prompt, max_new=3)) <= 3
    assert train.complete(model, corpus, corpus.chars[0] * 31) == ''            # prompt + BOS 已占滿 block_size
    assert len(train.complete(model, corpus, corpus.chars[0] * 30, max_new=5)) <= 1
    assert set(train.complete(model, corpus, prompt, temperature=0.8)) <= set(corpus.chars)
    candidates = sorted({a for _, a in unique})
    ranks = qa.answer_ranks(model, corpus, unique[:6], candidates)
    assert all(1 <= r <= len(candidates) for r in ranks)
    assert 1 <= qa.expected_random_rank(corpus, unique, candidates) <= len(candidates)


@test
def ranking_prefers_the_trained_answer():
    """讓模型過度訓練在單一題目上，這題的正確答案名次必須是第 1。"""
    corpus = project_corpus()
    unique = qa.unique_pairs(qa.read_pairs('finetune.txt'))
    candidates = sorted({a for _, a in unique})
    torch.manual_seed(0)
    model = models.GPT(corpus.vocab_size, 32, width=16, n_layer=1, n_head=2)
    target = unique[3]
    train.train_language_model(model, qa.encode_pairs(corpus, [target] * 16), epochs=20, lr=0.02, weight_decay=0.0)
    assert qa.answer_ranks(model, corpus, [target], candidates) == [1]
    assert qa.answer_accuracy(model, corpus, [target]) == 1.0


@test
def pretrain_then_finetune_reduces_qa_loss():
    """縮小版完整流程：預訓練 → 存檔 → 載入 → 微調，問答損失應明顯下降，且預訓練的權重確實被帶進來。"""
    corpus = project_corpus()
    torch.manual_seed(0)
    model = models.GPT(corpus.vocab_size, 32, width=16, n_layer=1, n_head=2)
    train.train_language_model(model, corpus.examples()[:600], epochs=2, lr=0.01)
    with tempfile.TemporaryDirectory() as d:
        checkpoint.save_gpt(os.path.join(d, 'p.pt'), model, corpus.chars)
        loaded, chars = checkpoint.load_gpt(os.path.join(d, 'p.pt'))
    examples = qa.encode_pairs(data.CharCorpus(docs=[], block_size=32, vocab=chars), qa.read_pairs('finetune.txt')[:300])
    before = train.evaluate_loss(loaded, examples)
    train.train_language_model(loaded, examples, epochs=2, lr=0.003, weight_decay=0.0)
    assert train.evaluate_loss(loaded, examples) < 0.5 * before
    assert (loaded.wte.weight - model.wte.weight).abs().max().item() < 0.5      # 微調只小幅移動權重，沒有重新初始化


# ───────── 4. MPS 相容性（在 CPU 上驗證相關程式路徑）─────────

@test
def manual_attention_matches_sdpa():
    """MPS 上改用手寫注意力：前向與梯度必須和 scaled_dot_product_attention 一致，且是因果的。"""
    torch.manual_seed(0)
    gpt = models.GPT(30, 16, width=16, n_layer=2, n_head=2).eval()
    ids = torch.as_tensor(rng.integers(0, 30, (4, 16)))

    def run():
        gpt.zero_grad()
        out = gpt(ids)
        out.square().mean().backward()
        return out.detach(), [p.grad.clone() for p in gpt.parameters()]

    ref_out, ref_grads = run()
    CausalSelfAttention.force_manual = True
    try:
        man_out, man_grads = run()
        shifted = ids.clone()
        shifted[:, 10:] = (shifted[:, 10:] + 1) % 30                          # 只改後面的 token
        assert torch.allclose(gpt(ids)[:, :10], gpt(shifted)[:, :10], atol=1e-6)
    finally:
        CausalSelfAttention.force_manual = False
    assert torch.allclose(ref_out, man_out, atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(ref_grads, man_grads))


@test
def device_selection_and_float32():
    with mock.patch('torch.cuda.is_available', return_value=False), mock.patch('torch.backends.mps.is_available', return_value=True):
        assert train.get_device().type == 'mps' and train.get_device('cpu').type == 'cpu'
        with mock.patch.dict(os.environ, {'DEVICE': 'cpu'}):
            assert train.get_device().type == 'cpu'
    with mock.patch('torch.cuda.is_available', return_value=True):
        assert train.get_device().type == 'cuda'
    with mock.patch('torch.cuda.is_available', return_value=False), mock.patch('torch.backends.mps.is_available', return_value=False):
        assert train.get_device().type == 'cpu'
        for bad in ('mps', 'cuda'):
            try:
                train.get_device(bad)
                raise SystemExit('應該要報錯')
            except RuntimeError:
                pass
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '1'
    # MPS 不支援 float64：所有參數與擴散排程的緩衝區都必須是 float32
    d = diffusion.Diffusion(T=100)
    buffers = [d.betas, d.sqrt_alpha_bars, d.sqrt_one_minus_alpha_bars, d.posterior_coef_x0, d.posterior_coef_xt, d.posterior_std]
    assert all(b.dtype == torch.float32 for b in buffers)
    for m in [models.MLP(), models.CNN(), models.GRULanguageModel(30, 16), models.GPT(30, 16), models.UNet(3)]:
        assert all(p.dtype == torch.float32 for p in m.parameters())


# ───────── 5. 模型與擴散模型的基本行為 ─────────

@test
def model_output_shapes():
    assert models.MLP()(torch.randn(3, 1, 28, 28)).shape == (3, 10)
    assert models.CNN()(torch.randn(3, 1, 28, 28)).shape == (3, 10)
    ids = torch.as_tensor(rng.integers(0, 30, (3, 16)))
    assert models.GRULanguageModel(30, 16)(ids).shape == (3, 16, 30)
    assert models.GPT(30, 16)(ids).shape == (3, 16, 30)
    try:
        models.GPT(30, 16)(torch.zeros(1, 17, dtype=torch.long))
        raise SystemExit('應該要丟出 ValueError')
    except ValueError:
        pass
    assert models.UNet(3)(torch.randn(2, 1, 14, 14), torch.tensor([3, 40]), torch.tensor([0, 2])).shape == (2, 1, 14, 14)


@test
def diffusion_schedule_and_oracle_sampler():
    d = diffusion.Diffusion(T=50)
    assert (d.sqrt_alpha_bars[1:] < d.sqrt_alpha_bars[:-1]).all() and d.sqrt_alpha_bars[-1] ** 2 < 0.05
    x0 = torch.full((20000, 1, 1, 1), 0.7)
    xt = d.q_sample(x0, torch.full((20000,), 25), torch.randn_like(x0))
    ab = d.sqrt_alpha_bars[25].item() ** 2
    assert abs(xt.mean().item() - ab ** 0.5 * 0.7) < 0.02 and abs(xt.var().item() - (1 - ab)) < 0.02
    # 若模型能精確預測雜訊，逆向取樣必須還原出原本的影像：驗證後驗係數與取樣迴圈
    target = torch.as_tensor(np.clip(rng.normal(size=(1, 5, 5)), -0.9, 0.9), dtype=torch.float32)

    class Oracle(torch.nn.Module):
        def forward(self, x, t, labels):
            ab_t = d.sqrt_alpha_bars[t[0]] ** 2
            return (x - ab_t.sqrt() * target) / (1 - ab_t).sqrt()

    torch.manual_seed(1)
    out = d.sample(Oracle(), torch.zeros(3, dtype=torch.long), (1, 5, 5))
    assert torch.allclose(out, target.expand(3, -1, -1, -1), atol=1e-4)


if __name__ == '__main__':
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'✓ {fn.__name__}')
        except Exception:
            failed += 1
            print(f'✗ {fn.__name__}')
            traceback.print_exc()
    print(f'\n{len(tests) - failed}/{len(tests)} 通過')
    sys.exit(1 if failed else 0)
