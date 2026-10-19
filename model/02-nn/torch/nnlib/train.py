"""訓練流程：分類、語言模型、擴散模型的訓練迴圈與取樣，以及可直接呼叫的完整 demo。"""

import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from . import viz


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def warmup_cosine(total_steps, warmup_steps=0, min_ratio=0.0):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


def _fit(model, loader, loss_fn, device, lr, weight_decay, epochs, warmup_steps, min_ratio,
         clip_norm, log_every, after_epoch=None):
    total = len(loader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total, warmup_steps, min_ratio))
    step, start = 0, time.time()
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = [b.to(device) for b in batch]
            loss = loss_fn(model, *batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % log_every == 0:
                print(f'步驟 {step:5d} / {total}，損失 {loss.item():.4f}，{time.time() - start:.0f} 秒')
        if after_epoch:
            after_epoch(epoch)


def _loader(dataset, batch_size, seed, **kwargs):
    return DataLoader(dataset, batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), **kwargs)


# ───────── 分類（MLP、CNN） ─────────

@torch.no_grad()
def predict(model, images, device, batch_size=500):
    model.eval()
    return torch.cat([model(images[i:i + batch_size].to(device)).argmax(dim=1).cpu()
                      for i in range(0, len(images), batch_size)])


def run_classifier_demo(model, data, batch_size=64, epochs=2, lr=0.003, weight_decay=0.0,
                        clip_norm=None, log_every=100, seed=42):
    device = get_device()
    x_train, y_train, x_test, y_test = (torch.as_tensor(a, dtype=torch.float32 if i % 2 == 0 else torch.long)
                                        for i, a in enumerate(data))
    model.to(device)
    print(f'裝置：{device}，訓練資料：{len(x_train)} 張，測試資料：{len(x_test)} 張，模型參數：{count_parameters(model)}')
    loader = _loader(TensorDataset(x_train, y_train), batch_size, seed, drop_last=True)
    total = len(loader) * epochs
    _fit(model, loader, lambda m, x, y: F.cross_entropy(m(x), y), device, lr, weight_decay, epochs,
         warmup_steps=min(50, total // 10), min_ratio=0.1, clip_norm=clip_norm, log_every=log_every,
         after_epoch=lambda e: print(
             f'第 {e + 1} 輪結束：測試準確率 {(predict(model, x_test, device) == y_test).float().mean().item():.2%}'))
    print('\n--- 前 10 張測試圖的預測 ---')
    print('預測：', predict(model, x_test[:10], device).tolist())
    print('答案：', y_test[:10].tolist())


# ───────── 語言模型（GRU、GPT） ─────────

def _pad_collate(items):
    tokens = pad_sequence([torch.as_tensor(t) for t, _ in items], batch_first=True, padding_value=0)
    targets = pad_sequence([torch.as_tensor(y) for _, y in items], batch_first=True, padding_value=-100)
    return tokens, targets


@torch.no_grad()
def generate(model, corpus, device, num_samples=12, temperature=0.6):
    """從 BOS 開始逐字取樣，遇到 BOS 或達到 block_size 就結束。"""
    model.eval()
    sentences = []
    for _ in range(num_samples):
        token_ids, sample = [corpus.BOS], []
        for _ in range(corpus.block_size):
            logits = model(torch.tensor([token_ids], device=device))[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=-1).tolist()
            token_id = random.choices(range(corpus.vocab_size), weights=probabilities)[0]
            if token_id == corpus.BOS:
                break
            sample.append(corpus.chars[token_id])
            token_ids.append(token_id)
        sentences.append(''.join(sample))
    return sentences


def run_language_model_demo(model, corpus, title, batch_size=16, epochs=60, lr=0.01,
                            weight_decay=0.01, clip_norm=1.0, log_every=100, seed=42):
    device = get_device()
    model.to(device)
    print(f'裝置：{device}，{corpus.describe()}，模型參數：{count_parameters(model)}')
    loader = _loader(corpus.examples(), batch_size, seed, collate_fn=_pad_collate)
    total = len(loader) * epochs

    def loss_fn(m, tokens, targets):
        logits = m(tokens)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    _fit(model, loader, loss_fn, device, lr, weight_decay, epochs, warmup_steps=total // 20,
         min_ratio=0.05, clip_norm=clip_norm, log_every=log_every)
    print(f'\n--- 生成的句子（{title}） ---')
    for index, sentence in enumerate(generate(model, corpus, device), 1):
        print(f'{index:2d}. {sentence}')


# ───────── 擴散模型 ─────────

def run_diffusion_demo(model, diffusion, images, labels, class_names, steps=3000, batch_size=64, lr=2e-3,
                       samples_per_class=6, png_path='samples.png', log_every=250, seed=42):
    """訓練條件式擴散模型，然後每個類別各生成幾張圖：印出 ASCII 並存成 PNG。"""
    device = diffusion.device
    model.to(device)
    print(f'裝置：{device}，訓練影像：{len(images)} 張（{images.shape[-2]}×{images.shape[-1]}），類別 {list(class_names)}，'
          f'模型參數：{count_parameters(model)}，擴散步數 T={diffusion.T}')
    dataset = TensorDataset(torch.as_tensor(images, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long))
    loader = _loader(dataset, batch_size, seed, drop_last=True)
    epochs = max(1, round(steps / len(loader)))
    total = len(loader) * epochs
    _fit(model, loader, lambda m, x, y: diffusion.loss(m, x, y), device, lr, 0.0, epochs,
         warmup_steps=total // 20, min_ratio=0.05, clip_norm=1.0, log_every=log_every)

    print('\n取樣中……')
    start = time.time()
    sample_labels = torch.arange(len(class_names), device=device).repeat_interleave(samples_per_class)
    generated = diffusion.sample(model, sample_labels, tuple(images.shape[1:])).cpu().numpy()
    pictures = np.clip((generated[:, 0] + 1) / 2, 0, 1)
    print(f'取樣完成（{time.time() - start:.1f} 秒）。每個類別的第一張：')
    print(viz.ascii_row(pictures[::samples_per_class]))
    print('  ' + '  |  '.join(f'數字 {c}'.center(images.shape[-1] * 2) for c in class_names))
    if png_path:
        viz.save_png(png_path, viz.make_grid(pictures, samples_per_class), scale=6)
        print(f'\n已儲存 {png_path}（每列一個類別：{list(class_names)}）')
    return pictures
