"""minitorch 的測試：數值梯度檢查、Module / 優化器 / DataLoader 行為、擴散模型。"""

import os
import sys
import tempfile
import traceback
import numpy as np

import minitorch as mt
from minitorch import diffusion, functional as F, layers, models, optim
from minitorch.data import DataLoader, TensorDataset, pad_collate

rng = np.random.default_rng(0)
T = lambda *shape: mt.Tensor(rng.normal(size=shape), requires_grad=True)
tests = []


def test(fn):
    tests.append(fn)
    return fn


# ───────── 1. autograd 算子 ─────────

@test
def elementwise_and_broadcast():
    a, b, c = T(3, 4), T(4), T(3, 1)
    mt.gradcheck(lambda a, b, c: a * b + c - a / (b * b + 2) , [a, b, c])
    mt.gradcheck(lambda a: (-a) ** 3 + 2 - a, [T(2, 3)])
    mt.gradcheck(lambda a, b: 3 / (a * a + 1) + 2 * b, [T(5), T(1, 5)])


@test
def activations():
    for name in ['exp', 'tanh', 'sigmoid', 'relu', 'sqrt']:
        x = mt.Tensor(rng.uniform(0.3, 2, (3, 4)) * rng.choice([-1, 1], (3, 4)), requires_grad=True)
        if name == 'sqrt':
            x = mt.Tensor(rng.uniform(0.3, 2, (3, 4)), requires_grad=True)
        mt.gradcheck(lambda x: getattr(x, name)(), [x])
    mt.gradcheck(lambda x: x.log(), [mt.Tensor(rng.uniform(0.5, 3, (3, 3)), requires_grad=True)])
    mt.gradcheck(F.gelu, [T(3, 4)])


@test
def matmul_variants():
    mt.gradcheck(lambda a, b: a @ b, [T(3, 4), T(4, 5)])
    mt.gradcheck(lambda a, b: a @ b, [T(2, 3, 4), T(4, 5)])        # (B,T,C) @ (C,D)
    mt.gradcheck(lambda a, b: a @ b, [T(2, 2, 3, 4), T(2, 2, 4, 3)])  # 多頭注意力
    mt.gradcheck(lambda a, b: a @ b, [T(3, 4), T(2, 4, 5)])        # 左邊 broadcast


@test
def reductions():
    x = T(3, 4, 2)
    mt.gradcheck(lambda x: x.sum(), [x])
    mt.gradcheck(lambda x: x.sum(axis=1), [x])
    mt.gradcheck(lambda x: x.sum(axis=(0, 2), keepdims=True), [x])
    mt.gradcheck(lambda x: x.mean(axis=-1), [x])
    mt.gradcheck(lambda x: x.max(axis=1), [x])
    mt.gradcheck(lambda x: x.max(axis=1, keepdims=True), [x])
    mt.gradcheck(lambda x: x.max(), [x])


@test
def shape_ops_and_indexing():
    x = T(2, 3, 4)
    mt.gradcheck(lambda x: x.reshape(6, 4), [x])
    mt.gradcheck(lambda x: x.transpose(2, 0, 1), [x])
    mt.gradcheck(lambda x: x.T, [T(3, 5)])
    mt.gradcheck(lambda x: x[:, 1:, ::2], [x])
    mt.gradcheck(lambda x: x[np.array([0, 2, 2, 1])], [T(3, 4)])      # 重複索引要累加
    mt.gradcheck(lambda x: x[np.arange(3), np.array([1, 0, 3])], [T(3, 4)])
    mt.gradcheck(lambda a, b: mt.stack([a, b], axis=1), [T(2, 3), T(2, 3)])
    mt.gradcheck(lambda a, b: mt.concatenate([a, b], axis=1), [T(2, 3), T(2, 2)])
    mask = rng.random((3, 4)) > 0.5
    mt.gradcheck(lambda x: x.masked_fill(mask, -5.0), [T(3, 4)])


@test
def softmax_and_losses():
    mt.gradcheck(lambda x: x.softmax(-1), [T(3, 5)])
    mt.gradcheck(lambda x: x.log_softmax(-1), [T(3, 5)])
    targets = np.array([1, 4, 0])
    mt.gradcheck(lambda x: F.cross_entropy(x, targets), [T(3, 5)])
    ignored = np.array([1, -100, 0, -100])
    mt.gradcheck(lambda x: F.cross_entropy(x, ignored), [T(4, 5)])
    mt.gradcheck(lambda x: F.cross_entropy(x, np.array([[1, 2], [0, 3]])), [T(2, 2, 5)])
    mt.gradcheck(lambda p, t: F.mse_loss(p, t), [T(3, 2), T(3, 2)])


@test
def cross_entropy_matches_manual_formula():
    logits, targets = rng.normal(size=(6, 4)), rng.integers(0, 4, 6)
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    expected = -np.log(p[np.arange(6), targets]).mean()
    assert abs(F.cross_entropy(mt.Tensor(logits), targets).item() - expected) < 1e-12


@test
def conv_and_pool():
    mt.gradcheck(lambda x, w, b: F.conv2d(x, w, b), [T(2, 2, 6, 6), T(3, 2, 3, 3), T(3)])
    mt.gradcheck(lambda x, w: F.conv2d(x, w, padding=1), [T(1, 2, 5, 5), T(2, 2, 3, 3)])
    mt.gradcheck(lambda x: F.max_pool2d(x, 2), [T(2, 2, 6, 6)])
    mt.gradcheck(lambda x: F.max_pool2d(x, 2), [T(1, 1, 5, 5)])  # 奇數邊長
    assert F.conv2d(T(1, 1, 5, 5), T(2, 1, 3, 3), padding=1).shape == (1, 2, 5, 5)


@test
def engine_semantics():
    # 同一個 Tensor 被多次使用 → 梯度累加
    x = mt.Tensor([2.0], requires_grad=True)
    (x * x + x).backward()
    assert x.grad[0] == 5.0
    # 重複 backward 梯度累加到葉節點
    y = mt.Tensor([3.0], requires_grad=True)
    (y * 2).sum().backward(); (y * 2).sum().backward()
    assert y.grad[0] == 4.0
    # no_grad 不建立計算圖；detach 切斷梯度
    with mt.no_grad():
        z = y * 2
    assert not z.requires_grad
    w = (y * 2).detach() * y
    w.sum().backward()
    assert y.grad[0] == 4.0 + 6.0
    # ndarray 在左邊的運算
    assert isinstance(np.ones(2) + mt.Tensor([1.0, 2.0]), mt.Tensor)
    # 深計算圖不會遞迴爆炸
    v = mt.Tensor([1.0], requires_grad=True)
    out = v
    for _ in range(3000):
        out = out * 1.0001
    out.backward()
    assert abs(v.grad[0] - 1.0001 ** 3000) < 1e-6


# ───────── 2. 分類模型 ─────────

@test
def classifier_models_numeric_gradients():
    """MLP、CNN 的所有參數都做數值梯度檢查。"""
    mt.manual_seed(0)
    for model, x in [(models.MLP(hidden_sizes=(8,)), rng.random((3, 1, 28, 28))),
                     (models.CNN(channels1=2, channels2=3), rng.random((2, 1, 28, 28)))]:
        for p in model.parameters():
            p.data += rng.normal(0, 0.05, p.shape)
        labels = rng.integers(0, 10, len(x))
        mt.gradcheck(lambda *ps: F.cross_entropy(model(x), labels), model.parameters(), tol=1e-6, max_checks=5)
        assert model.predict(x).shape == (len(x),) and 0 <= model.accuracy(x, labels) <= 1


@test
def adam_matches_formula():
    param = mt.Parameter(rng.normal(size=(3, 2)))
    expected, m, v = param.data.copy(), 0, 0
    opt = optim.Adam([param], lr=0.01, betas=(0.9, 0.99))
    for step in range(1, 6):
        grad = rng.normal(size=expected.shape)
        param.grad = grad.copy()
        opt.step()
        m, v = 0.9 * m + 0.1 * grad, 0.99 * v + 0.01 * grad ** 2
        expected = expected - 0.01 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.99 ** step)) + 1e-8)
    assert np.allclose(param.data, expected, atol=1e-12)


# ───────── 3. Module 系統 ─────────

@test
def module_registration_and_state_dict():
    mt.manual_seed(1)
    model = models.GPT(20, 8, width=8, n_layer=2, n_head=2)
    names = [n for n, _ in model.named_parameters()]
    assert 'blocks.1.attn.qkv.weight' in names and 'ln_f.bias' in names
    assert len(names) == len(set(names))
    assert model.num_parameters() == sum(p.size for p in model.parameters())
    # 存檔 / 載入 round trip
    ids = rng.integers(0, 20, (2, 8))
    before = model(ids).data
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.npz')
        model.save(path)
        clone = models.GPT(20, 8, width=8, n_layer=2, n_head=2).load(path)
    assert np.allclose(clone(ids).data, before)
    try:
        clone.load_state_dict({'wte.weight': np.zeros((20, 8))})
        raise SystemExit('應該要丟出 KeyError')
    except KeyError:
        pass


@test
def train_eval_and_dropout():
    mt.manual_seed(0)
    drop = layers.Dropout(0.5)
    x = mt.ones(1000)
    out = drop(x).data
    assert 0.3 < (out == 0).mean() < 0.7 and abs(out.mean() - 1) < 0.15   # 縮放後期望值不變
    drop.eval()
    assert np.array_equal(drop(x).data, x.data)
    net = layers.Sequential(layers.Linear(3, 3), layers.Dropout(0.5))
    net.eval()
    assert not net[1].training
    net.train()
    assert net[1].training


@test
def layernorm_and_zero_grad():
    ln = layers.LayerNorm(6)
    y = ln(T(4, 6)).data
    assert np.allclose(y.mean(-1), 0, atol=1e-9) and np.allclose(y.var(-1), 1, atol=1e-3)
    ln.weight.data[...] = rng.normal(size=6)
    ln.bias.data[...] = rng.normal(size=6)
    mt.gradcheck(lambda x, w, b: ln(x), [T(3, 6), ln.weight, ln.bias], tol=1e-5)
    F.mse_loss(ln(T(2, 6)), mt.zeros(2, 6)).backward()
    assert all(p.grad is not None for p in ln.parameters())
    ln.zero_grad()
    assert all(p.grad is None for p in ln.parameters())


@test
def model_gradients_numeric():
    """整個 GPT / GRU 的所有參數都做數值梯度檢查（含 padding 的 ignore_index）。"""
    for model in [models.GPT(11, 6, width=8, n_layer=2, n_head=2), models.GRULanguageModel(11, 6, width=6)]:
        tokens = rng.integers(0, 11, (3, 6))
        targets = rng.integers(0, 11, (3, 6))
        targets[1, 4:] = -100
        params = model.parameters()
        loss_fn = lambda *ps: F.cross_entropy(model(tokens), targets)
        mt.gradcheck(loss_fn, params, tol=1e-6, max_checks=8)


@test
def causal_mask_no_future_leak():
    mt.manual_seed(3)
    model = models.GPT(11, 8, width=8)
    a = rng.integers(0, 11, (1, 8))
    b = a.copy(); b[0, 5:] = (b[0, 5:] + 1) % 11   # 只改後面的 token
    assert np.allclose(model(a).data[0, :5], model(b).data[0, :5])
    gru = models.GRULanguageModel(11, 8, width=6)
    assert np.allclose(gru(a).data[0, :5], gru(b).data[0, :5])


# ───────── 4. 優化器、排程、DataLoader ─────────

def quadratic_loss(p):
    return ((p - mt.Tensor([3.0, -2.0])) ** 2).sum()


@test
def optimizers_converge():
    for make in [lambda ps: optim.SGD(ps, lr=0.1), lambda ps: optim.SGD(ps, lr=0.05, momentum=0.9),
                 lambda ps: optim.Adam(ps, lr=0.1), lambda ps: optim.AdamW(ps, lr=0.1, weight_decay=0.0)]:
        p = mt.Parameter(np.zeros(2))
        opt = make([p])
        for _ in range(300):
            opt.zero_grad(); quadratic_loss(p).backward(); opt.step()
        assert np.allclose(p.data, [3, -2], atol=1e-2), (make, p.data)


@test
def adamw_decoupled_weight_decay():
    p = mt.Parameter(np.ones(3)); p.grad = np.zeros(3)
    optim.AdamW([p], lr=0.1, weight_decay=0.5).step()
    assert np.allclose(p.data, 1 - 0.1 * 0.5)          # 零梯度時只有衰減
    q = mt.Parameter(np.ones(3)); q.grad = np.zeros(3)
    optim.Adam([q], lr=0.1, weight_decay=0.5).step()   # L2 版本經過 Adam 正規化，不等於上面
    assert not np.allclose(q.data, p.data)


@test
def clip_and_schedulers():
    a, b = mt.Parameter(np.zeros(2)), mt.Parameter(np.zeros(2))
    a.grad, b.grad = np.array([3.0, 0.0]), np.array([0.0, 4.0])
    norm = optim.clip_grad_norm_([a, b], 1.0)
    assert abs(norm - 5) < 1e-12
    assert abs(np.sqrt((a.grad ** 2).sum() + (b.grad ** 2).sum()) - 1) < 1e-4
    opt = optim.SGD([a], lr=1.0)
    sched = optim.LambdaLR(opt, optim.warmup_cosine(100, warmup_steps=10, min_ratio=0.1))
    lrs = []
    for _ in range(100):
        lrs.append(opt.lr); sched.step()
    assert lrs[0] < lrs[9] <= 1.0 and abs(lrs[10] - 1.0) < 1e-9 and abs(lrs[-1] - 0.1) < 0.01
    assert abs(optim.linear_decay(10)(5) - 0.5) < 1e-12


@test
def dataloader():
    x, y = np.arange(10)[:, None] * np.ones((1, 2)), np.arange(10)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    batches = list(loader)
    assert len(loader) == 3 and [len(b[1]) for b in batches] == [4, 4, 2]
    assert len(DataLoader(TensorDataset(x, y), batch_size=4, drop_last=True)) == 2
    shuffled = np.concatenate([b[1] for b in DataLoader(TensorDataset(x, y), 4, shuffle=True, seed=1)])
    assert sorted(shuffled) == list(range(10)) and list(shuffled) != list(range(10))
    seqs = [(np.array([1, 2, 3]), np.array([2, 3, 4])), (np.array([5]), np.array([6]))]
    tokens, targets = next(iter(DataLoader(seqs, 2, collate_fn=pad_collate((0, -100)))))
    assert tokens.tolist() == [[1, 2, 3], [5, 0, 0]] and targets.tolist() == [[2, 3, 4], [6, -100, -100]]


@test
def tiny_training_actually_learns():
    """MLP 學會 XOR；證明 forward / backward / 優化器 / Module 整條流程接起來了。"""
    mt.manual_seed(0)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    net = layers.Sequential(layers.Linear(2, 8), layers.Tanh(), layers.Linear(8, 2))
    opt = optim.Adam(net.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(net(x), y).backward(); opt.step()
    assert (net(x).argmax(axis=1) == y).all()


# ───────── 5. 擴散模型 ─────────

@test
def upsample_and_silu():
    mt.gradcheck(F.upsample2x, [T(2, 2, 3, 3)])
    x = mt.Tensor(np.arange(4.0).reshape(1, 1, 2, 2))
    assert F.upsample2x(x).data[0, 0].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]
    mt.gradcheck(F.silu, [T(3, 4)])


@test
def unet_shapes_conditioning_and_gradients():
    mt.manual_seed(0)
    net = models.UNet(num_classes=3, channels=2, emb_dim=8)
    x, t, y = rng.normal(size=(2, 1, 8, 8)), np.array([3, 40]), np.array([0, 2])
    assert net(x, t, y).shape == x.shape
    for p in net.parameters():                       # 讓輸出對條件有反應（初始 conv_out 偏小）
        p.data += rng.normal(0, 0.05, p.shape)
    assert not np.allclose(net(x, t, y).data, net(x, t, np.array([1, 1])).data)   # 類別有影響
    assert not np.allclose(net(x, t, y).data, net(x, np.array([9, 9]), y).data)   # 時間步有影響
    target = rng.normal(size=x.shape)
    mt.gradcheck(lambda *ps: F.mse_loss(net(x, t, y), target), net.parameters(), tol=1e-6, max_checks=4)


@test
def diffusion_schedule_and_forward_process():
    d = diffusion.Diffusion(T=100)
    assert np.all(np.diff(d.alpha_bars) < 0) and d.alpha_bars[-1] < 0.05 and d.alpha_bars[0] > 0.99
    assert d.posterior_var[0] == 0 and np.all(d.posterior_var[1:] > 0) and np.all(d.posterior_var <= d.betas)
    x0 = np.ones((20000, 1, 1, 1)) * 0.7
    t = np.full(20000, 50)
    xt = d.q_sample(x0, t, rng.normal(size=x0.shape))
    assert abs(xt.mean() - np.sqrt(d.alpha_bars[50]) * 0.7) < 0.02
    assert abs(xt.var() - (1 - d.alpha_bars[50])) < 0.02


@test
def diffusion_sampler_recovers_x0_with_oracle_noise_predictor():
    """若模型能精確預測雜訊，逆向取樣必須還原出原本的影像：驗證後驗係數與取樣迴圈。"""
    d = diffusion.Diffusion(T=50)
    target = np.clip(rng.normal(size=(1, 5, 5)), -0.9, 0.9)

    def oracle(x, t, labels):
        ab = d.alpha_bars[t[0]]
        return mt.Tensor((x - np.sqrt(ab) * target) / np.sqrt(1 - ab))

    mt.manual_seed(1)
    out = d.sample(oracle, np.zeros(3, dtype=int), (1, 5, 5))
    assert np.allclose(out, target[None], atol=1e-9)


@test
def diffusion_training_reduces_loss():
    """極小資料、極小 UNet：訓練幾十步，雜訊預測損失應明顯下降。"""
    mt.manual_seed(0)
    d = diffusion.Diffusion(T=20)
    images = np.concatenate([np.full((16, 1, 8, 8), -0.8), np.full((16, 1, 8, 8), 0.8)])
    labels = np.repeat([0, 1], 16)
    net = models.UNet(num_classes=2, channels=4, emb_dim=16)
    opt = optim.Adam(net.parameters(), lr=0.01)
    losses = []
    for _ in range(60):
        opt.zero_grad(); loss = d.loss(net, images, labels); loss.backward(); opt.step(); losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.7 * np.mean(losses[:10]), (np.mean(losses[:10]), np.mean(losses[-10:]))


@test
def data_and_viz_helpers():
    from minitorch import data, viz
    images, labels = np.arange(6)[:, None, None, None] * np.ones((6, 1, 4, 4)), np.array([3, 5, 3, 7, 5, 3])
    x, y = data.select_classes(images, labels, (7, 3), per_class=2)
    assert y.tolist() == [0, 1, 1] and x[:, 0, 0, 0].tolist() == [3, 0, 2]
    assert data.shrink(np.arange(16.0).reshape(1, 1, 4, 4), 2)[0, 0].tolist() == [[2.5, 4.5], [10.5, 12.5]]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'a.png')
        viz.save_png(path, viz.make_grid(rng.random((4, 5, 5)), 2), scale=3)
        raw = open(path, 'rb').read()
    assert raw[:8] == b'\x89PNG\r\n\x1a\n' and raw.endswith(b'IEND\xaeB`\x82')
    assert len(viz.ascii_row(rng.random((2, 3, 3))).splitlines()) == 3


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
