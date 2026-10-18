"""numpy-nn 的測試：手寫梯度的數值梯度檢查，以及訓練、取樣的基本行為。"""

import contextlib
import io
import os
import sys
import tempfile
import traceback
import numpy as np

from npnn import data, models, train

rng = np.random.default_rng(0)
tests = []


def test(fn):
    tests.append(fn)
    return fn


def check_gradients(model, loss_fn, samples=6, eps=1e-5, tol=1e-6):
    """對每個參數隨機抽樣幾個元素，比較解析梯度與中央差分數值梯度；回傳最大相對誤差。"""
    for p in model.parameters.values():            # 加一點雜訊，避開全零、對稱等特殊點
        p += rng.normal(0, 0.05, p.shape)
    _, gradients = loss_fn()
    worst = 0.0
    for name, p in model.parameters.items():
        for _ in range(samples):
            index = tuple(rng.integers(0, n) for n in p.shape)
            old = p[index]
            p[index] = old + eps; plus, _ = loss_fn()
            p[index] = old - eps; minus, _ = loss_fn()
            p[index] = old
            numeric = (plus - minus) / (2 * eps)
            worst = max(worst, abs(numeric - gradients[name][index]) / (1 + abs(numeric)))
    assert worst < tol, f'最大誤差 {worst:.3e}'
    return worst


# ───────── 1. 手寫梯度 ─────────

@test
def gradients_gpt():
    model = models.GPT(12, 8, width=8)
    tokens, targets = np.array([11, 1, 2, 3, 2, 4]), np.array([1, 2, 3, 2, 4, 11])
    check_gradients(model, lambda: model.loss_and_gradients(tokens, targets))


@test
def gradients_gru():
    model = models.GRU(12, 8, width=6)
    tokens, targets = np.array([11, 1, 2, 3, 2, 4]), np.array([1, 2, 3, 2, 4, 11])
    check_gradients(model, lambda: model.loss_and_gradients(tokens, targets))


@test
def gradients_mlp():
    model = models.MLP(hidden_sizes=(16, 8))
    x, y = rng.random((5, 1, 28, 28)), rng.integers(0, 10, 5)
    check_gradients(model, lambda: model.loss_and_gradients(x, y))


@test
def gradients_cnn():
    model = models.CNN(channels1=3, channels2=4)
    x, y = rng.random((3, 1, 28, 28)), rng.integers(0, 10, 3)
    check_gradients(model, lambda: model.loss_and_gradients(x, y), samples=4)


# ───────── 2. 基本元件 ─────────

@test
def softmax_is_stable_and_normalized():
    p = train.softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p[1], 1 / 3)


@test
def adam_matches_formula():
    param = {'w': rng.normal(size=(3, 2))}
    start, expected = param['w'].copy(), param['w'].copy()
    adam = train.Adam(param)
    m = v = np.zeros_like(expected)
    for step in range(1, 6):
        grad = rng.normal(size=expected.shape)
        adam.step({'w': grad}, 0.01)
        m, v = 0.9 * m + 0.1 * grad, 0.99 * v + 0.01 * grad ** 2
        expected = expected - 0.01 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.99 ** step)) + 1e-8)
    assert np.allclose(param['w'], expected, atol=1e-12) and not np.allclose(param['w'], start)


@test
def corpus_encoding():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'c.txt')
        open(path, 'w', encoding='utf-8').write('小貓\n\n小狗在跑\n')
        corpus = data.CharCorpus(path, block_size=4)
    assert corpus.docs == ['小貓', '小狗在跑'] and corpus.vocab_size == len(corpus.chars) + 1
    tokens, targets = corpus.encode('小狗在跑')
    assert tokens[0] == corpus.BOS and len(tokens) == 4                    # 超過 block_size 會被截斷
    assert targets[:3].tolist() == tokens[1:].tolist()                     # target 是 token 向左錯一位
    tokens, targets = corpus.encode('小貓')
    assert targets[-1] == corpus.BOS and len(tokens) == 3                  # 結尾補 BOS
    try:
        data.CharCorpus(os.path.join(d, 'nope.txt'))
        raise SystemExit('應該要丟出 FileNotFoundError')
    except FileNotFoundError:
        pass


# ───────── 3. 模型行為 ─────────

@test
def language_models_are_causal():
    for model in [models.GPT(12, 8, width=8), models.GRU(12, 8, width=6)]:
        a = rng.integers(0, 12, 8)
        b = a.copy(); b[5:] = (b[5:] + 1) % 12     # 只改後面的 token
        assert np.allclose(model.forward(a)[:5], model.forward(b)[:5])
        try:
            model.forward(np.zeros(9, dtype=int))   # 超過 block_size 要報錯
            raise SystemExit('應該要丟出 ValueError')
        except ValueError:
            pass


@test
def language_models_learn():
    """極小語料訓練 150 步，損失應明顯下降。"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'c.txt')
        open(path, 'w', encoding='utf-8').write('小貓吃魚\n小狗吃肉\n小貓吃魚\n小狗吃肉\n')
        corpus = data.CharCorpus(path, block_size=8)
    for model in [models.GPT(corpus.vocab_size, 8, width=12), models.GRU(corpus.vocab_size, 8, width=12)]:
        adam, losses = train.Adam(model.parameters), []
        for step in range(150):
            tokens, targets = corpus.encode(corpus.docs[step % len(corpus.docs)])
            loss, grads = model.loss_and_gradients(tokens, targets)
            adam.step(grads, 0.02)
            losses.append(loss)
        assert np.mean(losses[-8:]) < 0.5 * np.mean(losses[:8]), (np.mean(losses[:8]), np.mean(losses[-8:]))
        with contextlib.redirect_stdout(io.StringIO()):
            sentences = train.generate(model, corpus, num_samples=5)
        assert all(len(s) <= 8 and set(s) <= set(corpus.chars) for s in sentences)


@test
def classifiers_learn():
    """每個類別一個固定圖樣加雜訊：MLP、CNN 訓練後應能完全分辨。"""
    protos = rng.random((10, 28, 28)) > 0.7
    def make(n):
        y = rng.integers(0, 10, n)
        return ((protos[y] * 200 + rng.random((n, 28, 28)) * 55) / 255.0)[:, None], y
    (x_train, y_train), (x_test, y_test) = make(400), make(100)
    for model, epochs in [(models.MLP(hidden_sizes=(32,)), 4), (models.CNN(channels1=4, channels2=4), 4)]:
        with contextlib.redirect_stdout(io.StringIO()):
            train.train_classifier(model, x_train, y_train, x_test, y_test, num_epochs=epochs, batch_size=32, learning_rate=0.01)
        assert model.accuracy(x_test, y_test) > 0.95, type(model).__name__
        assert model.predict(x_test[:7]).shape == (7,)


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
