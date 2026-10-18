"""自動微分核心：Tensor 包住 ndarray 並記錄計算圖，backward() 依拓撲順序反向傳播。

每個算子只需提供 forward 結果與一個 backward(grad) -> 各父節點梯度 的函式；
broadcasting 的梯度加總（unbroadcast）與梯度累加由引擎統一處理。
"""

import functools
import numpy as np

_grad_enabled = True
_rng = np.random.default_rng(0)


def manual_seed(seed):
    """設定全域亂數種子（影響權重初始化、dropout）。"""
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng():
    return _rng


class no_grad:
    """with no_grad(): ... 內不建立計算圖（評估、取樣時使用）；也可當裝飾器。"""

    def __enter__(self):
        global _grad_enabled
        self._previous, _grad_enabled = _grad_enabled, False

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._previous

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)
        return wrapper


def is_grad_enabled():
    return _grad_enabled


def unbroadcast(grad, shape):
    """把 broadcasting 後的梯度加總回原本的 shape。"""
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad


def _wrap(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def _make(data, parents, backward):
    """建立輸出 Tensor；只有在需要梯度時才記錄計算圖。"""
    requires = _grad_enabled and any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=requires)
    if requires:
        out._parents, out._backward = parents, backward
    return out


def _is_advanced(index):
    items = index if isinstance(index, tuple) else (index,)
    return any(isinstance(i, (np.ndarray, list)) for i in items)


class Tensor:
    __array_ufunc__ = None  # 讓 ndarray + Tensor 交給 Tensor.__radd__ 處理

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = ()
        self._backward = None

    # ───────── 基本屬性 ─────────
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    size = property(lambda self: self.data.size)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        flag = ', requires_grad=True' if self.requires_grad else ''
        return f'Tensor({np.array2string(self.data, precision=4, threshold=20)}{flag})'

    def numpy(self):
        return self.data

    def item(self):
        return float(self.data)

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    # ───────── 反向傳播 ─────────
    def _accumulate(self, grad):
        if self.grad is None:
            # 葉節點（參數）必須複製，之後才能安全地就地修改（如梯度裁剪）
            self.grad = np.array(grad) if not self._parents else grad
        elif self._parents:
            self.grad = self.grad + grad
        else:
            self.grad += grad

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError('此 Tensor 不需要梯度（requires_grad=False 或在 no_grad 內建立）')
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError('非純量 Tensor 呼叫 backward() 需提供 grad')
            grad = np.ones_like(self.data)
        order, visited, stack = [], set(), [(self, False)]
        while stack:  # 迭代式 DFS，避免深計算圖造成遞迴過深
            node, done = stack.pop()
            if done:
                order.append(node)
            elif id(node) not in visited:
                visited.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents
                             if p.requires_grad and id(p) not in visited)
        self._accumulate(np.asarray(grad, dtype=np.float64))
        for node in reversed(order):
            if node._backward is None or node.grad is None:
                continue
            grads = node._backward(node.grad)
            for parent, g in zip(node._parents, grads):
                if g is not None and parent.requires_grad:
                    parent._accumulate(unbroadcast(g, parent.shape))
            node.grad = None  # 釋放中間節點的梯度與計算圖（同 torch 預設行為）
            node._parents, node._backward = (), None

    # ───────── 算術 ─────────
    def __add__(self, other):
        other = _wrap(other)
        return _make(self.data + other.data, (self, other), lambda g: (g, g))

    __radd__ = __add__

    def __sub__(self, other):
        other = _wrap(other)
        return _make(self.data - other.data, (self, other), lambda g: (g, -g))

    def __rsub__(self, other):
        return _wrap(other) - self

    def __mul__(self, other):
        other = _wrap(other)
        a, b = self.data, other.data
        return _make(a * b, (self, other), lambda g: (g * b, g * a))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _wrap(other)
        a, b = self.data, other.data
        return _make(a / b, (self, other), lambda g: (g / b, -g * a / b ** 2))

    def __rtruediv__(self, other):
        return _wrap(other) / self

    def __neg__(self):
        return _make(-self.data, (self,), lambda g: (-g,))

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), '只支援純量次方'
        a = self.data
        return _make(a ** exponent, (self,), lambda g: (g * exponent * a ** (exponent - 1),))

    def __matmul__(self, other):
        other = _wrap(other)
        a, b = self.data, other.data
        assert a.ndim >= 2 and b.ndim >= 2, 'matmul 需要至少 2 維'

        def backward(g):
            if b.ndim == 2 and a.ndim > 2:  # (B,T,C) @ (C,D)：攤平成單一矩陣乘法
                return g @ b.T, a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])
            return g @ np.swapaxes(b, -1, -2), np.swapaxes(a, -1, -2) @ g

        return _make(a @ b, (self, other), backward)

    def __rmatmul__(self, other):
        return _wrap(other) @ self

    # ───────── 逐元素函數 ─────────
    def exp(self):
        out = np.exp(self.data)
        return _make(out, (self,), lambda g: (g * out,))

    def log(self):
        a = self.data
        return _make(np.log(a), (self,), lambda g: (g / a,))

    def sqrt(self):
        return self ** 0.5

    def tanh(self):
        out = np.tanh(self.data)
        return _make(out, (self,), lambda g: (g * (1 - out ** 2),))

    def sigmoid(self):
        out = 0.5 * (1 + np.tanh(0.5 * self.data))  # 數值穩定
        return _make(out, (self,), lambda g: (g * out * (1 - out),))

    def relu(self):
        a = self.data
        return _make(np.maximum(a, 0), (self,), lambda g: (g * (a > 0),))

    # ───────── 縮減 ─────────
    def sum(self, axis=None, keepdims=False):
        shape = self.shape

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape),)

        return _make(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward)

    def mean(self, axis=None, keepdims=False):
        out = self.sum(axis=axis, keepdims=keepdims)
        return out * (out.size / self.size)

    def max(self, axis=None, keepdims=False):
        a = self.data
        peak = a.max(axis=axis, keepdims=True)
        mask = a == peak
        mask = mask / mask.sum(axis=axis, keepdims=True)  # 平手時平均分配梯度

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (mask * g,)

        out = peak if keepdims else (peak.squeeze(axis=axis) if axis is not None else peak.reshape(()))
        return _make(out, (self,), backward)

    def argmax(self, axis=None):
        return self.data.argmax(axis=axis)

    # ───────── 形狀 ─────────
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original = self.shape
        return _make(self.data.reshape(shape), (self,), lambda g: (g.reshape(original),))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = np.argsort(axes)
        return _make(self.data.transpose(axes), (self,), lambda g: (g.transpose(inverse),))

    T = property(lambda self: self.transpose())

    def __getitem__(self, index):
        shape, advanced = self.shape, _is_advanced(index)

        def backward(g):
            grad = np.zeros(shape)
            if advanced:
                np.add.at(grad, index, g)
            else:
                grad[index] += g
            return (grad,)

        return _make(self.data[index], (self,), backward)

    def masked_fill(self, mask, value):
        """mask 為布林 ndarray（可 broadcast），True 的位置填入 value，且不傳梯度。"""
        mask = np.asarray(mask, dtype=bool)
        return _make(np.where(mask, value, self.data), (self,), lambda g: (g * ~mask,))

    # ───────── softmax ─────────
    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exps = np.exp(shifted)
        out = exps / exps.sum(axis=axis, keepdims=True)
        return _make(out, (self,), lambda g: (out * (g - (g * out).sum(axis=axis, keepdims=True)),))

    def log_softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        return _make(out, (self,), lambda g: (g - np.exp(out) * g.sum(axis=axis, keepdims=True),))


class Parameter(Tensor):
    """可訓練參數：Module 會自動收集 Parameter。"""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


# ───────── 多 Tensor 算子與建構函式 ─────────

def stack(tensors, axis=0):
    tensors = [_wrap(t) for t in tensors]
    return _make(np.stack([t.data for t in tensors], axis=axis), tuple(tensors),
                 lambda g: tuple(np.take(g, i, axis=axis) for i in range(len(tensors))))


def concatenate(tensors, axis=0):
    tensors = [_wrap(t) for t in tensors]
    sizes = np.cumsum([t.shape[axis] for t in tensors])[:-1]
    return _make(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors),
                 lambda g: tuple(np.split(g, sizes, axis=axis)))


def as_tensor(value):
    return _wrap(value)


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad)


def zeros(*shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad)


def ones(*shape, requires_grad=False):
    return Tensor(np.ones(shape), requires_grad)


def randn(*shape, requires_grad=False):
    return Tensor(_rng.normal(size=shape), requires_grad)


def gradcheck(fn, inputs, eps=1e-6, tol=1e-6, max_checks=30, seed=0):
    """以中央差分數值梯度驗證 fn(*inputs) 的解析梯度；回傳最大誤差，超過 tol 就丟 AssertionError。"""
    rng = np.random.default_rng(seed)
    for t in inputs:
        t.requires_grad, t.grad = True, None
    out = fn(*inputs)
    weights = rng.normal(size=out.shape)  # 隨機加權，避免梯度剛好對稱而漏掉錯誤
    (out * weights).sum().backward()
    analytic = [np.zeros(t.shape) if t.grad is None else t.grad.copy() for t in inputs]

    def scalar():
        with no_grad():
            return float((fn(*inputs).data * weights).sum())

    worst = 0.0
    for t, grad in zip(inputs, analytic):
        indices = list(np.ndindex(*t.shape)) if t.shape else [()]
        if len(indices) > max_checks:
            indices = [indices[i] for i in rng.choice(len(indices), max_checks, replace=False)]
        for index in indices:
            old = t.data[index]
            t.data[index] = old + eps
            plus = scalar()
            t.data[index] = old - eps
            minus = scalar()
            t.data[index] = old
            numeric = (plus - minus) / (2 * eps)
            worst = max(worst, abs(numeric - grad[index]) / (1 + abs(numeric)))
    if worst > tol:
        raise AssertionError(f'梯度不一致：最大誤差 {worst:.3e} > {tol:.0e}')
    return worst
