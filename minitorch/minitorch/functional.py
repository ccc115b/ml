"""函數式算子：激活函數、損失函數、卷積與池化（皆支援 autograd）。"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .tensor import Tensor, _make, _wrap, get_rng


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def gelu(x):
    return 0.5 * x * (1 + (0.7978845608028654 * (x + 0.044715 * x ** 3)).tanh())


def silu(x):
    return x * x.sigmoid()


def softmax(x, axis=-1):
    return x.softmax(axis)


def log_softmax(x, axis=-1):
    return x.log_softmax(axis)


def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    mask = (get_rng().random(x.shape) >= p) / (1 - p)
    return x * mask


def cross_entropy(logits, targets, ignore_index=-100, reduction='mean'):
    """logits (...,C)、targets 為整數 ndarray（形狀同 logits 去掉最後一維）。
    targets 等於 ignore_index 的位置不計入損失（用於 padding）。"""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    valid = targets != ignore_index
    safe = np.where(valid, targets, 0)
    nll = -flat.log_softmax(-1)[np.arange(len(safe)), safe] * valid.astype(np.float64)
    if reduction == 'none':
        return nll
    total = nll.sum()
    return total if reduction == 'sum' else total * (1.0 / max(int(valid.sum()), 1))


def mse_loss(prediction, target):
    return ((prediction - target) ** 2).mean()


# ───────── 卷積與池化（im2col） ─────────

def _conv_forward(x, weight, bias):
    filters, channels, k, _ = weight.shape
    windows = sliding_window_view(x, (k, k), axis=(2, 3))  # (N,C,Ho,Wo,k,k)
    n, _, out_h, out_w = windows.shape[:4]
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * out_h * out_w, channels * k * k)
    out = cols @ weight.reshape(filters, -1).T + bias
    return out.reshape(n, out_h, out_w, filters).transpose(0, 3, 1, 2), cols


def _conv_backward(d_out, cols, x_shape, weight):
    filters, channels, k, _ = weight.shape
    n, _, out_h, out_w = d_out.shape
    d_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, filters)
    d_weight = (d_flat.T @ cols).reshape(weight.shape)
    d_cols = (d_flat @ weight.reshape(filters, -1)).reshape(n, out_h, out_w, channels, k, k)
    d_x = np.zeros(x_shape)
    for i in range(k):
        for j in range(k):
            d_x[:, :, i:i + out_h, j:j + out_w] += d_cols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return d_x, d_weight, d_flat.sum(axis=0)


def conv2d(x, weight, bias=None, padding=0):
    """x (N,C,H,W)、weight (F,C,k,k)，stride 固定為 1。"""
    x, weight = _wrap(x), _wrap(weight)
    padded = np.pad(x.data, ((0, 0), (0, 0), (padding,) * 2, (padding,) * 2)) if padding else x.data
    out, cols = _conv_forward(padded, weight.data, bias.data if bias is not None else 0.0)

    def backward(g):
        d_x, d_weight, d_bias = _conv_backward(g, cols, padded.shape, weight.data)
        if padding:
            d_x = d_x[:, :, padding:-padding, padding:-padding]
        return (d_x, d_weight, d_bias) if bias is not None else (d_x, d_weight)

    parents = (x, weight) + ((bias,) if bias is not None else ())
    return _make(out, parents, backward)


def max_pool2d(x, kernel_size=2):
    """stride = kernel_size；無法整除的邊緣會被捨棄。"""
    k = kernel_size
    n, c, h, w = x.shape
    h2, w2 = h // k, w // k
    blocks = x.data[:, :, :h2 * k, :w2 * k].reshape(n, c, h2, k, w2, k)
    out = blocks.max(axis=(3, 5))
    mask = blocks == out[:, :, :, None, :, None]
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)  # 平手時平均分配梯度

    def backward(g):
        d_x = np.zeros(x.shape)
        d_x[:, :, :h2 * k, :w2 * k] = (mask * g[:, :, :, None, :, None]).reshape(n, c, h2 * k, w2 * k)
        return (d_x,)

    return _make(out, (x,), backward)


def upsample2x(x):
    """最近鄰 2 倍上取樣：(N,C,H,W) → (N,C,2H,2W)；反向傳播是把 2×2 區塊的梯度加總。"""
    n, c, h, w = x.shape
    out = np.repeat(np.repeat(x.data, 2, axis=2), 2, axis=3)
    return _make(out, (x,), lambda g: (g.reshape(n, c, h, 2, w, 2).sum(axis=(3, 5)),))
