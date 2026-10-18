"""權重初始化：回傳 ndarray，使用 tensor.get_rng() 的全域亂數（可由 manual_seed 重現）。"""

import numpy as np
from .tensor import get_rng


def normal(shape, std=0.02):
    return get_rng().normal(0, std, shape)


def kaiming_normal(shape, fan_in):
    return get_rng().normal(0, np.sqrt(2 / fan_in), shape)  # 適合 ReLU


def xavier_uniform(shape, fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return get_rng().uniform(-limit, limit, shape)


def zeros(shape):
    return np.zeros(shape)


def ones(shape):
    return np.ones(shape)
