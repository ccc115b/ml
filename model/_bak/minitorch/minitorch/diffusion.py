"""DDPM：噪聲排程、加噪（forward process）、訓練損失與逐步去噪取樣（reverse process）。

模型介面：model(x_t, t, labels) -> 預測的雜訊（與 x_t 同 shape 的 Tensor）。影像值域為 [-1, 1]。
"""

import numpy as np
from . import functional as F
from .tensor import get_rng, no_grad


class Diffusion:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        scale = 1000 / T  # 步數少時等比放大 β，讓最後仍接近純雜訊
        self.T = T
        self.betas = np.linspace(beta_start * scale, beta_end * scale, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        previous = np.concatenate([[1.0], self.alpha_bars[:-1]])
        # 後驗 q(x_{t-1} | x_t, x_0) 的係數
        self.posterior_coef_x0 = self.betas * np.sqrt(previous) / (1 - self.alpha_bars)
        self.posterior_coef_xt = (1 - previous) * np.sqrt(self.alphas) / (1 - self.alpha_bars)
        self.posterior_var = self.betas * (1 - previous) / (1 - self.alpha_bars)

    def q_sample(self, x0, t, noise):
        """x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε，t 為每張圖各自的時間步 (B,)。"""
        shape = (-1,) + (1,) * (x0.ndim - 1)
        return (np.sqrt(self.alpha_bars[t]).reshape(shape) * x0
                + np.sqrt(1 - self.alpha_bars[t]).reshape(shape) * noise)

    def loss(self, model, x0, labels):
        """隨機挑時間步與雜訊，讓模型預測雜訊；損失是預測與真實雜訊的 MSE。"""
        x0, rng = np.asarray(x0, dtype=np.float64), get_rng()
        t = rng.integers(0, self.T, len(x0))
        noise = rng.normal(size=x0.shape)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t, labels), noise)

    def sample(self, model, labels, image_shape, clip=True):
        """從純雜訊開始，逐步去噪 T 次；回傳 (N, *image_shape) 的 ndarray。"""
        labels = np.asarray(labels, dtype=np.int64)
        n, rng = len(labels), get_rng()
        x = rng.normal(size=(n, *image_shape))
        with no_grad():
            for t in reversed(range(self.T)):
                eps = model(x, np.full(n, t), labels).data
                x0 = (x - np.sqrt(1 - self.alpha_bars[t]) * eps) / np.sqrt(self.alpha_bars[t])
                if clip:
                    x0 = np.clip(x0, -1, 1)  # 影像值域已知，修正預測可穩定取樣
                x = self.posterior_coef_x0[t] * x0 + self.posterior_coef_xt[t] * x
                if t > 0:
                    x = x + np.sqrt(self.posterior_var[t]) * rng.normal(size=x.shape)
        return x
