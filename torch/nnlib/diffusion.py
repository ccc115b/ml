"""DDPM：噪聲排程、加噪（forward process）、訓練損失與逐步去噪取樣（reverse process）。

模型介面：model(x_t, t, labels) -> 預測的雜訊（與 x_t 同 shape）。影像值域為 [-1, 1]。
"""

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        scale = 1000 / T  # 步數少時等比放大 β，讓最後仍接近純雜訊
        self.T, self.device = T, device
        betas = torch.linspace(beta_start * scale, beta_end * scale, T, dtype=torch.float64)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
        to = lambda x: x.float().to(device)
        self.betas = to(betas)
        self.sqrt_alpha_bars = to(alpha_bars.sqrt())
        self.sqrt_one_minus_alpha_bars = to((1 - alpha_bars).sqrt())
        # 後驗 q(x_{t-1} | x_t, x_0) 的係數
        self.posterior_coef_x0 = to(betas * previous.sqrt() / (1 - alpha_bars))
        self.posterior_coef_xt = to((1 - previous) * alphas.sqrt() / (1 - alpha_bars))
        self.posterior_std = to((betas * (1 - previous) / (1 - alpha_bars)).sqrt())

    def q_sample(self, x0, t, noise):
        """x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε，t 為每張圖各自的時間步 (B,)。"""
        shape = (-1,) + (1,) * (x0.dim() - 1)
        return self.sqrt_alpha_bars[t].view(shape) * x0 + self.sqrt_one_minus_alpha_bars[t].view(shape) * noise

    def loss(self, model, x0, labels):
        """隨機挑時間步與雜訊，讓模型預測雜訊；損失是預測與真實雜訊的 MSE。"""
        t = torch.randint(0, self.T, (len(x0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t, labels), noise)

    @torch.no_grad()
    def sample(self, model, labels, image_shape, clip=True):
        """從純雜訊開始，逐步去噪 T 次；回傳 (N, *image_shape) 的 tensor。"""
        model.eval()
        n = len(labels)
        x = torch.randn(n, *image_shape, device=labels.device)
        for t in reversed(range(self.T)):
            eps = model(x, torch.full((n,), t, device=x.device, dtype=torch.long), labels)
            x0 = (x - self.sqrt_one_minus_alpha_bars[t] * eps) / self.sqrt_alpha_bars[t]
            if clip:
                x0 = x0.clamp(-1, 1)  # 影像值域已知，修正預測可穩定取樣
            x = self.posterior_coef_x0[t] * x0 + self.posterior_coef_xt[t] * x
            if t > 0:
                x = x + self.posterior_std[t] * torch.randn_like(x)
        return x
