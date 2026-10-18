"""PyTorch 模型：MLP、CNN、GRU 語言模型、GPT、擴散模型用的 UNet。"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# ───────── 分類模型 ─────────

def _init_he(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=(128, 64), num_classes=10):
        super().__init__()
        sizes = [input_size, *hidden_sizes, num_classes]
        layers = [nn.Flatten()]
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(fan_in, fan_out))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.apply(_init_he)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """conv3x3 → ReLU → pool → conv3x3 → ReLU → pool → 全連接。"""

    def __init__(self, in_channels=1, num_classes=10, image_size=28, channels1=8, channels2=16):
        super().__init__()
        size = ((image_size - 2) // 2 - 2) // 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels1, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels1, channels2, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.fc = nn.Linear(channels2 * size * size, num_classes)
        self.apply(_init_he)

    def forward(self, x):
        return self.fc(self.features(x))


# ───────── 語言模型 ─────────

def _init_normal(module, std):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, 0.0, std)
        if getattr(module, 'bias', None) is not None:
            nn.init.zeros_(module.bias)


class GRULanguageModel(nn.Module):
    """輸入 token id (B,T)，輸出 logits (B,T,V)。"""

    def __init__(self, vocab_size, block_size, width=32, std=0.1):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.apply(lambda m: _init_normal(m, std))

    def forward(self, token_ids):
        hidden, _ = self.gru(self.wte(token_ids))
        return self.lm_head(hidden)


class CausalSelfAttention(nn.Module):
    def __init__(self, width, n_head, dropout):
        super().__init__()
        assert width % n_head == 0, 'width 必須能被 n_head 整除'
        self.n_head, self.dropout = n_head, dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=2)
        q, k, v = (t.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                             is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, width, n_head, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, n_head, dropout)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        return x + self.dropout(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    """多層、多頭、pre-LayerNorm 的 causal Transformer。"""

    def __init__(self, vocab_size, block_size, width=32, n_layer=2, n_head=2, dropout=0.0, std=0.08):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, width)
        self.wpe = nn.Embedding(block_size, width)
        self.blocks = nn.ModuleList(TransformerBlock(width, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.apply(lambda m: _init_normal(m, std))

    def forward(self, token_ids):
        length = token_ids.shape[1]
        if length > self.block_size:
            raise ValueError(f'序列長度不可超過 block_size={self.block_size}')
        x = self.wte(token_ids) + self.wpe(torch.arange(length, device=token_ids.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# ───────── 擴散模型用的 UNet ─────────

def timestep_embedding(t, dim):
    """正弦位置編碼：整數時間步 (B,) → (B, dim)。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class UNetBlock(nn.Module):
    """conv → 加上條件（時間 + 類別）→ SiLU → conv → SiLU。"""

    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x, emb):
        h = F.silu(self.conv1(x) + self.emb_proj(emb)[:, :, None, None])
        return F.silu(self.conv2(h))


class UNet(nn.Module):
    """小型條件式 UNet，預測加在影像上的雜訊。輸入 (B,1,H,W)（H、W 需為偶數）、時間步 t (B,)、類別 labels (B,)。"""

    def __init__(self, num_classes, channels=8, in_channels=1, emb_dim=32):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.down = UNetBlock(channels, channels, emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.mid1 = UNetBlock(channels, 2 * channels, emb_dim)
        self.mid2 = UNetBlock(2 * channels, 2 * channels, emb_dim)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block = UNetBlock(3 * channels, channels, emb_dim)  # 上採樣的 2c + skip 的 c
        self.conv_out = nn.Conv2d(channels, in_channels, 1)

    def forward(self, x, t, labels):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.class_emb(labels)
        h = F.silu(self.conv_in(x))
        skip = self.down(h, emb)
        h = self.mid2(self.mid1(self.pool(skip), emb), emb)
        h = self.up_block(torch.cat([self.up(h), skip], dim=1), emb)
        return self.conv_out(h)
