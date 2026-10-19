"""以 minitorch 組成的模型：MLP、CNN、GRU 語言模型、GPT。不需要手寫任何反向傳播。"""

import numpy as np
from . import functional as F
from . import layers as L
from .tensor import Tensor, as_tensor, concatenate, no_grad


class Classifier(L.Module):
    """分類模型共用：predict / accuracy。子類別需實作 forward(images) -> logits (N, 類別數)。"""

    def predict(self, images, batch_size=500):
        was_training = self.training
        self.eval()
        with no_grad():
            labels = np.concatenate([self(images[i:i + batch_size]).argmax(axis=1)
                                     for i in range(0, len(images), batch_size)])
        self.train(was_training)
        return labels

    def accuracy(self, images, labels):
        return float((self.predict(images) == np.asarray(labels)).mean())


class MLP(Classifier):
    def __init__(self, input_size=784, hidden_sizes=(128, 64), num_classes=10):
        sizes = [input_size, *hidden_sizes, num_classes]
        modules = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            modules.append(L.Linear(fan_in, fan_out))
            if i < len(sizes) - 2:
                modules.append(L.ReLU())
        self.net = L.Sequential(L.Flatten(), *modules)

    def forward(self, x):
        return self.net(x)


class CNN(Classifier):
    """conv3x3 → ReLU → pool → conv3x3 → ReLU → pool → 全連接。"""

    def __init__(self, in_channels=1, num_classes=10, image_size=28, channels1=8, channels2=16):
        size = ((image_size - 2) // 2 - 2) // 2
        self.conv1 = L.Conv2d(in_channels, channels1, 3)
        self.conv2 = L.Conv2d(channels1, channels2, 3)
        self.pool = L.MaxPool2d(2)
        self.fc = L.Linear(channels2 * size * size, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(as_tensor(x)).relu())
        x = self.pool(self.conv2(x).relu())
        return self.fc(x.reshape(x.shape[0], -1))


class LanguageModel(L.Module):
    """語言模型共用：輸入 (B,T) 或 (T,) 的 token id，輸出 logits (B,T,V) 或 (T,V)。"""

    def forward(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim == 1:
            return self.forward_batch(token_ids[None])[0]
        return self.forward_batch(token_ids)


class GRULanguageModel(LanguageModel):
    def __init__(self, vocab_size, block_size, width=32, std=0.1):
        self.block_size = block_size
        self.wte = L.Embedding(vocab_size, width, std)
        self.gru = L.GRU(width, width, std)
        self.lm_head = L.Linear(width, vocab_size, bias=False, init_scheme=std)

    def forward_batch(self, token_ids):
        hidden, _ = self.gru(self.wte(token_ids))
        return self.lm_head(hidden)


class TransformerBlock(L.Module):
    def __init__(self, width, n_head, block_size, dropout, std):
        self.ln1 = L.LayerNorm(width)
        self.attn = L.CausalSelfAttention(width, n_head, block_size, dropout, std)
        self.ln2 = L.LayerNorm(width)
        self.fc1 = L.Linear(width, 4 * width, init_scheme=std)
        self.fc2 = L.Linear(4 * width, width, init_scheme=std)
        self.dropout = L.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        return x + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class GPT(LanguageModel):
    """多層、多頭、pre-LayerNorm 的 causal Transformer。"""

    def __init__(self, vocab_size, block_size, width=32, n_layer=2, n_head=2, dropout=0.0, std=0.08):
        self.config = dict(vocab_size=vocab_size, block_size=block_size, width=width, n_layer=n_layer,
                           n_head=n_head, dropout=dropout, std=std)  # 存檔時用來重建相同結構
        self.block_size = block_size
        self.wte = L.Embedding(vocab_size, width, std)
        self.wpe = L.Embedding(block_size, width, std)
        self.blocks = L.ModuleList(TransformerBlock(width, n_head, block_size, dropout, std)
                                   for _ in range(n_layer))
        self.ln_f = L.LayerNorm(width)
        self.lm_head = L.Linear(width, vocab_size, bias=False, init_scheme=std)

    def forward_batch(self, token_ids):
        length = token_ids.shape[1]
        if length > self.block_size:
            raise ValueError(f'序列長度不可超過 block_size={self.block_size}')
        x = self.wte(token_ids) + self.wpe(np.arange(length))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# ───────── 擴散模型用的 UNet ─────────

def timestep_embedding(t, dim):
    """正弦位置編碼：整數時間步 (B,) → (B, dim)。這是常數，不需要梯度。"""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    angles = np.asarray(t, dtype=np.float64)[:, None] * freqs[None]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


class UNetBlock(L.Module):
    """conv → 加上條件（時間 + 類別）→ SiLU → conv → SiLU。"""

    def __init__(self, in_channels, out_channels, emb_dim):
        self.conv1 = L.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = L.Conv2d(out_channels, out_channels, 3, padding=1)
        self.emb_proj = L.Linear(emb_dim, out_channels)

    def forward(self, x, emb):
        cond = self.emb_proj(emb)
        h = F.silu(self.conv1(x) + cond.reshape(cond.shape[0], -1, 1, 1))
        return F.silu(self.conv2(h))


class UNet(L.Module):
    """小型條件式 UNet，預測加在影像上的雜訊。輸入 (B,1,H,W)（H、W 需為偶數）、時間步 t (B,)、類別 labels (B,)。
    結構：下採樣一次、中間層加寬、上採樣後與 skip 連接，最後 1×1 卷積輸出。"""

    def __init__(self, num_classes, channels=8, in_channels=1, emb_dim=32):
        self.emb_dim = emb_dim
        self.time_mlp = L.Sequential(L.Linear(emb_dim, emb_dim), L.SiLU(), L.Linear(emb_dim, emb_dim))
        self.class_emb = L.Embedding(num_classes, emb_dim, std=0.1)
        self.conv_in = L.Conv2d(in_channels, channels, 3, padding=1)
        self.down = UNetBlock(channels, channels, emb_dim)
        self.pool = L.MaxPool2d(2)
        self.mid1 = UNetBlock(channels, 2 * channels, emb_dim)
        self.mid2 = UNetBlock(2 * channels, 2 * channels, emb_dim)
        self.up = L.Upsample()
        self.up_block = UNetBlock(3 * channels, channels, emb_dim)  # 上採樣的 2c + skip 的 c
        self.conv_out = L.Conv2d(channels, in_channels, 1)

    def forward(self, x, t, labels):
        emb = self.time_mlp(Tensor(timestep_embedding(t, self.emb_dim))) + self.class_emb(labels)
        h = F.silu(self.conv_in(as_tensor(x)))
        skip = self.down(h, emb)
        h = self.mid2(self.mid1(self.pool(skip), emb), emb)
        h = self.up_block(concatenate([self.up(h), skip], axis=1), emb)
        return self.conv_out(h)
