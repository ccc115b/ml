"""Module 系統與常用層：Linear、Embedding、Conv2d、LayerNorm、GRU、注意力等。

權重 shape 慣例：Linear 為 (in, out)（x @ weight），Conv2d 為 (out, in, k, k)。
"""

import numpy as np
from . import functional as F
from . import init
from .tensor import Parameter, Tensor, as_tensor, stack


class Module:
    """自動收集子模組與 Parameter；提供 train/eval、state_dict、存檔與載入。"""

    training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix=''):
        seen = set()

        def walk(module, path):
            for name, p in module.__dict__.get('_parameters', {}).items():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield path + name, p
            for name, child in module.__dict__.get('_modules', {}).items():
                yield from walk(child, f'{path}{name}.')

        yield from walk(self, prefix)

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        self.training = mode
        for child in self.__dict__.get('_modules', {}).values():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        own = dict(self.named_parameters())
        if set(own) != set(state):
            raise KeyError(f'參數名稱不符：缺少 {sorted(set(own) - set(state))}，多出 {sorted(set(state) - set(own))}')
        for name, p in own.items():
            if p.shape != state[name].shape:
                raise ValueError(f'{name} 的 shape 不符：{p.shape} vs {state[name].shape}')
            p.data[...] = state[name]

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        with np.load(path) as data:
            self.load_state_dict({name: data[name] for name in data.files})
        return self


class Sequential(Module):
    def __init__(self, *modules):
        for i, module in enumerate(modules):
            setattr(self, str(i), module)
        self.length = len(modules)

    def forward(self, x):
        for i in range(self.length):
            x = getattr(self, str(i))(x)
        return x

    def __getitem__(self, i):
        return getattr(self, str(i % self.length))

    def __len__(self):
        return self.length


class ModuleList(Module):
    def __init__(self, modules=()):
        self.length = 0
        for module in modules:
            self.append(module)

    def append(self, module):
        setattr(self, str(self.length), module)
        self.length += 1

    def __getitem__(self, i):
        return getattr(self, str(i % self.length))

    def __len__(self):
        return self.length

    def __iter__(self):
        return (getattr(self, str(i)) for i in range(self.length))


class Linear(Module):
    """y = x @ weight + bias。init: 'kaiming'（預設）、'xavier'，或一個數字代表 normal 標準差。"""

    def __init__(self, in_features, out_features, bias=True, init_scheme='kaiming'):
        shape = (in_features, out_features)
        if init_scheme == 'kaiming':
            weight = init.kaiming_normal(shape, in_features)
        elif init_scheme == 'xavier':
            weight = init.xavier_uniform(shape, in_features, out_features)
        else:
            weight = init.normal(shape, init_scheme)
        self.weight = Parameter(weight)
        self.bias = Parameter(init.zeros(out_features)) if bias else None

    def forward(self, x):
        out = as_tensor(x) @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    def __init__(self, num_embeddings, dim, std=0.08):
        self.weight = Parameter(init.normal((num_embeddings, dim), std))

    def forward(self, indices):
        return self.weight[np.asarray(indices, dtype=np.int64)]


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, bias=True):
        self.padding = padding
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter(init.kaiming_normal(shape, in_channels * kernel_size ** 2))
        self.bias = Parameter(init.zeros(out_channels)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size)


class Flatten(Module):
    def forward(self, x):
        x = as_tensor(x)
        return x.reshape(x.shape[0], -1)


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class SiLU(Module):
    def forward(self, x):
        return F.silu(x)


class Upsample(Module):
    """最近鄰 2 倍上取樣。"""

    def forward(self, x):
        return F.upsample2x(x)


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

    def forward(self, x):
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = (centered ** 2).mean(axis=-1, keepdims=True)
        return centered / (variance + self.eps) ** 0.5 * self.weight + self.bias


# ───────── 序列模型 ─────────

class GRUCell(Module):
    """z = σ(xWz + hUz + bz)；r = σ(xWr + hUr + br)；n = tanh(xWn + (r*h)Un + bn)；h' = (1-z)h + zn。
    三組 x 權重合併成 w_x (I,3H)，z、r 的 h 權重合併成 w_h (H,2H)。"""

    def __init__(self, input_size, hidden_size, std=0.1):
        self.hidden_size = hidden_size
        self.w_x = Parameter(init.normal((input_size, 3 * hidden_size), std))
        self.w_h = Parameter(init.normal((hidden_size, 2 * hidden_size), std))
        self.u_n = Parameter(init.normal((hidden_size, hidden_size), std))
        self.bias = Parameter(init.zeros(3 * hidden_size))

    def forward(self, x, h):
        H = self.hidden_size
        gx = x @ self.w_x + self.bias
        gh = h @ self.w_h
        z = (gx[:, :H] + gh[:, :H]).sigmoid()
        r = (gx[:, H:2 * H] + gh[:, H:]).sigmoid()
        n = (gx[:, 2 * H:] + (r * h) @ self.u_n).tanh()
        return (1 - z) * h + z * n


class GRU(Module):
    """輸入 (B,T,I)，回傳 (所有時間步的輸出 (B,T,H), 最後狀態 (B,H))。"""

    def __init__(self, input_size, hidden_size, std=0.1):
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size, std)

    def forward(self, x, h=None):
        batch, length, _ = x.shape
        if h is None:
            h = Tensor(np.zeros((batch, self.hidden_size)))
        outputs = []
        for t in range(length):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        return stack(outputs, axis=1), h


class CausalSelfAttention(Module):
    """多頭因果自注意力。輸入 (B,T,C)。"""

    def __init__(self, width, n_head, block_size, dropout=0.0, std=0.08):
        assert width % n_head == 0, 'width 必須能被 n_head 整除'
        self.n_head = n_head
        self.qkv = Linear(width, 3 * width, init_scheme=std)
        self.proj = Linear(width, width, init_scheme=std)
        self.attn_dropout = Dropout(dropout)
        self.future = np.triu(np.ones((block_size, block_size), dtype=bool), k=1)

    def forward(self, x):
        batch, length, width = x.shape
        heads, dim = self.n_head, width // self.n_head
        qkv = self.qkv(x)
        split = lambda i: qkv[:, :, i * width:(i + 1) * width].reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)
        q, k, v = split(0), split(1), split(2)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1 / np.sqrt(dim))
        attention = self.attn_dropout(scores.masked_fill(self.future[:length, :length], -1e9).softmax(-1))
        out = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.proj(out)
