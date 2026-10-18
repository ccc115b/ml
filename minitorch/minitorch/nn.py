"""NumPy tensor 版的極簡神經網路（GPT、GRU、MLP、CNN）：手寫矩陣反向傳播，不建立純量運算圖。

共用的 softmax / Adam / 訓練 / 取樣邏輯在 common.py，資料載入在 data.py。"""

import numpy as np
from .common import Adam, softmax  # Adam 在此重新匯出，nn.Adam 仍可使用


class GPT:
    """單層、單頭的 causal Transformer，所有權重梯度皆以 tensor 計算。"""

    def __init__(self, vocab_size, block_size, width=24, seed=42):
        self.block_size = block_size
        self.width = width
        rng = np.random.default_rng(seed)
        normal = lambda *shape: rng.normal(0, 0.08, shape)
        self.parameters = {
            'wte': normal(vocab_size, width),
            'wpe': normal(block_size, width),
            'wq': normal(width, width),
            'wk': normal(width, width),
            'wv': normal(width, width),
            'wo': normal(width, width),
            'w1': normal(width, 4 * width),
            'w2': normal(4 * width, width),
            'lm_head': normal(width, vocab_size),
        }

    def forward(self, token_ids, keep_cache=False):
        """輸入 shape (T,)，輸出每個位置的下一個字元 logits，shape (T, V)。"""
        p = self.parameters
        token_ids = np.asarray(token_ids, dtype=np.int64)
        length = len(token_ids)
        if length > self.block_size:
            raise ValueError(f'序列長度不可超過 block_size={self.block_size}')

        x = p['wte'][token_ids] + p['wpe'][:length]
        q, k, v = x @ p['wq'], x @ p['wk'], x @ p['wv']
        scores = q @ k.T / np.sqrt(self.width)
        future = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores[future] = -1e9
        attention = softmax(scores)
        attended = attention @ v
        x2 = x + attended @ p['wo']
        pre_activation = x2 @ p['w1']
        hidden = np.maximum(pre_activation, 0)
        x3 = x2 + hidden @ p['w2']
        logits = x3 @ p['lm_head']

        if not keep_cache:
            return logits
        return logits, (token_ids, x, q, k, v, future, attention, attended,
                        x2, pre_activation, hidden, x3)

    def loss_and_gradients(self, token_ids, targets):
        """交叉熵損失與全部參數的解析梯度。"""
        p = self.parameters
        logits, cache = self.forward(token_ids, keep_cache=True)
        token_ids, x, q, k, v, future, attention, attended, x2, pre, hidden, x3 = cache
        targets = np.asarray(targets, dtype=np.int64)
        length = len(token_ids)

        probabilities = softmax(logits)
        loss = -np.log(probabilities[np.arange(length), targets] + 1e-12).mean()
        d_logits = probabilities.copy()
        d_logits[np.arange(length), targets] -= 1
        d_logits /= length
        gradients = {name: np.zeros_like(value) for name, value in p.items()}

        gradients['lm_head'] = x3.T @ d_logits
        d_x3 = d_logits @ p['lm_head'].T

        d_x2 = d_x3.copy()  # MLP 殘差連接
        gradients['w2'] = hidden.T @ d_x3
        d_hidden = d_x3 @ p['w2'].T
        d_pre = d_hidden * (pre > 0)
        gradients['w1'] = x2.T @ d_pre
        d_x2 += d_pre @ p['w1'].T

        d_x = d_x2.copy()  # attention 殘差連接
        gradients['wo'] = attended.T @ d_x2
        d_attended = d_x2 @ p['wo'].T
        d_attention = d_attended @ v.T
        d_v = attention.T @ d_attended
        # softmax Jacobian-vector product，逐列計算即可。
        d_scores = attention * (d_attention - (d_attention * attention).sum(axis=1, keepdims=True))
        d_scores[future] = 0
        scale = 1 / np.sqrt(self.width)
        d_q = d_scores @ k * scale
        d_k = d_scores.T @ q * scale
        gradients['wq'] = x.T @ d_q
        gradients['wk'] = x.T @ d_k
        gradients['wv'] = x.T @ d_v
        d_x += d_q @ p['wq'].T + d_k @ p['wk'].T + d_v @ p['wv'].T

        np.add.at(gradients['wte'], token_ids, d_x)
        gradients['wpe'][:length] += d_x
        return float(loss), gradients


# ───────────────────────── GRU ─────────────────────────


def _sigmoid(values):
    return 1 / (1 + np.exp(-values))


class GRU:
    """單層 GRU 字元語言模型（RNN 的改良版），以 BPTT 手寫 tensor 梯度。

    z = σ(x·wz + h·uz + bz)          更新閘：新舊狀態的混合比例
    r = σ(x·wr + h·ur + br)          重置閘：決定要看多少過去的狀態
    n = tanh(x·wn + (r*h)·un + bn)   候選狀態
    h' = (1 - z) * h + z * n
    """

    def __init__(self, vocab_size, block_size, width=24, seed=42):
        self.block_size = block_size
        self.width = width
        rng = np.random.default_rng(seed)
        normal = lambda *shape: rng.normal(0, 0.1, shape)
        zeros = lambda *shape: np.zeros(shape)
        self.parameters = {
            'wte': normal(vocab_size, width),
            'wz': normal(width, width), 'uz': normal(width, width), 'bz': zeros(width),
            'wr': normal(width, width), 'ur': normal(width, width), 'br': zeros(width),
            'wn': normal(width, width), 'un': normal(width, width), 'bn': zeros(width),
            'lm_head': normal(width, vocab_size),
        }

    def forward(self, token_ids, keep_cache=False):
        """輸入 shape (T,)，輸出每個位置的下一個字元 logits，shape (T, V)。"""
        p = self.parameters
        token_ids = np.asarray(token_ids, dtype=np.int64)
        length = len(token_ids)
        if length > self.block_size:
            raise ValueError(f'序列長度不可超過 block_size={self.block_size}')

        x = p['wte'][token_ids]
        h = np.zeros(self.width)
        hs, steps = np.empty((length, self.width)), []
        for t in range(length):
            z = _sigmoid(x[t] @ p['wz'] + h @ p['uz'] + p['bz'])
            r = _sigmoid(x[t] @ p['wr'] + h @ p['ur'] + p['br'])
            n = np.tanh(x[t] @ p['wn'] + (r * h) @ p['un'] + p['bn'])
            steps.append((h, z, r, n))
            h = (1 - z) * h + z * n
            hs[t] = h
        logits = hs @ p['lm_head']

        if not keep_cache:
            return logits
        return logits, (token_ids, x, hs, steps)

    def loss_and_gradients(self, token_ids, targets):
        """交叉熵損失與全部參數的解析梯度（BPTT）。"""
        p = self.parameters
        logits, (token_ids, x, hs, steps) = self.forward(token_ids, keep_cache=True)
        targets = np.asarray(targets, dtype=np.int64)
        length = len(token_ids)

        probabilities = softmax(logits)
        loss = -np.log(probabilities[np.arange(length), targets] + 1e-12).mean()
        d_logits = probabilities.copy()
        d_logits[np.arange(length), targets] -= 1
        d_logits /= length
        gradients = {name: np.zeros_like(value) for name, value in p.items()}

        gradients['lm_head'] = hs.T @ d_logits
        d_hs = d_logits @ p['lm_head'].T
        d_x = np.zeros_like(x)
        d_h_next = np.zeros(self.width)

        for t in reversed(range(length)):
            h_prev, z, r, n = steps[t]
            d_h = d_hs[t] + d_h_next
            d_z = d_h * (n - h_prev)
            d_n = d_h * z
            d_h_prev = d_h * (1 - z)

            d_n_pre = d_n * (1 - n ** 2)
            gradients['wn'] += np.outer(x[t], d_n_pre)
            gradients['un'] += np.outer(r * h_prev, d_n_pre)
            gradients['bn'] += d_n_pre
            d_rh = d_n_pre @ p['un'].T
            d_r = d_rh * h_prev
            d_h_prev += d_rh * r

            d_z_pre = d_z * z * (1 - z)
            d_r_pre = d_r * r * (1 - r)
            gradients['wz'] += np.outer(x[t], d_z_pre)
            gradients['uz'] += np.outer(h_prev, d_z_pre)
            gradients['bz'] += d_z_pre
            gradients['wr'] += np.outer(x[t], d_r_pre)
            gradients['ur'] += np.outer(h_prev, d_r_pre)
            gradients['br'] += d_r_pre
            d_h_prev += d_z_pre @ p['uz'].T + d_r_pre @ p['ur'].T
            d_x[t] = d_z_pre @ p['wz'].T + d_r_pre @ p['wr'].T + d_n_pre @ p['wn'].T
            d_h_next = d_h_prev

        np.add.at(gradients['wte'], token_ids, d_x)
        return float(loss), gradients


# ───────────────────────── 分類模型：MLP / CNN（MNIST） ─────────────────────────


class _ClassifierMixin:
    """分類模型共用：需有 forward(images) -> logits (N, 類別數)。"""

    def predict(self, images, batch_size=500):
        return np.concatenate([self.forward(images[i:i + batch_size]).argmax(axis=1)
                               for i in range(0, len(images), batch_size)])

    def accuracy(self, images, labels):
        return float((self.predict(images) == np.asarray(labels)).mean())




from numpy.lib.stride_tricks import sliding_window_view


def _conv_forward(x, weight, bias):
    """x: (N,C,H,W)，weight: (F,C,k,k)，輸出 (N,F,H-k+1,W-k+1)。以 im2col 轉成一次矩陣乘法。"""
    filters, channels, k, _ = weight.shape
    windows = sliding_window_view(x, (k, k), axis=(2, 3))  # (N,C,Ho,Wo,k,k)
    n, _, out_h, out_w = windows.shape[:4]
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * out_h * out_w, channels * k * k)
    out = cols @ weight.reshape(filters, -1).T + bias
    out = out.reshape(n, out_h, out_w, filters).transpose(0, 3, 1, 2)
    return out, cols


def _conv_backward(d_out, cols, x_shape, weight):
    """回傳 (d_x, d_weight, d_bias)。"""
    filters, channels, k, _ = weight.shape
    n, _, out_h, out_w = d_out.shape
    d_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, filters)
    d_weight = (d_flat.T @ cols).reshape(weight.shape)
    d_bias = d_flat.sum(axis=0)
    d_cols = (d_flat @ weight.reshape(filters, -1)).reshape(n, out_h, out_w, channels, k, k)
    d_x = np.zeros(x_shape)
    for i in range(k):  # col2im：只需迴圈 k*k 次
        for j in range(k):
            d_x[:, :, i:i + out_h, j:j + out_w] += d_cols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return d_x, d_weight, d_bias


def _maxpool_forward(x):
    """2x2 max pooling，stride 2；奇數邊長多出的一列/行會被捨棄。"""
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    blocks = x[:, :, :h2 * 2, :w2 * 2].reshape(n, c, h2, 2, w2, 2)
    out = blocks.max(axis=(3, 5))
    return out, (blocks, out, x.shape)


def _maxpool_backward(d_out, cache):
    blocks, out, x_shape = cache
    mask = blocks == out[:, :, :, None, :, None]
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)  # 平手時平均分配梯度
    d_blocks = mask * d_out[:, :, :, None, :, None]
    n, c, h2, _, w2, _ = blocks.shape
    d_x = np.zeros(x_shape)
    d_x[:, :, :h2 * 2, :w2 * 2] = d_blocks.reshape(n, c, h2 * 2, w2 * 2)
    return d_x


class CNN(_ClassifierMixin):
    """conv3x3 → ReLU → pool → conv3x3 → ReLU → pool → 全連接。輸入 (N,1,28,28)，輸出 (N,10) logits。"""

    def __init__(self, in_channels=1, num_classes=10, image_size=28,
                 channels1=8, channels2=16, seed=42):
        rng = np.random.default_rng(seed)
        he = lambda fan_in, *shape: rng.normal(0, np.sqrt(2 / fan_in), shape)
        size = ((image_size - 2) // 2 - 2) // 2  # 28 → 26 → 13 → 11 → 5
        self.parameters = {
            'conv1_w': he(in_channels * 9, channels1, in_channels, 3, 3),
            'conv1_b': np.zeros(channels1),
            'conv2_w': he(channels1 * 9, channels2, channels1, 3, 3),
            'conv2_b': np.zeros(channels2),
            'fc_w': he(channels2 * size * size, channels2 * size * size, num_classes),
            'fc_b': np.zeros(num_classes),
        }

    def forward(self, images, keep_cache=False):
        p = self.parameters
        x = np.asarray(images, dtype=np.float64)
        z1, cols1 = _conv_forward(x, p['conv1_w'], p['conv1_b'])
        a1 = np.maximum(z1, 0)
        pool1, pool_cache1 = _maxpool_forward(a1)
        z2, cols2 = _conv_forward(pool1, p['conv2_w'], p['conv2_b'])
        a2 = np.maximum(z2, 0)
        pool2, pool_cache2 = _maxpool_forward(a2)
        flat = pool2.reshape(len(x), -1)
        logits = flat @ p['fc_w'] + p['fc_b']
        if not keep_cache:
            return logits
        return logits, (x.shape, cols1, z1, pool_cache1, pool1.shape, cols2, z2, pool_cache2, pool2.shape, flat)

    def loss_and_gradients(self, images, labels):
        """批次平均交叉熵損失與全部參數的解析梯度。"""
        p = self.parameters
        logits, cache = self.forward(images, keep_cache=True)
        x_shape, cols1, z1, pool_cache1, pool1_shape, cols2, z2, pool_cache2, pool2_shape, flat = cache
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)

        probabilities = softmax(logits)
        loss = -np.log(probabilities[np.arange(n), labels] + 1e-12).mean()
        d_logits = probabilities.copy()
        d_logits[np.arange(n), labels] -= 1
        d_logits /= n

        gradients = {}
        gradients['fc_w'] = flat.T @ d_logits
        gradients['fc_b'] = d_logits.sum(axis=0)
        d_pool2 = (d_logits @ p['fc_w'].T).reshape(pool2_shape)

        d_z2 = _maxpool_backward(d_pool2, pool_cache2) * (z2 > 0)
        d_pool1, gradients['conv2_w'], gradients['conv2_b'] = _conv_backward(d_z2, cols2, pool1_shape, p['conv2_w'])

        d_z1 = _maxpool_backward(d_pool1, pool_cache1) * (z1 > 0)
        _, gradients['conv1_w'], gradients['conv1_b'] = _conv_backward(d_z1, cols1, x_shape, p['conv1_w'])
        return float(loss), gradients


class MLP(_ClassifierMixin):
    """多層感知器：輸入攤平成向量，隱藏層用 ReLU，輸出 logits。可接受 (N,784) 或 (N,1,28,28)。"""

    def __init__(self, input_size=784, hidden_sizes=(128, 64), num_classes=10, seed=42):
        rng = np.random.default_rng(seed)
        sizes = [input_size, *hidden_sizes, num_classes]
        self.num_layers = len(sizes) - 1
        self.parameters = {}
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.parameters[f'w{i}'] = rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
            self.parameters[f'b{i}'] = np.zeros(fan_out)

    def forward(self, images, keep_cache=False):
        p = self.parameters
        a = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
        inputs, pre_activations = [], []
        for i in range(self.num_layers):
            inputs.append(a)
            z = a @ p[f'w{i}'] + p[f'b{i}']
            pre_activations.append(z)
            a = np.maximum(z, 0) if i < self.num_layers - 1 else z
        if not keep_cache:
            return a
        return a, (inputs, pre_activations)

    def loss_and_gradients(self, images, labels):
        """批次平均交叉熵損失與全部參數的解析梯度。"""
        p = self.parameters
        logits, (inputs, pre_activations) = self.forward(images, keep_cache=True)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)

        probabilities = softmax(logits)
        loss = -np.log(probabilities[np.arange(n), labels] + 1e-12).mean()
        d = probabilities.copy()
        d[np.arange(n), labels] -= 1
        d /= n

        gradients = {}
        for i in reversed(range(self.num_layers)):
            if i < self.num_layers - 1:
                d = d * (pre_activations[i] > 0)  # ReLU
            gradients[f'w{i}'] = inputs[i].T @ d
            gradients[f'b{i}'] = d.sum(axis=0)
            d = d @ p[f'w{i}'].T
        return float(loss), gradients
