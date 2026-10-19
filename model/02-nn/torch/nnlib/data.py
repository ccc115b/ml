"""資料載入（純 NumPy，與 minitorch 版相同）：字元級文字語料、MNIST、挑選類別、縮小影像。"""

import os
import urllib.request
import numpy as np


def read_docs(path):
    """逐行讀取文字檔，去掉空行。"""
    try:
        with open(path, encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到 {path}；請從範例庫載入。') from None


def build_vocab(*doc_lists):
    """多份文件的字元聯集（排序後）。預訓練前先把微調資料也算進來，詞彙表才能兩階段共用。"""
    return sorted(set(''.join(''.join(docs) for docs in doc_lists)))


def split_docs(docs, val_fraction=0.1, seed=42):
    """固定種子切出驗證集，回傳 (訓練句子, 驗證句子)；預訓練與微調要用同一個切法。"""
    order = np.random.default_rng(seed).permutation(len(docs))
    n_val = max(1, round(len(docs) * val_fraction))
    return [docs[i] for i in sorted(order[n_val:])], [docs[i] for i in sorted(order[:n_val])]


class CharCorpus:
    """字元級語料；BOS 同時作為句首與句尾標記。
    預設從 path 逐行讀取並由內容建立詞彙表；也可直接給 docs，並用 vocab 指定既有的字元表（例如微調時沿用預訓練的）。"""

    def __init__(self, path='input.txt', block_size=16, docs=None, vocab=None):
        self.docs = read_docs(path) if docs is None else list(docs)
        self.chars = sorted(set(''.join(self.docs))) if vocab is None else list(vocab)
        unknown = set(''.join(self.docs)) - set(self.chars)
        if unknown:
            raise ValueError(f'詞彙表中沒有這些字元：{sorted(unknown)}')
        self.token_of = {char: index for index, char in enumerate(self.chars)}
        self.BOS = len(self.chars)
        self.vocab_size = self.BOS + 1
        self.block_size = block_size

    def encode(self, doc):
        """回傳 (tokens, targets)：targets 是 tokens 向左錯一位的下一字元。"""
        sequence = [self.BOS] + [self.token_of[char] for char in doc] + [self.BOS]
        tokens = np.array(sequence[:-1][:self.block_size])
        targets = np.array(sequence[1:][:self.block_size])
        return tokens, targets

    def examples(self):
        """每行文字對應一組 (tokens, targets)，供 DataLoader 使用。"""
        return [self.encode(doc) for doc in self.docs]

    def describe(self):
        return f'訓練資料：{len(self.docs)} 行，詞彙表：{self.vocab_size} 個字元'


def load_mnist(path='mnist.npz'):
    """回傳 (x_train, y_train, x_test, y_test)；x shape (N,1,28,28)，值域 0~1。
    若本機沒有 mnist.npz（Keras 格式），會自動下載。"""
    if not os.path.exists(path):
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        print(f'下載 MNIST：{url}')
        urllib.request.urlretrieve(url, path)
    with np.load(path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    prepare = lambda images: (images.astype(np.float64) / 255.0)[:, None, :, :]
    return prepare(x_train), y_train.astype(np.int64), prepare(x_test), y_test.astype(np.int64)


def select_classes(images, labels, classes, per_class=None):
    """只留下指定的類別（每類最多 per_class 張），標籤重新編為 0..len(classes)-1。"""
    picked, new_labels = [], []
    for new, digit in enumerate(classes):
        index = np.where(labels == digit)[0][:per_class]
        picked.append(index)
        new_labels.append(np.full(len(index), new, dtype=np.int64))
    picked = np.concatenate(picked)
    return images[picked], np.concatenate(new_labels)


def shrink(images, factor=2):
    """以區塊平均縮小影像：(N,C,H,W) → (N,C,H/factor,W/factor)。"""
    n, c, h, w = images.shape
    return images.reshape(n, c, h // factor, factor, w // factor, factor).mean(axis=(3, 5))
