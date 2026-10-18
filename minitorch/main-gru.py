"""使用 minitorch 訓練字元級 GRU（批次化、padding、梯度裁剪、AdamW、cosine 排程）。"""

import random
import minitorch as mt
from minitorch import data, models, train

random.seed(42)
mt.manual_seed(42)
corpus = data.CharCorpus('input.txt', block_size=16)
model = models.GRULanguageModel(corpus.vocab_size, corpus.block_size, width=32)
train.run_language_model_demo(model, corpus, 'minitorch GRU')
