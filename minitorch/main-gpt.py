"""使用 minitorch 訓練字元級 GPT（多層、多頭、LayerNorm、GELU）。"""

import random
import minitorch as mt
from minitorch import data, models, train

random.seed(42)
mt.manual_seed(42)
corpus = data.CharCorpus('input.txt', block_size=16)
model = models.GPT(corpus.vocab_size, corpus.block_size, width=32, n_layer=2, n_head=2)
train.run_language_model_demo(model, corpus, 'minitorch GPT')
