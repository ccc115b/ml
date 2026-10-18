"""使用 NumPy tensor 版 nn 訓練字元級 GRU；請由範例庫載入。"""

import random
from minitorch import common, data, nn

random.seed(42)
corpus = data.CharCorpus('input.txt', block_size=16)
model = nn.GRU(corpus.vocab_size, corpus.block_size, width=24)
common.run_language_model_demo(model, corpus, 'NumPy tensor 版 GRU')
