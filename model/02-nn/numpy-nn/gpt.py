"""numpy-nn：以手寫反向傳播的 GPT 訓練字元級語言模型，並生成句子。"""

import random
from npnn import data, models, train

random.seed(42)
corpus = data.CharCorpus('input.txt', block_size=16)
model = models.GPT(corpus.vocab_size, corpus.block_size, width=24)
train.run_language_model_demo(model, corpus, '手寫反向傳播版 GPT')
