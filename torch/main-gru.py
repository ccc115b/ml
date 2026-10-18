"""使用 PyTorch 訓練字元級 GRU。"""

from nnlib import data, models, train

train.set_seed(42)
corpus = data.CharCorpus('input.txt', block_size=16)
model = models.GRULanguageModel(corpus.vocab_size, corpus.block_size, width=32)
train.run_language_model_demo(model, corpus, 'PyTorch GRU')
