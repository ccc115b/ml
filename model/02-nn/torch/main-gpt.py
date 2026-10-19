"""使用 PyTorch 訓練字元級 GPT（多層、多頭、LayerNorm、GELU）。"""

from nnlib import data, models, train

train.set_seed(42)
corpus = data.CharCorpus('input.txt', block_size=16)
model = models.GPT(corpus.vocab_size, corpus.block_size, width=32, n_layer=2, n_head=2)
train.run_language_model_demo(model, corpus, 'PyTorch GPT')
