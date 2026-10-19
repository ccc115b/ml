"""使用 PyTorch 預訓練 GPT（第一階段）——在 pretrain.txt 上學「下一個字預測」，存成檢查點 pretrained.pt。"""

import math
import random
from nnlib import checkpoint, data, models, train

BLOCK_SIZE = 32   # pretrain.txt 最長 23 字、finetune.txt 最長 28 字，要放得下

train.set_seed(42)
device = train.get_device()

docs = data.read_docs('pretrain.txt')
unique = sorted(set(docs))
vocab = data.build_vocab(docs, data.read_docs('finetune.txt'))  # 字元表同時涵蓋兩階段的資料（含 <Q>、<A> 標記）
corpus = data.CharCorpus(docs=docs, block_size=BLOCK_SIZE, vocab=vocab)
unique_examples = [corpus.encode(doc) for doc in unique]

model = models.GPT(corpus.vocab_size, BLOCK_SIZE, width=32, n_layer=2, n_head=2).to(device)
print(f'裝置：{device}')
print(f'預訓練資料：{len(docs)} 行，其中不同的句子 {len(unique)} 種（每種約重複 {len(docs) // len(unique)} 次）')
print(f'字元表 {corpus.vocab_size}，模型參數 {train.count_parameters(model)}')
print(f'訓練前句子損失：{train.evaluate_loss(model, unique_examples):.4f}')

train.train_language_model(model, corpus.examples(), epochs=2, lr=0.01, log_every=200)

# 資料裡只有這些句子，模型最好也只能做到「記住全部句子，只剩挑哪一句的不確定性」，損失的理論下限約為 ln(句數) / 平均 token 數
floor = math.log(len(unique)) / (sum(len(d) + 1 for d in unique) / len(unique))
print(f'訓練後句子損失：{train.evaluate_loss(model, unique_examples):.4f}（理論下限約 {floor:.4f}）')

print('\n--- 預訓練模型生成的句子（✓ = 與語料中的某句完全相同）---')
known = set(docs)
for i, sentence in enumerate(train.generate(model, corpus, device, num_samples=10), 1):
    print(f'{i:2d}. {"✓" if sentence in known else "✗"} {sentence}')

checkpoint.save_gpt('pretrained.pt', model, corpus.chars)
print('\n已儲存 pretrained.pt')
