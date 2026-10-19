"""minitorch：GPT 微調（第二階段）——載入預訓練檢查點，用 finetune.txt 的問答微調，並與其他做法比較。

finetune.txt 有 50 種不同的問答，每種重複 27 行。評估時以「不同的問答」為單位做 5 折交叉驗證：
每次留下 10 題完全不拿來訓練（連同它們的所有重複行），讓模型回答；5 折合起來 50 題都被當作「沒看過的題目」測過一次。
損失只計在答案上。除了「完整寫出正確答案」，也算「從 48 個候選答案中挑對」的名次（選擇題式評估），
用來分辨模型是「不知道」還是「知道但寫不出來」。請先執行 pretrain.py。
"""

import math
import os
import random
import minitorch as mt
from minitorch import checkpoint, data, models, qa, train

FOLDS = 5
EPOCHS = 3            # 以訓練行數計（每種問答已重複 27 行，所以 3 輪就等於每題看了上百次）
LR_SCRATCH = 0.01     # 從頭訓練用和預訓練相同的學習率
LR_FINETUNE = 0.002   # 微調用較小的學習率，避免一下子把預訓練學到的東西洗掉

if not os.path.exists('pretrained.npz'):
    raise SystemExit('找不到 pretrained.npz，請先執行 pretrain.py')

random.seed(42)
mt.manual_seed(42)
pretrained, chars = checkpoint.load_gpt('pretrained.npz')
corpus = data.CharCorpus(docs=[], block_size=pretrained.block_size, vocab=chars)
sentences = [corpus.encode(doc) for doc in sorted(set(data.read_docs('pretrain.txt')))]  # 用來量「遺忘」

line_pairs = qa.read_pairs('finetune.txt')          # 1350 行，含重複
unique = qa.unique_pairs(line_pairs)                # 50 種不同的問答
candidates = sorted({answer for _, answer in unique})   # 選擇題的候選答案
folds = qa.make_folds(unique, k=FOLDS, seed=42)
print(f'微調資料：{len(line_pairs)} 行，{len(unique)} 種不同的問答；{FOLDS} 折交叉驗證，每折 {len(folds[0])} 題當作沒看過的題目\n')


def width(text):
    return sum(2 if ord(char) > 0x2E80 else 1 for char in text)  # 中文字佔兩格


def ljust(text, size):
    return text + ' ' * (size - width(text))


def rjust(text, size):
    return ' ' * (size - width(text)) + text


def load_pretrained():
    return checkpoint.load_gpt('pretrained.npz')[0]


# (名稱, 建立模型, 學習率, 要凍結的參數前綴)
VARIANTS = [
    ('從頭訓練（不預訓練）', lambda: models.GPT(**pretrained.config), LR_SCRATCH, ()),
    ('預訓練 + 全部微調', load_pretrained, LR_FINETUNE, ()),
    ('預訓練 + 凍結詞嵌入與第 1 層', load_pretrained, LR_FINETUNE, ('wte', 'wpe', 'blocks.0.')),
]

results, curves, predictions = {}, {}, {}
zero_shot = load_pretrained()
results['預訓練後（未微調）'] = {
    'val_acc': qa.answer_accuracy(zero_shot, corpus, unique), 'train_acc': float('nan'),
    'val_loss': train.evaluate_loss(zero_shot, qa.encode_pairs(corpus, unique)),
    'val_rank': sum(qa.answer_ranks(zero_shot, corpus, unique, candidates)) / len(unique),
    'sentence_loss': train.evaluate_loss(zero_shot, sentences)}

for name, make_model, lr, freeze in VARIANTS:
    print(name, flush=True)
    correct, val_losses, ranks, train_accs, sentence_losses, curve_sum = 0, [], [], [], [], [0.0] * EPOCHS
    wrong = []
    for fold, held_out in enumerate(folds):
        model = make_model()
        frozen = model.freeze(*freeze) if freeze else 0
        if frozen and fold == 0:
            print(f'  凍結 {frozen} 個參數，只訓練其餘 {model.num_parameters(trainable_only=True)} 個')
        held_examples = qa.encode_pairs(corpus, held_out)
        curve = []
        train.train_language_model(model, qa.encode_pairs(corpus, qa.without(line_pairs, held_out)),
                                   epochs=EPOCHS, lr=lr, weight_decay=0.0,
                                   after_epoch=lambda e: curve.append(train.evaluate_loss(model, held_examples)))
        got = qa.predict_answers(model, corpus, held_out)
        correct += sum(g == a for g, (_, a) in zip(got, held_out))
        wrong += [(q, a, g) for g, (q, a) in zip(got, held_out)]
        val_losses.append(curve[-1])
        ranks += qa.answer_ranks(model, corpus, held_out, candidates)
        curve_sum = [s + c for s, c in zip(curve_sum, curve)]
        seen = [p for p in unique if p not in set(held_out)]
        train_accs.append(qa.answer_accuracy(model, corpus, seen))
        sentence_losses.append(train.evaluate_loss(model, sentences))
    results[name] = {'val_acc': correct / len(unique), 'train_acc': sum(train_accs) / FOLDS,
                     'val_loss': sum(val_losses) / FOLDS, 'val_rank': sum(ranks) / len(ranks), 'sentence_loss': sum(sentence_losses) / FOLDS}
    curves[name] = [c / FOLDS for c in curve_sum]
    predictions[name] = wrong

print('\n=== 結果（驗證 = 沒看過的題目，5 折合起來 50 題）===')
print(ljust('', 32) + ''.join(rjust(h, 10) for h in ('驗證答對', '驗證損失', '平均名次', '訓練答對', '句子損失')))
for name, r in results.items():
    train_acc = '—' if math.isnan(r['train_acc']) else f'{r["train_acc"]:.0%}'
    print(ljust(name, 32) + f'{r["val_acc"]:>10.0%}{r["val_loss"]:>10.3f}{r["val_rank"]:>10.1f}{train_acc:>10}{r["sentence_loss"]:>10.3f}')
chance = qa.expected_random_rank(corpus, unique, candidates)
print(f'（平均名次：選擇題式評估，正確答案在約 48 個候選答案中的名次，越小越好，亂猜平均約 {chance:.1f}；'
      '句子損失：預訓練句子的損失，上升代表遺忘；「預訓練後」那列的驗證是對全部 50 題，因為沒有拿任何題目訓練）')

print('\n=== 各輪結束時的驗證損失（5 折平均，答案部分）===')
print(ljust('輪數', 32) + ''.join(f'{e + 1:>8}' for e in range(EPOCHS)))
for name, curve in curves.items():
    print(ljust(name, 32) + ''.join(f'{v:>8.3f}' for v in curve))

best = '預訓練 + 全部微調'
errors = [(q, a, g) for q, a, g in predictions[best] if a != g]
print(f'\n=== {best}：沒看過的題目答錯了 {len(errors)} 題（列出前 10 題）===')
for q, a, g in errors[:10]:
    print(f'  {qa.prompt(q)}  模型：{g}  正確：{a}')

print('\n=== 用全部 50 題微調，存成 finetuned.npz ===')
final = load_pretrained()
train.train_language_model(final, qa.encode_pairs(corpus, line_pairs), epochs=EPOCHS, lr=LR_FINETUNE, weight_decay=0.0)
print(f'訓練題目答對：{qa.answer_accuracy(final, corpus, unique):.0%}')
checkpoint.save_gpt('finetuned.npz', final, chars)
print('\n改寫過的問題（不在檔案裡，沒有標準答案可比對，僅供觀察）：')
for question in ['火星有幾顆衛星？', '木星的直徑是多少？', '水星有大氣層嗎？']:
    if all(char in corpus.token_of for char in qa.prompt(question)):
        print(f'  {qa.prompt(question)}{train.complete(final, corpus, qa.prompt(question), max_new=16)}')
print('\n已儲存 finetuned.npz')
