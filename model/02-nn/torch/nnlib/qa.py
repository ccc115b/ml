"""微調用的問答資料：讀取 finetune.txt，每行格式「<Q>問題<A>答案」。

檔案裡每一種問答會重複很多行，所以「切分資料」必須以不同的問答為單位（同一題的所有重複行一起進訓練或一起進驗證），
否則驗證題目會和訓練行完全相同，結果就沒有意義。
"""

import re
import numpy as np
from . import train

_LINE = re.compile(r'<Q>([^<>]+)<A>([^<>]+)')


def parse_line(line):
    match = _LINE.fullmatch(line.strip())
    if match is None:
        raise ValueError(f'格式應為「<Q>問題<A>答案」：{line!r}')
    return match.group(1), match.group(2)


def read_pairs(path):
    """逐行讀取，回傳 [(問題, 答案), ...]；重複的行會保留（訓練時照檔案原樣使用）。"""
    try:
        with open(path, encoding='utf-8') as file:
            return [parse_line(line) for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f'找不到 {path}') from None


def unique_pairs(pairs):
    """去除重複並排序，結果與行的順序無關。"""
    return sorted(set(pairs))


def make_folds(unique, k=5, seed=42):
    """把不同的問答隨機分成 k 份（交叉驗證用）；每份輪流當驗證集，所以每一題都會被當作「沒看過的題目」測一次。"""
    order = np.random.default_rng(seed).permutation(len(unique))
    return [[unique[i] for i in sorted(order[f::k])] for f in range(k)]


def without(pairs, held_out):
    """從（含重複的）問答行中移除驗證題目的所有重複行。"""
    held = set(held_out)
    return [pair for pair in pairs if pair not in held]


def prompt(question):
    return f'<Q>{question}<A>'


def encode_pair(corpus, question, answer):
    """(tokens, targets)；提示部分的 target 設為 -100，只有答案（與結尾的 BOS）計入損失。"""
    text = prompt(question)
    assert len(text + answer) + 1 <= corpus.block_size, '問答太長，超過 block_size'
    tokens, targets = corpus.encode(text + answer)
    targets = targets.copy()
    targets[:len(text)] = -100
    return tokens, targets


def encode_pairs(corpus, pairs):
    return [encode_pair(corpus, q, a) for q, a in pairs]


def predict_answers(model, corpus, pairs, max_new=16):
    """貪婪解碼，回傳每題模型寫出的答案。"""
    return [train.complete(model, corpus, prompt(q), max_new=max_new) for q, _ in pairs]


def answer_accuracy(model, corpus, pairs):
    """答案必須與標準答案完全相同才算對。"""
    predictions = predict_answers(model, corpus, pairs)
    return sum(got == a for got, (_, a) in zip(predictions, pairs)) / len(pairs)


def _fitting(corpus, question, candidates):
    return [c for c in candidates if len(prompt(question) + c) + 1 <= corpus.block_size]


def answer_ranks(model, corpus, pairs, candidates):
    """選擇題式的評估：每題把候選答案依模型給的機率由高到低排序，回傳正確答案的名次（1 = 模型最相信它）。
    接在長問題後面會超過 block_size 的候選答案不參與該題的比較（正確答案一定放得下）。"""
    ranks = []
    for question, answer in pairs:
        fits = _fitting(corpus, question, candidates)
        scores = train.sequence_nll(model, [encode_pair(corpus, question, c) for c in fits])
        ranks.append(1 + int((scores < scores[fits.index(answer)]).sum()))
    return ranks


def expected_random_rank(corpus, pairs, candidates):
    """亂猜時正確答案的平均名次，用來對照 answer_ranks。"""
    return sum((len(_fitting(corpus, q, candidates)) + 1) / 2 for q, _ in pairs) / len(pairs)
