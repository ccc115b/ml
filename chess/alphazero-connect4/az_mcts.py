"""AlphaZero 的 MCTS、自我對弈、對戰評估 (只依賴 numpy，與深度學習框架無關)。

predict_fn(states) -> (probs, values)
    states: list of (2,6,7) float32 array (輪到下棋者視角)
    probs : (B,7) 各欄位的先驗機率
    values: (B,)  輪到下棋者的預期勝負 (-1 ~ +1)
"""
import math
import random

import numpy as np

from connect4 import COLS, Connect4, ai_move


def encode(g):
    b = np.array(g.board, dtype=np.int8) * g.player
    return np.stack([b == 1, b == -1]).astype(np.float32)


class Node:
    __slots__ = ("p", "n", "w", "ch")

    def __init__(self, p):
        self.p = p      # 先驗機率
        self.n = 0      # 訪問次數
        self.w = 0.0    # 價值總和 (以「該節點輪到下棋者」為視角)
        self.ch = None  # 子節點 {欄位: Node}；None = 未展開或終局


def expand(node, legal, probs):
    pr = np.array([probs[c] for c in legal], dtype=np.float64)
    s = pr.sum()
    pr = pr / s if s > 1e-8 else np.full(len(legal), 1.0 / len(legal))
    node.ch = {c: Node(float(p)) for c, p in zip(legal, pr)}


def select(node, c_puct):
    sq = math.sqrt(node.n + 1)
    best, bm, bc = -1e9, None, None
    for m, ch in node.ch.items():
        q = -ch.w / ch.n if ch.n else 0.0   # 子節點的價值要換成父節點視角
        s = q + c_puct * ch.p * sq / (1 + ch.n)
        if s > best:
            best, bm, bc = s, m, ch
    return bm, bc


def backup(path, v):
    for node in reversed(path):
        node.n += 1
        node.w += v
        v = -v


def search(games, predict_fn, sims, noise=False, c_puct=1.5, dir_alpha=1.0, eps=0.25):
    """同時對多盤棋做 MCTS，葉節點集中成一個 batch 丟給網路。回傳每盤的 root。"""
    roots = [Node(1.0) for _ in games]
    probs, _ = predict_fn([encode(g) for g in games])
    for g, r, p in zip(games, roots, probs):
        expand(r, g.legal_moves(), p)
        if noise:
            ms = list(r.ch)
            nz = np.random.dirichlet([dir_alpha] * len(ms))
            for m, x in zip(ms, nz):
                r.ch[m].p = (1 - eps) * r.ch[m].p + eps * x

    for _ in range(sims):
        pend = []
        for g, root in zip(games, roots):
            node, path, moves = root, [root], []
            while node.ch:
                m, node = select(node, c_puct)
                g.play(m)
                moves.append(m)
                path.append(node)
            if g.is_over():
                backup(path, -1.0 if g.winner else 0.0)  # 上一手的人贏了 -> 對輪到者是 -1
            else:
                pend.append((path, encode(g), g.legal_moves()))
            for m in reversed(moves):
                g.undo(m)
        if pend:
            probs, vals = predict_fn([x[1] for x in pend])
            for (path, _, legal), p, v in zip(pend, probs, vals):
                expand(path[-1], legal, p)
                backup(path, float(v))
    return roots


def visit_policy(root):
    pi = np.zeros(COLS, dtype=np.float32)
    for m, c in root.ch.items():
        pi[m] = c.n
    return pi / pi.sum()


def pick_move(root, sample):
    ms = list(root.ch)
    ns = np.array([root.ch[m].n for m in ms], dtype=np.float64)
    if sample:
        return ms[np.random.choice(len(ms), p=ns / ns.sum())]
    return ms[int(np.argmax(ns))]


def self_play(predict_fn, n_games, parallel=64, sims=100, temp_moves=10):
    """跑 n_games 盤自我對弈，回傳 [(state, pi, z), ...] (已含左右鏡射增強)。"""
    new = lambda: {"g": Connect4(), "hist": []}
    started = min(parallel, n_games)
    active = [new() for _ in range(started)]
    samples, lengths = [], []
    while active:
        roots = search([a["g"] for a in active], predict_fn, sims, noise=True)
        nxt = []
        for a, r in zip(active, roots):
            g = a["g"]
            a["hist"].append((encode(g), visit_policy(r), g.player))
            g.play(pick_move(r, sample=len(a["hist"]) <= temp_moves))
            if g.is_over():
                lengths.append(g.moves)
                for s, p, pl in a["hist"]:
                    z = 0.0 if g.winner == 0 else (1.0 if g.winner == pl else -1.0)
                    samples.append((s, p, z))
                    samples.append((s[:, :, ::-1].copy(), p[::-1].copy(), z))  # 鏡射
                if started < n_games:
                    nxt.append(new())
                    started += 1
            else:
                nxt.append(a)
        active = nxt
    return samples, lengths


def match_vs_alphabeta(predict_fn, depth, n_games=20, sims=100, random_plies=2):
    """網路(MCTS) 對 alpha-beta AI。開局隨機下幾手以增加變化。回傳 (勝, 和, 負)。"""
    games = [Connect4() for _ in range(n_games)]
    side = [1 if i % 2 == 0 else -1 for i in range(n_games)]  # 網路的顏色
    for g in games:
        for _ in range(random_plies):
            g.play(random.choice(g.legal_moves()))
    while True:
        net_i = [i for i, g in enumerate(games) if not g.is_over() and g.player == side[i]]
        ab_i = [i for i, g in enumerate(games) if not g.is_over() and g.player != side[i]]
        if not net_i and not ab_i:
            break
        if net_i:
            roots = search([games[i] for i in net_i], predict_fn, sims)
            for i, r in zip(net_i, roots):
                games[i].play(pick_move(r, sample=False))
        for i in ab_i:
            games[i].play(ai_move(games[i], depth))
    w = d = l = 0
    for g, s in zip(games, side):
        if g.winner == 0:
            d += 1
        elif g.winner == s:
            w += 1
        else:
            l += 1
    return w, d, l
