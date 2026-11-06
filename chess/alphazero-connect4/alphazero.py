#!/usr/bin/env python3
"""AlphaZero 訓練 Connect 4 (PyTorch, 自動使用 Apple MPS / CUDA / CPU)

需要: pip install torch numpy
檔案: connect4.py, az_mcts.py, alphazero.py 放在同一個資料夾

用法:
  python3 alphazero.py train --minutes 60      # 訓練 60 分鐘 (可中斷、可續訓)
  python3 alphazero.py play                    # 和訓練好的 AI 對戰
  python3 alphazero.py eval                    # 對 alpha-beta AI 測勝率
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from az_mcts import encode, match_vs_alphabeta, pick_move, search, self_play
from connect4 import COLS, ROWS, Connect4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------- 網路 ----------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.p_conv = nn.Conv2d(ch, 2, 1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * ROWS * COLS, COLS)
        self.v_conv = nn.Conv2d(ch, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(ROWS * COLS, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.body(self.stem(x))
        p = self.p_fc(F.relu(self.p_bn(self.p_conv(x))).flatten(1))
        v = F.relu(self.v_bn(self.v_conv(x))).flatten(1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v)))).squeeze(1)
        return p, v


def make_predict_fn(net, device):
    net.eval()

    @torch.inference_mode()
    def predict(states):
        x = torch.from_numpy(np.stack(states)).to(device)
        logits, v = net(x)
        return F.softmax(logits, dim=1).cpu().numpy(), v.cpu().numpy()

    return predict


def save(net, path, it, ch, blocks):
    torch.save({"state": net.state_dict(), "iter": it, "ch": ch, "blocks": blocks}, path)


def load(path, device):
    ck = torch.load(path, map_location=device)
    net = Net(ck["ch"], ck["blocks"]).to(device)
    net.load_state_dict(ck["state"])
    return net, ck


# ---------------- 訓練 ----------------
def train_steps(net, opt, buf, steps, bs, device):
    net.train()
    tp = tv = 0.0
    for _ in range(steps):
        idx = np.random.randint(len(buf), size=bs)
        s = torch.from_numpy(np.stack([buf[i][0] for i in idx])).to(device)
        pi = torch.from_numpy(np.stack([buf[i][1] for i in idx])).to(device)
        z = torch.tensor([buf[i][2] for i in idx], dtype=torch.float32, device=device)
        logits, v = net(s)
        lp = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean()
        lv = F.mse_loss(v, z)
        opt.zero_grad()
        (lp + lv).backward()
        opt.step()
        tp += lp.item()
        tv += lv.item()
    return tp / steps, tv / steps


def cmd_train(a):
    device = get_device()
    print(f"裝置: {device}")
    if a.resume and os.path.exists(a.model):
        net, ck = load(a.model, device)
        start, ch, blocks = ck["iter"], ck["ch"], ck["blocks"]
        print(f"從 {a.model} 續訓 (已完成 {start} 輪)")
    else:
        ch, blocks, start = a.channels, a.blocks, 0
        net = Net(ch, blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    buf = []
    t0 = time.time()
    it = start
    while it < start + a.iters:
        if a.minutes and (time.time() - t0) / 60 >= a.minutes:
            print("時間到，停止訓練。")
            break
        it += 1
        t1 = time.time()
        predict = make_predict_fn(net, device)
        samples, lens = self_play(predict, a.games, parallel=a.parallel, sims=a.sims)
        buf.extend(samples)
        del buf[: max(0, len(buf) - a.buffer)]
        lp, lv = train_steps(net, opt, buf, a.steps, a.batch, device)
        save(net, a.model, it, ch, blocks)
        print(f"[輪 {it}] 自我對弈 {a.games} 盤 (平均 {np.mean(lens):.1f} 手) | "
              f"policy loss {lp:.3f} value loss {lv:.3f} | 資料 {len(buf)} | "
              f"本輪 {time.time() - t1:.0f}s, 累計 {(time.time() - t0) / 60:.1f} 分", flush=True)
        if a.eval_every and it % a.eval_every == 0:
            w, d, l = match_vs_alphabeta(make_predict_fn(net, device), a.eval_depth, 20, a.sims)
            print(f"   ↳ 對 alpha-beta(深度{a.eval_depth}) 20 盤: {w} 勝 {d} 和 {l} 負", flush=True)
    save(net, a.model, it, ch, blocks)
    print(f"已儲存 {a.model}")


# ---------------- 評估 ----------------
def cmd_eval(a):
    device = get_device()
    net, ck = load(a.model, device)
    predict = make_predict_fn(net, device)
    print(f"模型已訓練 {ck['iter']} 輪")
    for depth in (2, 4, 6):
        w, d, l = match_vs_alphabeta(predict, depth, a.games, a.play_sims)
        print(f"對 alpha-beta 深度 {depth}: {w} 勝 {d} 和 {l} 負 (共 {a.games} 盤)")


# ---------------- 人機對戰 ----------------
def cmd_play(a):
    device = get_device()
    net, ck = load(a.model, device)
    predict = make_predict_fn(net, device)
    human = 1 if input("你要先手嗎？(Y/n): ").strip().lower() != "n" else -1
    g, last = Connect4(), None
    while not g.is_over():
        print(g.render(last))
        if g.player == human:
            while True:
                s = input("你的選擇 (1-7, q 離開): ").strip().lower()
                if s == "q":
                    return
                if s.isdigit() and 1 <= int(s) <= COLS and g.heights[int(s) - 1] < ROWS:
                    col = int(s) - 1
                    break
                print("不合法，請重新輸入。")
        else:
            root = search([g], predict, a.play_sims)[0]
            col = pick_move(root, sample=False)
            win = (root.w / root.n + 1) / 2 * 100
            print(f"AI 下第 {col + 1} 欄 (AI 自評勝率約 {win:.0f}%)")
        g.play(col)
        last = col
    print(g.render(last))
    print("🎉 你贏了！" if g.winner == human else "AI 獲勝。" if g.winner else "和局。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", default="c4_az.pt")
    t.add_argument("--minutes", type=float, default=0, help="訓練時間上限(分鐘)，0=不限")
    t.add_argument("--iters", type=int, default=1000)
    t.add_argument("--games", type=int, default=64, help="每輪自我對弈盤數")
    t.add_argument("--parallel", type=int, default=64, help="同時進行的對局數 (batch 大小)")
    t.add_argument("--sims", type=int, default=100, help="每步 MCTS 模擬次數")
    t.add_argument("--steps", type=int, default=100, help="每輪訓練步數")
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--buffer", type=int, default=60000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--channels", type=int, default=64)
    t.add_argument("--blocks", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=5)
    t.add_argument("--eval-depth", type=int, default=4)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(fn=cmd_train)

    for name, fn in (("play", cmd_play), ("eval", cmd_eval)):
        p = sub.add_parser(name)
        p.add_argument("--model", default="c4_az.pt")
        p.add_argument("--play-sims", type=int, default=400)
        p.add_argument("--games", type=int, default=20)
        p.set_defaults(fn=fn)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
