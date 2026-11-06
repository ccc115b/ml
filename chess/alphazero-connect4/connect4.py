#!/usr/bin/env python3
"""Connect 4 (四子棋) - 終端機對戰版
用法:  python3 connect4.py

遊戲核心 Connect4 類別之後可直接給 AlphaZero 使用 (copy / play / undo /
legal_moves / winner / to_array)。
"""
import random
import time

ROWS, COLS = 6, 7
ORDER = [3, 2, 4, 1, 5, 0, 6]  # 由中間往外搜尋，alpha-beta 剪枝更有效


class Connect4:
    """board[r][c]: r=0 為最底列。1 = 先手, -1 = 後手, 0 = 空。"""

    def __init__(self):
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS
        self.player = 1      # 輪到誰下
        self.moves = 0
        self.winner = 0      # 0 = 尚未分勝負

    # ---------- 基本操作 ----------
    def copy(self):
        g = Connect4()
        g.board = [row[:] for row in self.board]
        g.heights = self.heights[:]
        g.player, g.moves, g.winner = self.player, self.moves, self.winner
        return g

    def legal_moves(self):
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def is_over(self):
        return self.winner != 0 or self.moves == ROWS * COLS

    def play(self, col):
        r = self.heights[col]
        self.board[r][col] = self.player
        self.heights[col] += 1
        self.moves += 1
        if self._is_win(r, col):
            self.winner = self.player
        self.player = -self.player

    def undo(self, col):
        self.heights[col] -= 1
        self.board[self.heights[col]][col] = 0
        self.moves -= 1
        self.winner = 0
        self.player = -self.player

    def _is_win(self, r, c):
        p = self.board[r][c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr][cc] == p:
                    n += 1
                    rr += dr * s
                    cc += dc * s
            if n >= 4:
                return True
        return False

    # ---------- 給神經網路用 ----------
    def to_array(self):
        """回傳 (2, 6, 7): [自己的棋子, 對手的棋子]，以「輪到的人」為視角。"""
        me = [[1 if v == self.player else 0 for v in row] for row in self.board]
        op = [[1 if v == -self.player else 0 for v in row] for row in self.board]
        return [me, op]

    # ---------- 顯示 ----------
    def render(self, last=None):
        sym = {0: "·", 1: "\033[91m●\033[0m", -1: "\033[93m●\033[0m"}
        lines = ["\n  1 2 3 4 5 6 7"]
        for r in range(ROWS - 1, -1, -1):
            lines.append("| " + " ".join(sym[v] for v in self.board[r]) + " |")
        lines.append("+" + "-" * 15 + "+")
        if last is not None:
            lines.append("  " + "  " * last + "^")
        return "\n".join(lines)


# ---------- AI: negamax + alpha-beta ----------
def _windows():
    w = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + 3 * dr, c + 3 * dc
                if 0 <= er < ROWS and 0 <= ec < COLS:
                    w.append([(r + i * dr, c + i * dc) for i in range(4)])
    return w


WINDOWS = _windows()
WSCORE = {0: 0, 1: 1, 2: 5, 3: 25}


def evaluate(g):
    """以「輪到下棋的人」為視角的啟發式分數。"""
    me = g.player
    score = sum(3 for r in range(ROWS) if g.board[r][3] == me)
    score -= sum(3 for r in range(ROWS) if g.board[r][3] == -me)
    for w in WINDOWS:
        m = o = 0
        for r, c in w:
            v = g.board[r][c]
            if v == me:
                m += 1
            elif v:
                o += 1
        if o == 0:
            score += WSCORE[m]
        elif m == 0:
            score -= WSCORE[o]
    return score


def negamax(g, depth, alpha, beta):
    if g.winner:                       # 上一手的人剛贏 -> 對輪到的人是壞事
        return -(1000 + depth)
    if g.moves == ROWS * COLS:
        return 0
    if depth == 0:
        return evaluate(g)
    best = -10**9
    for c in ORDER:
        if g.heights[c] >= ROWS:
            continue
        g.play(c)
        v = -negamax(g, depth - 1, -beta, -alpha)
        g.undo(c)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def ai_move(g, depth):
    scores = {}
    for c in g.legal_moves():
        g.play(c)
        scores[c] = -negamax(g, depth - 1, -10**9, 10**9)
        g.undo(c)
    top = max(scores.values())
    return random.choice([c for c, s in scores.items() if s == top])


LEVELS = {"1": ("簡單", 2), "2": ("普通", 4), "3": ("困難", 6)}


# ---------- 終端機介面 ----------
def main():
    print("=== Connect 4 四子棋 ===")
    lv = ""
    while lv not in LEVELS:
        lv = input("難度 1=簡單 2=普通 3=困難: ").strip()
    name, depth = LEVELS[lv]
    first = input("你要先手嗎？(Y/n): ").strip().lower() != "n"
    human = 1 if first else -1

    g = Connect4()
    last = None
    print(f"難度：{name}。你是紅色 ●" if first else f"難度：{name}。你是黃色 ●")
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
            t = time.time()
            col = ai_move(g, depth)
            print(f"AI 下第 {col + 1} 欄 ({time.time() - t:.1f} 秒)")
        g.play(col)
        last = col

    print(g.render(last))
    if g.winner == human:
        print("🎉 你贏了！")
    elif g.winner:
        print("AI 獲勝。")
    else:
        print("和局。")


if __name__ == "__main__":
    main()
