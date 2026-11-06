# Connect 4 + AlphaZero

四子棋 (7x6) 的終端機對戰版，以及用 AlphaZero 方法訓練的 AI。

## 檔案結構

```
connect4-alphazero/
├── connect4.py       遊戲規則 + alpha-beta AI + 終端機對戰 (不需任何套件)
├── az_mcts.py        MCTS、自我對弈、對戰評估 (只需 numpy)
├── alphazero.py      PyTorch 網路、訓練、人機對戰、評估
├── requirements.txt
└── README.md
```

## 安裝

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 使用

```bash
# 1. 和 alpha-beta AI 對戰 (不需安裝 torch)
python3 connect4.py

# 2. 訓練 AlphaZero (Apple Silicon 會自動使用 MPS)
python3 alphazero.py train --minutes 10     # 先試跑 10 分鐘
python3 alphazero.py train --minutes 60     # 正式訓練一小時
python3 alphazero.py train --resume --minutes 60   # 接續上次的進度

# 3. 和訓練好的 AI 對戰
python3 alphazero.py play

# 4. 對 alpha-beta 深度 2/4/6 測勝率
python3 alphazero.py eval
```

模型存在 `c4_az.pt`，每輪訓練後自動更新，可隨時 Ctrl+C 中斷。

## 常用訓練參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--sims` | 100 | 每步 MCTS 模擬次數，越大越強但越慢 |
| `--games` | 64 | 每輪自我對弈盤數 |
| `--parallel` | 64 | 同時對弈數 (推論 batch 大小) |
| `--steps` | 100 | 每輪訓練步數 |
| `--blocks` / `--channels` | 4 / 64 | 網路深度 / 寬度 |
| `--eval-every` | 5 | 每幾輪對 alpha-beta 測一次勝率 |
