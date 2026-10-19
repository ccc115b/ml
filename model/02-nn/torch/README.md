# PyTorch 版

用 PyTorch 實作的 MLP、CNN、GRU、GPT（含預訓練與微調）和擴散模型；結構與超參數對應 `minitorch` 版。

| 腳本 | 內容 |
|---|---|
| `main-mlp.py` `main-cnn.py` | 訓練 MNIST 手寫數字辨識 |
| `main-gru.py` `main-gpt.py` | 訓練字元級語言模型（`input.txt`），並生成句子 |
| `main-pretrain.py` `main-finetune.py` | GPT 兩階段訓練：先在 `pretrain.txt` 上預訓練並存成 `pretrained.pt`，再用 `finetune.txt` 的問答（`<Q>問題<A>答案`）微調；以 5 折交叉驗證和「不預訓練」、「部分凍結」比較（需先跑 `main-pretrain.py`） |
| `main-diffusion.py` | 訓練條件式擴散模型（DDPM），生成手寫數字，輸出 `samples.png` |
| `tests.py` | 預訓練 / 微調元件、MPS 相容性、模型與擴散模型基本行為的測試 |
| `run.sh` | 依序執行測試與全部範例 |

套件 `nnlib/`：`models.py`（模型）、`train.py`（訓練、評估、取樣）、`diffusion.py`、`qa.py`（讀取 `finetune.txt`、切分、評估）、`checkpoint.py`、`data.py`、`viz.py`。

裝置會自動選擇 CUDA → Apple MPS → CPU，也可用 `DEVICE=cpu|mps|cuda` 指定。

資料檔：`input.txt`、`pretrain.txt`、`finetune.txt`。`pretrain.txt` 與 `finetune.txt` 都是少數幾種內容大量重複（各 50 種、每種 134 / 27 行），所以切分驗證資料時一定要以「不同的內容」為單位。

MNIST 範例需要 `mnist.npz` 放在此資料夾；沒有的話第一次執行會自動下載。
