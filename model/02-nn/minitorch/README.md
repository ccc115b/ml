# minitorch

以 NumPy 實作的**小型 torch**：自動微分（autograd）、Module 系統、優化器、DataLoader。模型只寫前向，反向傳播由引擎自動完成。

| 腳本 | 內容 |
|---|---|
| `mlp.py` `cnn.py` | 訓練 MNIST 手寫數字辨識 |
| `gru.py` `gpt.py` | 訓練字元級語言模型（`input.txt`），並生成句子 |
| `pretrain.py` `finetune.py` | GPT 兩階段訓練：先在 `pretrain.txt` 上預訓練並存成 `pretrained.npz`，再用 `finetune.txt` 的問答（`<Q>問題<A>答案`）微調；以 5 折交叉驗證和「不預訓練」、「部分凍結」比較（需先跑 `pretrain.py`） |
| `diffusion.py` | 訓練條件式擴散模型（DDPM），生成手寫數字，輸出 `samples.png` |
| `tests.py` | 每個算子與模型的數值梯度檢查，以及 Module、優化器、DataLoader、擴散模型、預訓練與微調的行為檢查 |
| `run.sh` | 依序執行測試與全部範例；`./run.sh gpt` 只跑指定的 |

套件 `minitorch/`：

| 模組 | 內容 |
|---|---|
| `tensor.py` | `Tensor`、計算圖、`backward()`、`no_grad`、`gradcheck` |
| `functional.py` | 激活函數、`cross_entropy`、`conv2d`、`max_pool2d`、`upsample2x` |
| `layers.py` `init.py` | `Module` 與 Linear、Embedding、Conv2d、LayerNorm、GRU、注意力等層、權重初始化 |
| `optim.py` | SGD、Adam、AdamW、學習率排程、梯度裁剪 |
| `models.py` | MLP、CNN、GRU 語言模型、GPT、UNet |
| `diffusion.py` `train.py` | DDPM 排程與取樣、各範例的訓練流程 |
| `checkpoint.py` `qa.py` | GPT 檢查點存取（權重、結構設定、字元表）；讀取 `finetune.txt`、依「不同的問答」切分、答案遮罩、候選答案名次評估 |
| `data.py` `viz.py` | 資料載入與 DataLoader、PNG / ASCII 輸出 |

資料檔：`input.txt`（`gru.py`、`gpt.py`）、`pretrain.txt`、`finetune.txt`（兩階段訓練）。
注意 `pretrain.txt` 與 `finetune.txt` 都是少數幾種內容大量重複（各 50 種、每種 134 / 27 行），所以切分驗證資料時一定要以「不同的內容」為單位。

MNIST 範例需要 `mnist.npz` 放在此資料夾；沒有的話第一次執行會自動下載。

同系列專案：`numpy-nn`（手寫反向傳播版，腳本名稱相同）、`pytorch-nn`（PyTorch 版）。
