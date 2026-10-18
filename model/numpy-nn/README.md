# numpy-nn

以 NumPy **手寫反向傳播**的小型神經網路：每個模型都自己推導並實作矩陣梯度，不使用自動微分。

| 腳本 | 內容 |
|---|---|
| `mlp.py` `cnn.py` | 訓練 MNIST 手寫數字辨識 |
| `gru.py` `gpt.py` | 訓練字元級語言模型（`input.txt`），並生成句子 |
| `tests.py` | 手寫梯度的數值梯度檢查，以及訓練、取樣的基本行為檢查 |
| `run.sh` | 依序執行測試與全部範例；`./run.sh gpt` 只跑指定的 |

套件 `npnn/`：`models.py`（四個模型與其梯度）、`train.py`（Adam、訓練迴圈、取樣）、`data.py`（資料載入）。

MNIST 範例需要 `mnist.npz` 放在此資料夾；沒有的話第一次執行會自動下載。

同系列專案：`minitorch`（自動微分版，腳本名稱相同）、`pytorch-nn`（PyTorch 版）。
