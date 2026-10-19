"""PyTorch 版的 MLP / CNN / GRU / GPT / 擴散模型；結構與超參數對應 minitorch 版。

支援 CUDA、Apple Silicon 的 MPS 與 CPU（見 train.get_device）。"""

import os

# 少數運算在 MPS 上尚未實作時，自動退回 CPU 執行而不是報錯。必須在 import torch 之前設定。
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
