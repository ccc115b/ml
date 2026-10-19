"""檢查點：把 GPT 的權重、結構設定與字元表存成單一 .pt 檔，之後可重建並載入（預訓練 → 微調）。"""

import torch
from .models import GPT


def save_gpt(path, model, chars):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({'config': model.config, 'chars': ''.join(chars), 'state': state}, path)


def load_gpt(path, device='cpu'):
    """回傳 (model, chars)：結構與權重都還原並放到指定裝置，chars 是訓練時的字元表（順序固定）。"""
    blob = torch.load(path, map_location='cpu', weights_only=True)
    model = GPT(**blob['config'])
    model.load_state_dict(blob['state'])
    return model.to(device), list(blob['chars'])
