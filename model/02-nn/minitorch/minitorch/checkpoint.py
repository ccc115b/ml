"""檢查點：把 GPT 的權重、結構設定與字元表存成單一 .npz 檔，之後可重建並載入（預訓練 → 微調）。"""

import json
import numpy as np
from .models import GPT


def save_gpt(path, model, chars):
    np.savez(path, __config__=np.array(json.dumps(model.config)), __chars__=np.array(''.join(chars)),
             **model.state_dict())


def load_gpt(path):
    """回傳 (model, chars)：結構與權重都還原，chars 是訓練時的字元表（順序固定）。"""
    with np.load(path) as file:
        config = json.loads(str(file['__config__']))
        chars = list(str(file['__chars__']))
        state = {name: file[name] for name in file.files if not name.startswith('__')}
    model = GPT(**config)
    model.load_state_dict(state)
    return model, chars
