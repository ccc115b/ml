import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 詞表與語料生成（修正 <PAD> 的加入方式）
# ==========================================
# 36 個核心字 + 3 個標點
BASE_CHARS = list("我你他此人物地時日水火言食行有知視聽感愛作大小好壞同異多少做於之以不麼昔將。？！")
# 將 <PAD> 作為獨立字詞加入，避免被 list() 拆碎
CHARS = BASE_CHARS + ["<PAD>"]

char_to_ix = {ch: i for i, ch in enumerate(CHARS)}
ix_to_char = {i: ch for i, ch in enumerate(CHARS)}
vocab_size = len(CHARS)
PAD_IDX = char_to_ix["<PAD>"]


def generate_sentence():
    time_m = random.choice(["", "昔", "將"])
    subject = random.choice(["我", "你", "他", "此", "人", "物", "地", "時"])
    if random.random() < 0.3:
        subject = random.choice(["好", "大", "小"]) + "之" + subject

    neg = "不" if random.random() < 0.2 else ""
    action = random.choice(["言", "食", "行", "有", "知", "視", "聽", "感", "愛", "作"])

    obj = ""
    if random.random() < 0.6:
        obj_core = random.choice(["人", "物", "地", "時", "日", "水", "火"])
        obj = f"於{obj_core}"

    is_q = random.random() < 0.25
    punct = "？" if is_q else random.choice(["。", "！"])
    q_mark = "麼" if is_q else ""

    return f"{time_m}{subject}{neg}做{action}{obj}{q_mark}{punct}"


# 建立 Dataset 與 DataLoader（批次訓練）
class JianDataset(Dataset):
    def __init__(self, data_size=2400):
        self.sentences = [generate_sentence() for _ in range(data_size)]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return [char_to_ix[c] for c in self.sentences[idx]]


def collate_fn(batch):
    max_len = max(len(s) for s in batch)
    padded = [s + [PAD_IDX] * (max_len - len(s)) for s in batch]
    tensor = torch.tensor(padded, dtype=torch.long)
    return tensor[:, :-1], tensor[:, 1:]


# ==========================================
# 2. Transformer 模型
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz, device=x.device) * float("-inf"), diagonal=1)
        out = self.embedding(x) * math.sqrt(64)
        out = self.pos_encoder(out)
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc(out)


# ==========================================
# 3. 訓練函數（Batch 化 + Loss 忽略 PAD）
# ==========================================
def train_transformer(name="Agent", epochs=10):
    dataset = JianDataset(data_size=2400)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = MiniTransformerLM(vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    print(f"--- 開始訓練 {name} (Batch Size=32) ---")
    model.train()
    for epoch in range(epochs):
        total_loss, total_steps = 0.0, 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        print(f"[{name}] Epoch {epoch+1:2d}/{epochs} | 平均 Loss: {avg_loss:.4f}")

    print(f"--- {name} 訓練完成 ---\n")
    return model


# ==========================================
# 4. 生成邏輯（加入重複懲罰避免復讀）
# ==========================================
def generate_text(model, seed_str, max_len=18, temperature=0.7, repetition_penalty=1.5):
    model.eval()
    with torch.no_grad():
        input_ids = [char_to_ix[c] for c in seed_str]
        result = seed_str

        for _ in range(max_len):
            x = torch.tensor([input_ids], dtype=torch.long)
            logits = model(x)[0, -1, :]

            # 重複懲罰：對已經生成的字降低權重，避免「於於於於」死迴圈
            for prev_idx in set(input_ids):
                if logits[prev_idx] > 0:
                    logits[prev_idx] /= repetition_penalty
                else:
                    logits[prev_idx] *= repetition_penalty

            probs = torch.softmax(logits / temperature, dim=-1)
            # 排除 PAD 不被隨機抽到
            probs[PAD_IDX] = 0.0
            probs /= probs.sum()

            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = ix_to_char[next_idx]

            result += next_char
            input_ids.append(next_idx)

            if next_char in ["。", "？", "！"]:
                break

        return result


# ==========================================
# 5. 主程式：對話交替模擬
# ==========================================
if __name__ == "__main__":
    agent_a = train_transformer(name="Agent-A", epochs=10)
    agent_b = train_transformer(name="Agent-B", epochs=10)

    print("=== Transformer 語言模型對話模擬 (修復複讀問題) ===")

    speakers = [
        ("🤖 Agent-A", agent_a, "你"),
        ("🤖 Agent-B", agent_b, "我")
    ]

    current_prompt = "你做"
    print(f"[初始引導]：{current_prompt}\n")

    for turn in range(1, 10):
        for name, model, reply_pronoun in speakers:
            utterance = generate_text(model, current_prompt)
            print(f"回合 {turn} - {name}: {utterance}")

            # 引導下一個 Agent 採用合適的主語開頭接話
            current_prompt = f"{reply_pronoun}做"