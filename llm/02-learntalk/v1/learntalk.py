import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 符號集 (a-z) 與 環境世界事物
# ==========================================
LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)] # a 到 z 共 26 個字母
VOCAB = ["<PAD>", "<SOS>"] + LETTERS
char_to_ix = {ch: i for i, ch in enumerate(VOCAB)}
ix_to_char = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# 現實世界中的 8 個概念（完全不告訴 AI 字母和這些有何關係）
CONCEPTS = ["SUN", "MOON", "WATER", "FIRE", "TREE", "BIRD", "FISH", "STONE"]
NUM_CONCEPTS = len(CONCEPTS)
MSG_LEN = 2  # 規定每個單字由 2 個字母組成 (例如: "ab", "qz")

# ==========================================
# 2. 模型架構：Speaker-GPT 與 Listener-GPT
# ==========================================
class SpeakerGPT(nn.Module):
    """說話者：看到事物 -> 用 a-z 生成單字 (自回歸 Transformer)"""
    def __init__(self, vocab_size, num_concepts, d_model=32):
        super().__init__()
        self.concept_embed = nn.Embedding(num_concepts, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, MSG_LEN + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, concept_ids):
        batch_size = concept_ids.size(0)
        c_emb = self.concept_embed(concept_ids).unsqueeze(1) # [B, 1, D]
        
        curr_tokens = torch.full((batch_size, 1), char_to_ix["<SOS>"], dtype=torch.long, device=concept_ids.device)
        log_probs = []
        sampled_tokens = []
        
        for t in range(MSG_LEN):
            t_emb = self.token_embed(curr_tokens)
            x = torch.cat([c_emb, t_emb], dim=1) # 結合看到的秘密事物與當前產生的字母
            x = x + self.pos_embed[:, :x.size(1), :]
            
            # 因果注意力遮罩
            sz = x.size(1)
            mask = torch.triu(torch.ones(sz, sz, device=x.device) * float("-inf"), diagonal=1)
            out = self.transformer(x, mask=mask)
            
            logits = self.fc(out[:, -1, :]) # 只取最後一個 token
            # 遮蔽 PAD 和 SOS，強制只輸出 a-z
            logits[:, char_to_ix["<PAD>"]] = -1e9
            logits[:, char_to_ix["<SOS>"]] = -1e9
            
            dist = Categorical(logits=logits)
            action = dist.sample() # 依機率抽樣字母 (RL 探索)
            
            log_probs.append(dist.log_prob(action))
            sampled_tokens.append(action)
            curr_tokens = torch.cat([curr_tokens, action.unsqueeze(1)], dim=1)
            
        return torch.stack(sampled_tokens, dim=1), torch.stack(log_probs, dim=1)

class ListenerGPT(nn.Module):
    """聆聽者：接收 a-z 字母串 -> 猜測背後的概念事物"""
    def __init__(self, vocab_size, num_concepts, d_model=32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, MSG_LEN, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_concepts)

    def forward(self, msg_tokens):
        x = self.token_embed(msg_tokens) + self.pos_embed[:, :msg_tokens.size(1), :]
        out = self.transformer(x)
        logits = self.fc(out[:, -1, :]) # 讀完單字後做決策
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

# ==========================================
# 3. 強化學習自我博弈訓練 (REINFORCE)
# ==========================================
def train_emergent_language(episodes=1500, batch_size=64):
    speaker = SpeakerGPT(VOCAB_SIZE, NUM_CONCEPTS)
    listener = ListenerGPT(VOCAB_SIZE, NUM_CONCEPTS)
    
    spk_optimizer = optim.Adam(speaker.parameters(), lr=0.003)
    lis_optimizer = optim.Adam(listener.parameters(), lr=0.003)
    
    baseline = 1.0 / NUM_CONCEPTS # 初始基準線 (隨機猜對機率 1/8 = 0.125)
    print("--- 兩個 GPT 開始從零摸索，自我學習發明語言 ---")
    
    for ep in range(episodes):
        # 1. 環境隨機挑選一批事物給 Speaker 看
        targets = torch.randint(0, NUM_CONCEPTS, (batch_size,))
        
        # 2. Speaker 說話 (產生 2 個字母組成的詞彙)
        msg_tokens, spk_log_probs = speaker(targets)
        
        # 3. Listener 聽話並猜測目標事物
        predictions, lis_log_prob, lis_entropy = listener(msg_tokens)
        
        # 4. 計算獎勵：猜對得 1 分，猜錯得 0 分
        rewards = (predictions == targets).float()
        accuracy = rewards.mean().item()
        
        # 5. 政策梯度更新 (Policy Gradient with Baseline)
        advantages = rewards - baseline
        baseline = 0.95 * baseline + 0.05 * accuracy # 平滑更新基準線
        
        # Speaker Loss (兩個字母的 log prob 總和)
        spk_loss = -(spk_log_probs.sum(dim=1) * advantages).mean()
        # Listener Loss (加上熵正則化鼓勵初期多嘗試不同答案)
        lis_loss = -(lis_log_prob * advantages).mean() - 0.01 * lis_entropy.mean()
        
        spk_optimizer.zero_grad()
        spk_loss.backward()
        spk_optimizer.step()
        
        lis_optimizer.zero_grad()
        lis_loss.backward()
        lis_optimizer.step()
        
        if (ep + 1) % 150 == 0:
            print(f"訓練回合 {ep+1:4d}/{episodes} | 溝通成功率: {accuracy * 100:5.1f}%")

    print("--- 訓練完成！語言共識已達成 ---\n")
    return speaker, listener

# ==========================================
# 4. 解析「AI 自創字典」
# ==========================================
def inspect_emergent_dictionary(speaker, listener):
    speaker.eval()
    listener.eval()
    
    print("======================================================")
    print("       🔍 兩位 GPT 達成共識的【自創字典】")
    print("======================================================")
    print(f"{'事物 (真實世界概念)':<20} -> {'AI 自創詞彙':<12} -> {'對方理解結果'}")
    print("-" * 54)
    
    with torch.no_grad():
        for cid in range(NUM_CONCEPTS):
            concept_name = CONCEPTS[cid]
            input_cid = torch.tensor([cid])
            
            # Speaker 生成單字
            tokens, _ = speaker(input_cid)
            word = "".join([ix_to_char[t.item()] for t in tokens[0]])
            
            # Listener 根據這個單字進行解讀
            pred, _, _ = listener(tokens)
            understood_name = CONCEPTS[pred.item()]
            
            status = "✓ 成功" if concept_name == understood_name else "✗ 失敗"
            print(f"{concept_name:<20} -> {word:<12} -> {understood_name:<10} ({status})")
    print("======================================================\n")

if __name__ == "__main__":
    spk, lis = train_emergent_language(episodes=1500, batch_size=64)
    inspect_emergent_dictionary(spk, lis)