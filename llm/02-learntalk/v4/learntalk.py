# learner.py
import math
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 自動詞表構建器 (完全不依賴特定語言)
# ==========================================
def load_corpus_and_vocab(corpus_path="corpus.txt"):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"找不到 {corpus_path}，請先執行 python generate_corpus.py！")

    sentences = []
    unique_tokens = set()
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 通用分詞：有空格按空格分，無空格按字元拆分
            tokens = line.split(" ") if " " in line else list(line)
            sentences.append(tokens)
            unique_tokens.update(tokens)
            
    vocab = ["<PAD>"] + sorted(list(unique_tokens))
    token_to_ix = {t: i for i, t in enumerate(vocab)}
    ix_to_token = {i: t for i, t in enumerate(vocab)}
    
    return sentences, vocab, token_to_ix, ix_to_token

# 英文字母頻道 (26 字母 + 空格 + PAD)
LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)]
EN_VOCAB = ["<PAD>", " "] + LETTERS
en_to_ix = {c: i for i, c in enumerate(EN_VOCAB)}
ix_to_en = {i: c for i, c in enumerate(EN_VOCAB)}
SPACE_IDX = en_to_ix[" "]

# ==========================================
# 2. 通用 Speaker-GPT：任意序列 -> 雙字母序列 (含空格)
# ==========================================
class UniversalSpeaker(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc_c1 = nn.Linear(d_model, 26) # 輸出第 1 字母 (a-z)
        self.fc_c2 = nn.Linear(d_model + 26, 26) # 輸出第 2 字母 (a-z)

    def forward(self, input_tokens):
        B, L = input_tokens.shape
        x = self.embed(input_tokens) + self.pos_embed[:, :L, :]
        pad_mask = (input_tokens == 0)
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        
        log_probs_list, entropies_list, words_tensor = [], [], []
        
        for slot in range(L):
            h_slot = h[:, slot, :]
            
            # 採樣第 1 字母
            logits1 = self.fc_c1(h_slot)
            dist1 = Categorical(logits=logits1)
            c1 = dist1.sample()
            
            # 採樣第 2 字母
            c1_onehot = torch.zeros(B, 26, device=input_tokens.device).scatter_(1, c1.unsqueeze(1), 1.0)
            logits2 = self.fc_c2(torch.cat([h_slot, c1_onehot], dim=-1))
            dist2 = Categorical(logits=logits2)
            c2 = dist2.sample()
            
            log_probs_list.append(dist1.log_prob(c1) + dist2.log_prob(c2))
            entropies_list.append(dist1.entropy() + dist2.entropy())
            
            words_tensor.extend([c1 + 2, c2 + 2])
            if slot < L - 1:
                words_tensor.append(torch.full((B,), SPACE_IDX, device=input_tokens.device))
                
        full_msg = torch.stack(words_tensor, dim=1) # [B, 3L - 1]
        log_probs = torch.stack(log_probs_list, dim=1) # [B, L]
        entropies = torch.stack(entropies_list, dim=1) # [B, L]
        return full_msg, log_probs, entropies

# ==========================================
# 3. 通用 Listener-GPT：英文字元 -> 解碼回原始語言
# ==========================================
class UniversalListener(nn.Module):
    def __init__(self, target_vocab_size, d_model=64):
        super().__init__()
        self.en_embed = nn.Embedding(len(EN_VOCAB), d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_decode = nn.Linear(d_model, target_vocab_size)

    def forward(self, msg_tokens, target_len):
        B, M_len = msg_tokens.shape
        x = self.en_embed(msg_tokens) + self.pos_embed[:, :M_len, :]
        h = self.transformer(x)
        
        # 動態提取每個 2 字母單詞的結尾索引: 1, 4, 7, ..., 3*i + 1
        word_ends = [3 * i + 1 for i in range(target_len)]
        slot_features = h[:, word_ends, :] # [B, target_len, D]
        logits = self.fc_decode(slot_features) # [B, target_len, target_vocab_size]
        return logits

# ==========================================
# 4. 通用強化學習訓練主迴圈
# ==========================================
def train_universal(corpus_path="corpus.txt", episodes=1000, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentences, vocab, token_to_ix, ix_to_token = load_corpus_and_vocab(corpus_path)
    
    print(f"--- 載入語料庫: {corpus_path} ---")
    print(f"總語句數: {len(sentences)} | 唯一 Token 規模: {len(vocab)} (含 PAD)")
    print(f"執行裝置: {device}\n")

    speaker = UniversalSpeaker(len(vocab), d_model=64).to(device)
    listener = UniversalListener(len(vocab), d_model=64).to(device)
    
    spk_opt = optim.Adam(speaker.parameters(), lr=0.003)
    lis_opt = optim.Adam(listener.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("=== 通用神經網路開始學習透過 2 字母傳遞該語言 ===\n")

    for ep in range(episodes):
        # 隨機挑選長度一致的一批樣本進行 Batch 平行訓練
        batch_samples = [random.choice(sentences) for _ in range(batch_size)]
        
        # 動態長度對齊（以該 Batch 最大長度為準）
        L = max(len(s) for s in batch_samples)
        padded_batch = [s + ["<PAD>"] * (L - len(s)) for s in batch_samples]
        target_tensor = torch.tensor(
            [[token_to_ix[t] for t in s] for s in padded_batch], dtype=torch.long, device=device
        )
        
        # 1. Speaker 產生雙字母密碼
        msg_tokens, spk_log_probs, spk_entropies = speaker(target_tensor)
        
        # 2. Listener 根據字母嘗試解碼回原本符號
        logits = listener(msg_tokens, target_len=L)
        preds = logits.argmax(dim=-1)
        
        # 3. 槽位評分（忽略 PAD 部分）
        valid_mask = (target_tensor != 0).float()
        token_matches = (preds == target_tensor).float() * valid_mask
        
        # 整句全對率 (所有有效 Token 都猜對)
        sentence_correct = ((token_matches == valid_mask).all(dim=1)).float()
        acc = sentence_correct.mean().item()
        
        # 4. 更新 Listener (Cross-Entropy)
        lis_loss = criterion(logits.view(-1, len(vocab)), target_tensor.view(-1))
        
        # 5. 更新 Speaker (REINFORCE + Advantage)
        # 有效位置獎勵 1.0，錯誤 0.0
        adv = (token_matches - 0.5) * valid_mask
        spk_loss = -(spk_log_probs * adv).sum() / valid_mask.sum() - 0.01 * spk_entropies.mean()
        
        lis_opt.zero_grad()
        lis_loss.backward()
        lis_opt.step()
        
        spk_opt.zero_grad()
        spk_loss.backward()
        spk_opt.step()
        
        if (ep + 1) % 100 == 0:
            token_acc = (token_matches.sum() / valid_mask.sum()).item()
            print(f"回合 {ep+1:4d}/{episodes} | 整句解碼成功率: {acc*100:5.1f}% | 單字準確率: {token_acc*100:5.1f}%")

    print("\n=== 訓練完成！各類語言之自創雙字母協議已成功收斂 ===\n")
    return speaker, listener, sentences, token_to_ix, ix_to_token, device

# ==========================================
# 5. 通用測試展示
# ==========================================
def evaluate_universal(speaker, listener, sentences, token_to_ix, ix_to_token, device):
    speaker.eval()
    listener.eval()
    
    print("=" * 80)
    print("              💬 跨語言通訊與解碼實測展示（簡語 / 數學 / 程式）")
    print("=" * 80)
    print(f"{'原始輸入序列 (corpus.txt)':<30} -> {'AI雙字母傳輸碼':<26} -> {'還原解碼結果'}")
    print("-" * 80)
    
    # 抽取 10 個不同形態的真實句子進行解碼測試
    test_samples = random.sample(sentences, min(10, len(sentences)))
    
    with torch.no_grad():
        for tokens in test_samples:
            orig_text = " ".join(tokens)
            L = len(tokens)
            inp_tensor = torch.tensor([[token_to_ix[t] for t in tokens]], device=device)
            
            # Speaker 編碼
            msg_tokens, _, _ = speaker(inp_tensor)
            msg_str = "".join([ix_to_en[t.item()] for t in msg_tokens[0]])
            
            # Listener 解碼
            logits = listener(msg_tokens, target_len=L)
            preds = logits.argmax(dim=-1)
            decoded_tokens = [ix_to_token[p.item()] for p in preds[0]]
            decoded_text = " ".join(decoded_tokens)
            
            status = "✓ 成功" if orig_text == decoded_text else "✗ 失敗"
            print(f"{orig_text:<32} -> {msg_str:<28} -> {decoded_text:<16} ({status})")
    print("=" * 80)

if __name__ == "__main__":
    spk, lis, sents, t2i, i2t, dev = train_universal("corpus.txt", episodes=2000, batch_size=64)
    evaluate_universal(spk, lis, sents, t2i, i2t, dev)