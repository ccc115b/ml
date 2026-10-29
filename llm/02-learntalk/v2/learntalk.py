import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 中文簡語語法與世界事件
# ==========================================
SUBJECTS = ["我", "你", "人"]
ACTIONS  = ["食", "視", "行"]
OBJECTS  = ["水", "火", "物"]

# 27 種客觀中文事件（每句恰好 5 個中文字）
JIAN_SENTENCES = [
    [s, "做", v, "於", o] 
    for s in SUBJECTS 
    for v in ACTIONS 
    for o in OBJECTS
]
NUM_SENTENCES = len(JIAN_SENTENCES)

CH_WORDS = ["<PAD>", "我", "你", "人", "做", "食", "視", "行", "於", "水", "火", "物"]
ch_to_ix = {w: i for i, w in enumerate(CH_WORDS)}
ix_to_ch = {i: w for i, w in enumerate(CH_WORDS)}
CH_VOCAB_SIZE = len(CH_WORDS)

# 英文字母頻道 (a-z + 空格)
LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)]
EN_VOCAB = ["<PAD>", " "] + LETTERS
en_to_ix = {c: i for i, c in enumerate(EN_VOCAB)}
ix_to_en = {i: c for i, c in enumerate(EN_VOCAB)}
EN_VOCAB_SIZE = len(EN_VOCAB)
SPACE_IDX = en_to_ix[" "]

# ==========================================
# 2. Speaker-GPT：5 個中文字 -> 5 個雙字母單詞
# ==========================================
class DualLetterSpeaker(nn.Module):
    def __init__(self, ch_vocab_size, d_model=64):
        super().__init__()
        self.ch_embed = nn.Embedding(ch_vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, d_model))
        
        # 使用偶數 head=4，消除 PyTorch 警告
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc_char1 = nn.Linear(d_model, 26) # 輸出 a-z (第 1 字母)
        self.fc_char2 = nn.Linear(d_model + 26, 26) # 輸出 a-z (第 2 字母)

    def forward(self, ch_indices):
        batch_size = ch_indices.size(0)
        x = self.ch_embed(ch_indices) + self.pos_embed
        h = self.transformer(x) # [B, 5, D]
        
        log_probs_list = []
        entropies_list = []
        words_tensor = []
        
        for slot in range(5):
            h_slot = h[:, slot, :]
            
            # 第 1 字母
            logits1 = self.fc_char1(h_slot)
            dist1 = Categorical(logits=logits1)
            c1 = dist1.sample()
            
            # 第 2 字母 (Condition on 第 1 字母)
            c1_onehot = torch.zeros(batch_size, 26, device=ch_indices.device).scatter_(1, c1.unsqueeze(1), 1.0)
            logits2 = self.fc_char2(torch.cat([h_slot, c1_onehot], dim=-1))
            dist2 = Categorical(logits=logits2)
            c2 = dist2.sample()
            
            slot_log_p = dist1.log_prob(c1) + dist2.log_prob(c2)
            slot_ent = dist1.entropy() + dist2.entropy()
            
            log_probs_list.append(slot_log_p)
            entropies_list.append(slot_ent)
            
            # 轉換為 EN_VOCAB 索引（+2 跳過 PAD 與 SPACE）
            words_tensor.extend([c1 + 2, c2 + 2])
            if slot < 4:
                words_tensor.append(torch.full((batch_size,), SPACE_IDX, device=ch_indices.device))
                
        full_msg = torch.stack(words_tensor, dim=1) # [B, 14] -> 形如 "ab cd ef gh ij"
        log_probs = torch.stack(log_probs_list, dim=1) # [B, 5]
        entropies = torch.stack(entropies_list, dim=1) # [B, 5]
        return full_msg, log_probs, entropies

# ==========================================
# 3. Listener-GPT：接收 14 字符 -> 解碼 5 個中文字
# ==========================================
class DualLetterListener(nn.Module):
    def __init__(self, en_vocab_size, ch_vocab_size, d_model=64):
        super().__init__()
        self.en_embed = nn.Embedding(en_vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 14, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_decode = nn.Linear(d_model, ch_vocab_size)

    def forward(self, msg_tokens):
        x = self.en_embed(msg_tokens) + self.pos_embed
        h = self.transformer(x)
        
        # 提取每個 2 字母單詞的結尾位置 (索引: 1, 4, 7, 10, 13)
        word_end_indices = [1, 4, 7, 10, 13]
        slot_features = h[:, word_end_indices, :] # [B, 5, D]
        logits = self.fc_decode(slot_features)    # [B, 5, CH_VOCAB_SIZE]
        return logits

# ==========================================
# 4. 強化學習自我博弈 (Slot-level Credit Assignment)
# ==========================================
def train_rl(episodes=1000, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    speaker = DualLetterSpeaker(CH_VOCAB_SIZE, d_model=64).to(device)
    listener = DualLetterListener(EN_VOCAB_SIZE, CH_VOCAB_SIZE, d_model=64).to(device)
    
    spk_opt = optim.Adam(speaker.parameters(), lr=0.003)
    lis_opt = optim.Adam(listener.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    all_sentences_tensor = torch.tensor(
        [[ch_to_ix[w] for w in sent] for sent in JIAN_SENTENCES], dtype=torch.long, device=device
    )
    
    # 5 個槽位各自獨立的 baseline
    baseline = torch.zeros(5, device=device)
    
    print("=== 兩個 GPT 開始摸索：發明【雙字母對應單一中文字】的通訊協定 ===")
    
    for ep in range(episodes):
        target_ids = torch.randint(0, NUM_SENTENCES, (batch_size,), device=device)
        target_ch = all_sentences_tensor[target_ids] # [B, 5]
        
        # 1. Speaker 說話
        msg_tokens, spk_log_probs, spk_entropies = speaker(target_ch)
        
        # 2. Listener 聽話並輸出預測機率
        logits = listener(msg_tokens) # [B, 5, CH_VOCAB_SIZE]
        preds = logits.argmax(dim=-1) # [B, 5]
        
        # 3. 槽位級信用分配 (Slot-level Reward)
        slot_matches = (preds == target_ch).float() # [B, 5]
        perfect_sentence = (slot_matches.sum(dim=1) == 5.0).float()
        acc = perfect_sentence.mean().item()
        
        # Listener Loss (直接學習將接收到的字母對應到真實目標，提供穩定梯度)
        lis_loss = criterion(logits.view(-1, CH_VOCAB_SIZE), target_ch.view(-1))
        
        # Speaker Loss (REINFORCE：每個單詞只對自己的槽位獎勵負責)
        adv = slot_matches - baseline.unsqueeze(0) # [B, 5]
        baseline = 0.95 * baseline + 0.05 * slot_matches.mean(dim=0).detach()
        spk_loss = -(spk_log_probs * adv).mean() - 0.01 * spk_entropies.mean()
        
        # 更新網路
        lis_opt.zero_grad()
        lis_loss.backward()
        lis_opt.step()
        
        spk_opt.zero_grad()
        spk_loss.backward()
        spk_opt.step()
        
        if (ep + 1) % 100 == 0:
            slot_accs = [f"{slot_matches[:, s].mean().item()*100:.0f}%" for s in range(5)]
            print(f"回合 {ep+1:4d}/{episodes} | 整句全對率: {acc*100:5.1f}% | 各字準確率 [主,做,動,於,賓]: {slot_accs}")

    print("=== 訓練完成！雙字母單詞對應表已成功鎖定 ===\n")
    return speaker, listener, device, all_sentences_tensor

# ==========================================
# 5. 輸出 AI 發明的單字字典與實測對談
# ==========================================
def inspect_results(speaker, listener, device, all_sentences_tensor):
    speaker.eval()
    listener.eval()
    
    print("=" * 72)
    print("      📖 兩位 GPT 達成的【中文字 ↔ 雙字母單詞】對應字典")
    print("=" * 72)
    
    word_mapping = {}
    with torch.no_grad():
        for i in range(NUM_SENTENCES):
            target_ch = all_sentences_tensor[i].unsqueeze(0)
            msg_tokens, _, _ = speaker(target_ch)
            msg_str = "".join([ix_to_en[t.item()] for t in msg_tokens[0]])
            words = msg_str.split(" ")
            
            for ch_w, en_w in zip(JIAN_SENTENCES[i], words):
                word_mapping[ch_w] = en_w

    for ch_word in ["我", "你", "人", "做", "食", "視", "行", "於", "水", "火", "物"]:
        print(f"中文字『{ch_word}』 ---> AI 自創雙字母單詞: 【 {word_mapping.get(ch_word, '??')} 】")
    
    print("\n" + "=" * 72)
    print("            💬 完整句子傳遞與解碼實測（節錄 10 句）")
    print("=" * 72)
    print(f"{'中文輸入 (現實世界)':<18} -> {'AI雙字母訊息 (空格隔開)':<22} -> {'對方解讀還原'}")
    print("-" * 72)
    
    with torch.no_grad():
        for i in range(10):
            target_ch = all_sentences_tensor[i].unsqueeze(0)
            target_text = "".join(JIAN_SENTENCES[i])
            
            msg_tokens, _, _ = speaker(target_ch)
            msg_str = "".join([ix_to_en[t.item()] for t in msg_tokens[0]])
            
            logits = listener(msg_tokens)
            preds = logits.argmax(dim=-1)
            understood_text = "".join([ix_to_ch[t.item()] for t in preds[0]])
            
            status = "✓ 成功" if target_text == understood_text else "✗ 失敗"
            print(f"{target_text:<20} -> {msg_str:<24} -> {understood_text:<14} ({status})")
    print("=" * 72)

if __name__ == "__main__":
    spk, lis, dev, all_sents = train_rl(episodes=1000, batch_size=64)
    inspect_results(spk, lis, dev, all_sents)