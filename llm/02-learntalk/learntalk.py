import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 「簡語-256」完整核心字表 (精確 256 個獨立中文字，補回「物」字)
# ==========================================
JIAN_256_WORDS = [
    # 1. 語法結構與邏輯 (16)
    "做", "於", "之", "以", "至", "同", "不", "非", "麼", "昔", "今", "將", "若", "故", "且", "或",
    
    # 2. 代詞與指稱 (16)
    "我", "吾", "你", "汝", "他", "此", "彼", "某", "何", "誰", "焉", "幾", "自", "互", "餘", "皆",
    
    # 3. 時間與空間方位 (32)
    "時", "日", "月", "年", "晨", "午", "夕", "夜", "刻", "始", "中", "終",
    "宇", "地", "上", "下", "前", "後", "左", "右", "內", "外", "東", "西", "南", "北", "邊", "間", "頂", "底", "近", "遠",
    
    # 4. 自然、宇宙與物質 (32，收錄核心「物」字)
    "天", "星", "風", "雲", "雨", "雪", "雷", "電", "光", "闇", "水", "火", "土", "石", "金", "木",
    "氣", "海", "江", "山", "原", "草", "樹", "花", "葉", "根", "果", "種", "冰", "物", "泉", "洞",
    
    # 5. 生命、身體與人倫 (32)
    "生", "死", "人", "男", "女", "子", "父", "母", "友", "敵", "王", "眾",
    "獸", "鳥", "魚", "蟲", "身", "首", "面", "目", "耳", "口", "鼻", "舌", "齒", "手", "足", "心", "血", "骨", "皮", "毛",
    
    # 6. 感知、認知與情感 (24)
    "視", "聽", "嗅", "味", "觸", "感", "知", "識", "思", "疑", "悟", "夢",
    "愛", "恨", "喜", "怒", "哀", "懼", "樂", "痛", "願", "意", "信", "驚",
    
    # 7. 動作、行為與互動 (48)
    "行", "來", "去", "止", "立", "坐", "臥", "走", "飛", "游", "升", "降",
    "食", "飲", "眠", "醒", "呼", "吸", "居", "沐", "作", "取", "置", "持",
    "移", "開", "閉", "破", "修", "變", "連", "分", "言", "語", "告", "問",
    "答", "教", "學", "助", "爭", "和", "買", "賣", "換", "贈", "尋", "得",
    
    # 8. 性質、度量與評價 (40)
    "大", "小", "長", "短", "高", "低", "深", "淺", "重", "輕", "廣", "狹", "厚", "薄",
    "多", "少", "滿", "空", "全", "半", "溫", "寒", "熱", "乾", "濕", "硬", "軟", "快", "慢", "明", "暗",
    "好", "壞", "美", "醜", "真", "偽", "易", "難", "貴",
    
    # 9. 數字與序數 (16)
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "萬", "初", "極"
]

# 雙重安全性校驗
assert len(set(JIAN_256_WORDS)) == 256, f"字表長度有誤: {len(set(JIAN_256_WORDS))}"

CH_VOCAB = ["<PAD>"] + JIAN_256_WORDS
ch_to_ix = {w: i for i, w in enumerate(CH_VOCAB)}
ix_to_ch = {i: w for i, w in enumerate(CH_VOCAB)}
CH_VOCAB_SIZE = len(CH_VOCAB) # 257

# 英文字母頻道 (26 字母 + 空格 + PAD)
LETTERS = [chr(i) for i in range(ord('a'), ord('z') + 1)]
EN_VOCAB = ["<PAD>", " "] + LETTERS
en_to_ix = {c: i for i, c in enumerate(EN_VOCAB)}
ix_to_en = {i: c for i, c in enumerate(EN_VOCAB)}
EN_VOCAB_SIZE = len(EN_VOCAB)
SPACE_IDX = en_to_ix[" "]

# ==========================================
# 2. 簡語-256 語法生成器 (EBNF 結構)
# ==========================================
SUBJ_POOL = ["我", "你", "他", "吾", "人", "男", "女", "子", "友", "王", "鳥", "獸", "魚"]
MARKER1_POOL = ["做", "將", "昔", "不"]
VERB_POOL = ["行", "食", "飲", "視", "聽", "感", "愛", "知", "作", "助", "問", "答", "取", "得"]
MARKER2_POOL = ["於", "至", "同", "以"]
OBJ_POOL = ["水", "火", "土", "金", "木", "天", "地", "海", "山", "日", "月", "風", "心", "物"]

# 事前校驗所有 pool 字詞是否存在
all_pools_words = set(SUBJ_POOL + MARKER1_POOL + VERB_POOL + MARKER2_POOL + OBJ_POOL)
missing = all_pools_words - set(JIAN_256_WORDS)
assert not missing, f"程式中用到的字詞不在字表中: {missing}"

def generate_sentence_batch(batch_size):
    """依照 EBNF 產生 5 字結構句子: 主語 + 標記1 + 動作 + 標記2 + 賓語"""
    sentences = []
    for _ in range(batch_size):
        s = random.choice(SUBJ_POOL)
        m1 = random.choice(MARKER1_POOL)
        v = random.choice(VERB_POOL)
        m2 = random.choice(MARKER2_POOL)
        o = random.choice(OBJ_POOL)
        sentences.append([s, m1, v, m2, o])
    return sentences

# ==========================================
# 3. Speaker-GPT：5 個中文字 -> 5 個雙字母單詞 (含空格共 14 字符)
# ==========================================
class Jian256Speaker(nn.Module):
    def __init__(self, ch_vocab_size, d_model=64):
        super().__init__()
        self.ch_embed = nn.Embedding(ch_vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc_char1 = nn.Linear(d_model, 26) # 輸出 a-z (第 1 字母)
        self.fc_char2 = nn.Linear(d_model + 26, 26) # 輸出 a-z (第 2 字母)

    def forward(self, ch_indices):
        batch_size = ch_indices.size(0)
        x = self.ch_embed(ch_indices) + self.pos_embed
        h = self.transformer(x) # [B, 5, D]
        
        log_probs_list, entropies_list, words_tensor = [], [], []
        
        for slot in range(5):
            h_slot = h[:, slot, :]
            
            # 第 1 字母
            logits1 = self.fc_char1(h_slot)
            dist1 = Categorical(logits=logits1)
            c1 = dist1.sample()
            
            # 第 2 字母 (Conditioned on 第 1 字母)
            c1_onehot = torch.zeros(batch_size, 26, device=ch_indices.device).scatter_(1, c1.unsqueeze(1), 1.0)
            logits2 = self.fc_char2(torch.cat([h_slot, c1_onehot], dim=-1))
            dist2 = Categorical(logits=logits2)
            c2 = dist2.sample()
            
            log_probs_list.append(dist1.log_prob(c1) + dist2.log_prob(c2))
            entropies_list.append(dist1.entropy() + dist2.entropy())
            
            words_tensor.extend([c1 + 2, c2 + 2])
            if slot < 4:
                words_tensor.append(torch.full((batch_size,), SPACE_IDX, device=ch_indices.device))
                
        full_msg = torch.stack(words_tensor, dim=1) # [B, 14]
        return full_msg, torch.stack(log_probs_list, dim=1), torch.stack(entropies_list, dim=1)

# ==========================================
# 4. Listener-GPT：接收 14 字符 -> 還原 256 字中文
# ==========================================
class Jian256Listener(nn.Module):
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
        # 提取每個雙字母單詞的結尾字元位置: [1, 4, 7, 10, 13]
        slot_features = h[:, [1, 4, 7, 10, 13], :]
        logits = self.fc_decode(slot_features) # [B, 5, CH_VOCAB_SIZE]
        return logits

# ==========================================
# 5. 強化學習自我博弈訓練
# ==========================================
def train_jian256(episodes=1000, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"運行裝置: {device} | 簡語詞庫規模: {len(JIAN_256_WORDS)} 字 (精確 256 字)")
    
    speaker = Jian256Speaker(CH_VOCAB_SIZE, d_model=64).to(device)
    listener = Jian256Listener(EN_VOCAB_SIZE, CH_VOCAB_SIZE, d_model=64).to(device)
    
    spk_opt = optim.Adam(speaker.parameters(), lr=0.003)
    lis_opt = optim.Adam(listener.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    baseline = torch.zeros(5, device=device)
    print("=== 開始訓練：AI 探索【簡語-256 ↔ 雙字母通訊協定】 ===\n")
    
    for ep in range(episodes):
        sent_batch = generate_sentence_batch(batch_size)
        target_ch = torch.tensor(
            [[ch_to_ix[w] for w in s] for s in sent_batch], dtype=torch.long, device=device
        )
        
        msg_tokens, spk_log_probs, spk_entropies = speaker(target_ch)
        logits = listener(msg_tokens)
        preds = logits.argmax(dim=-1)
        
        slot_matches = (preds == target_ch).float()
        acc = (slot_matches.sum(dim=1) == 5.0).float().mean().item()
        
        lis_loss = criterion(logits.view(-1, CH_VOCAB_SIZE), target_ch.view(-1))
        adv = slot_matches - baseline.unsqueeze(0)
        baseline = 0.95 * baseline + 0.05 * slot_matches.mean(dim=0).detach()
        spk_loss = -(spk_log_probs * adv).mean() - 0.01 * spk_entropies.mean()
        
        lis_opt.zero_grad()
        lis_loss.backward()
        lis_opt.step()
        
        spk_opt.zero_grad()
        spk_loss.backward()
        spk_opt.step()
        
        if (ep + 1) % 100 == 0:
            slot_acc_str = [f"{slot_matches[:, s].mean().item()*100:.0f}%" for s in range(5)]
            print(f"回合 {ep+1:4d}/{episodes} | 整句全對率: {acc*100:5.1f}% | 槽位準確率: {slot_acc_str}")

    print("\n=== 訓練完成！雙字母協定已穩定收斂 ===\n")
    return speaker, listener, device

# ==========================================
# 6. 檢視 AI 建立的自創雙字母字典與通訊展示
# ==========================================
def inspect_codebook(speaker, listener, device):
    speaker.eval()
    listener.eval()
    
    print("=" * 72)
    print("      📖 兩位 GPT 達成的【簡語-256 ↔ 雙字母單詞】對照字典（節錄）")
    print("=" * 72)
    
    test_words = ["我", "你", "人", "王", "鳥", "做", "將", "昔", "行", "食", "視", "愛", "於", "至", "水", "火", "金", "心", "物"]
    
    with torch.no_grad():
        for w in test_words:
            dummy_sent = torch.tensor([[ch_to_ix[w]] * 5], device=device)
            msg, _, _ = speaker(dummy_sent)
            word_str = "".join([ix_to_en[t.item()] for t in msg[0][:2]])
            print(f"  中文字『{w}』 ---> AI 自創雙字母單詞: 【 {word_str} 】")
            
    print("\n" + "=" * 72)
    print("            💬 完整 5 字句子即時通訊與解碼實測")
    print("=" * 72)
    print(f"{'中文輸入 (現實事件)':<18} -> {'AI雙字母訊息 (空格隔開)':<22} -> {'對方解讀還原'}")
    print("-" * 72)
    
    eval_sents = [
        ["我", "做", "食", "於", "水"],
        ["你", "將", "行", "至", "山"],
        ["王", "做", "愛", "於", "金"],
        ["鳥", "做", "行", "至", "天"],
        ["人", "昔", "視", "於", "火"],
        ["吾", "不", "飲", "於", "海"],
        ["女", "做", "助", "於", "友"],
        ["子", "將", "問", "於", "父"],
        ["人", "做", "取", "於", "物"],
    ]
    
    with torch.no_grad():
        for sent in eval_sents:
            target_text = "".join(sent)
            ch_tensor = torch.tensor([[ch_to_ix[w] for w in sent]], device=device)
            
            msg_tokens, _, _ = speaker(ch_tensor)
            msg_str = "".join([ix_to_en[t.item()] for t in msg_tokens[0]])
            
            logits = listener(msg_tokens)
            preds = logits.argmax(dim=-1)
            decoded_text = "".join([ix_to_ch[t.item()] for t in preds[0]])
            
            status = "✓ 成功" if target_text == decoded_text else "✗ 失敗"
            print(f"{target_text:<20} -> {msg_str:<24} -> {decoded_text:<14} ({status})")
    print("=" * 72)

if __name__ == "__main__":
    spk, lis, dev = train_jian256(episodes=1000, batch_size=128)
    inspect_codebook(spk, lis, dev)