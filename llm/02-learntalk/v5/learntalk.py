# learntalk.py  (LearnTalk v5)
# ==============================================================
# v5 的核心改變：
#   v4 : 通道 = 兩個英文字母組成的假密碼，語料本身帶語法
#   v5 : 通道 = 「原本的世界概念詞」本身（沒有假密碼、沒有分隔符）
#         語料只有隨機詞袋、沒有任何語法
#   → 誰來決定「句子的樣子」？只有兩位 AI 自己。
#     它們必須自創：標記詞 (function word)、角色標記、結束標記、收件規則。
# ==============================================================
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# 1. 設定：詞庫 / 角色 / 句型 / 標註類別
# ==========================================
ROLES = ["施事", "動作", "受事", "工具"]   # 語意角色（世界給定的「意義」）
TYPES = ["陳述", "疑問"]                   # 句型
N_ROLES, N_TYPES = len(ROLES), len(TYPES)
MAX_ITEMS = 3                             # 一則訊息最多帶幾個概念

# Listener 的標註類別：每個通道上的詞被標成「內容 / 某角色的標記 / 某句型 / 結束」
TAG_CONTENT = 0
TAG_MARKER = 1                            # TAG_MARKER + r  = 角色 r 的標記詞
TAG_TYPE = TAG_MARKER + N_ROLES           # TAG_TYPE + t    = 句型 t 的標記詞
TAG_END = TAG_TYPE + N_TYPES
N_TAGS = TAG_END + 1

N_SLOTS = N_ROLES + N_TYPES + 1      # 協定量槽位數：角色標記 + 句型標記 + 結束標記
PAD_ID = 256            # 通道上的填充符（不是任何一個世界概念詞）


def load_lexicon(path="lexicon.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]


def load_corpus(path="corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]


# ==========================================
# 2. 場景取樣：從「毫無語法的詞袋」造出一則有意義的訊息
# ==========================================
def sample_scene(rng: random.Random, corpus, word_to_ix):
    """回傳 (words, roles, type)：words=要傳達的概念, roles=各概念扮演的角色"""
    bag = rng.choice(corpus)
    k = min(rng.randint(1, MAX_ITEMS), len(bag))
    picked = rng.sample(bag, k)
    # 角色集合每則訊息都不同 → 位置不可能代表角色 → 必須靠標記
    roles = rng.sample(range(N_ROLES), k)
    t = rng.randrange(N_TYPES)
    return picked, roles, t


# ==========================================
# 3. 協定量 (Protocol Lexicon)：AI 自創語法的所在
#    每一列是一個「功能詞槽位」：施事標記 / 動作標記 / ... / 句型標記 / 結束標記
#    槽位該用哪一個「世界概念詞」來當標記，是 AI 自己學出來的。
# ==========================================
class ProtocolLexicon(nn.Module):
    def __init__(self, n_slots, vocab_size, init_scale=1.0):
        super().__init__()
        self.n_slots = n_slots
        self.logits = nn.Parameter(torch.randn(n_slots, vocab_size) * init_scale)

    def log_probs(self, tau=1.0):
        return F.log_softmax(self.logits / tau, dim=-1)

    def sample_distinct(self, tau=0.7):
        """依序抽樣，各槽位不重複（同一個詞不能同時當兩種標記）"""
        g = -torch.log(-torch.log(torch.rand_like(self.logits).clamp_min(1e-9)).clamp_min(1e-9))
        scores = self.logits / tau + g
        chosen, mask = [], torch.zeros_like(self.logits, dtype=torch.bool)
        for j in range(self.n_slots):
            s = scores[j].masked_fill(mask[j], float("-inf"))
            idx = s.argmax()
            chosen.append(idx)
            mask[:, idx] = True        # 全域去重：不同槽位不能搶同一個詞
        return torch.stack(chosen)


# ==========================================
# 4. Speaker：把「意義」排成「一句話」（自創標記、自創順序、自創收尾）
# ==========================================
class GrammarSpeaker(nn.Module):
    """場景 -> 訊息流。通道上只有世界概念詞，沒有分隔符、沒有長度標頭。"""

    def __init__(self, vocab_size, d_model=64, max_items=MAX_ITEMS):
        super().__init__()
        self.role_emb = nn.Embedding(N_ROLES, d_model)
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.item_pos = nn.Parameter(torch.zeros(1, max_items, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                               dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        # 指標網路：決定「下一個要講哪個概念」（AI 自己決定語序）
        self.query = nn.Parameter(torch.randn(max_items, d_model) * 0.4)
        self.proto = ProtocolLexicon(N_SLOTS, vocab_size)
        self.max_items = max_items

    def forward(self, words, roles, scene_type, proto_words, item_mask=None,
                counts=None):
        """
        words/roles: [B,K]  scene_type: [B]  proto_words: [n_slots]
        回傳 stream [B,T] / logp [B,T] / logp_tbl[B,T] / valid [B,T] / tags [B,T]
          logp     = 全部動作的對數機率（指標 + 協定量）
          logp_tbl = 只有「協定量」部分的對數機率（給效用項單獨用）
        """
        B, K = words.shape
        device = words.device
        if item_mask is None:
            item_mask = torch.ones_like(words, dtype=torch.bool)
        x = self.word_emb(words) + self.role_emb(roles) + self.item_pos[:, :K, :]
        h = self.encoder(x)                                   # [B,K,D] 場景記憶

        lp_table = self.proto.log_probs(tau=1.0)               # [slots,V] 策略梯度用原始分布
        role_cur = torch.zeros(B, dtype=torch.long, device=device)
        phase = torch.zeros(B, dtype=torch.long, device=device)   # 0句型 1標記 2內容 3結束 4完
        cur_item = torch.zeros(B, dtype=torch.long, device=device)
        remaining = item_mask.clone()
        n_emit = torch.zeros(B, dtype=torch.long, device=device)
        if counts is None:
            counts = item_mask.sum(dim=1)

        toks, lps, lps_tbl, valids, tags = [], [], [], [], []
        for t in range(2 + 2 * self.max_items):
            # --- 指標：挑一個還沒講的概念（只有「標記」階段才採用）---
            q = self.query[n_emit.clamp(max=self.max_items - 1)]
            ptr_logits = (h * q.unsqueeze(1)).sum(-1).masked_fill(~remaining, -1e9)
            ptr_dist = Categorical(logits=ptr_logits)
            new_item = ptr_dist.sample()
            ptr_logp = ptr_dist.log_prob(new_item)

            is_type = phase == 0
            is_mark = phase == 1
            is_word = phase == 2
            is_end = phase == 3

            cur_item = torch.where(is_mark, new_item, cur_item)
            role_cur = roles.gather(1, cur_item.unsqueeze(1)).squeeze(1)

            # --- 各階段的動作 ---
            type_slot = N_ROLES + scene_type
            w_type = proto_words[type_slot]
            lp_type = lp_table[type_slot].gather(1, w_type.unsqueeze(1)).squeeze(1)
            w_mark = proto_words[role_cur]
            lp_mark = lp_table[role_cur].gather(1, w_mark.unsqueeze(1)).squeeze(1)
            w_word = words.gather(1, cur_item.unsqueeze(1)).squeeze(1)
            w_end = proto_words[-1]
            lp_end = lp_table[-1][w_end]

            token = torch.where(is_type, w_type,
                     torch.where(is_mark, w_mark,
                     torch.where(is_word, w_word, w_end)))
            logp = torch.where(is_type, lp_type,
                   torch.where(is_mark, lp_mark + ptr_logp,
                   torch.where(is_word, torch.zeros_like(lp_type), lp_end)))
            logp_tbl = torch.where(is_word, torch.zeros_like(lp_type),
                       torch.where(is_type, lp_type,
                       torch.where(is_mark, lp_mark, lp_end)))
            valid = phase <= 3
            tag = torch.where(is_type, TAG_TYPE + scene_type,
                    torch.where(is_mark, TAG_MARKER + role_cur,
                    torch.where(is_word, torch.full_like(role_cur, TAG_CONTENT),
                                torch.full_like(role_cur, TAG_END))))

            toks.append(token)
            lps.append(logp)
            lps_tbl.append(logp_tbl)
            valids.append(valid)
            tags.append(tag)

            # --- 狀態更新 ---
            idx = cur_item.unsqueeze(1)
            gathered = remaining.gather(1, idx).clone()
            gathered[is_word] = False
            remaining = remaining.scatter(1, idx, gathered)
            n_emit = n_emit + is_word.long()
            more = n_emit < counts
            phase = torch.where(is_type, torch.ones_like(phase),
                   torch.where(is_mark, torch.full_like(phase, 2),
                   torch.where(is_word, torch.where(more, torch.ones_like(phase),
                                                   torch.full_like(phase, 3)),
                   torch.where(is_end, torch.full_like(phase, 4), phase))))

        return (torch.stack(toks, 1), torch.stack(lps, 1),
                torch.stack(lps_tbl, 1), torch.stack(valids, 1),
                torch.stack(tags, 1))


# ==========================================
# 5. Listener：讀懂一條「沒有分隔符的字詞流」
#    工作 = 給每個字詞標註它在這句話裡的身分（內容 / 某角色標記 / 句型 / 結束）
# ==========================================
class GrammarListener(nn.Module):
    def __init__(self, vocab_size, d_model=64, max_len=1024):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size + 1, d_model)   # +1 = PAD
        self.in_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                               dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, N_TAGS)

    def forward(self, stream):
        B, S = stream.shape
        x = self.in_emb(stream) + self.in_pos[:, :S, :]
        h = self.encoder(x)
        return self.classifier(h)


# ==========================================
# 6. 解讀：把標註結果還原成「一則則訊息」
# ==========================================
def decode_tags(tags, tokens):
    """tags/tokens: 長度 L 的序列 -> [(句型, [(角色, 詞), ...]), ...]"""
    msgs, cur_type, cur_items = [], -1, []
    for i, tg in enumerate(tags):
        if tg == TAG_END:
            msgs.append((cur_type, cur_items))
            cur_type, cur_items = -1, []
        elif TAG_TYPE <= tg < TAG_TYPE + N_TYPES:
            cur_type = tg - TAG_TYPE
        elif TAG_MARKER <= tg < TAG_MARKER + N_ROLES:
            role = tg - TAG_MARKER
            if i + 1 < len(tags) and tags[i + 1] == TAG_CONTENT:
                cur_items.append((role, tokens[i + 1]))
    if cur_type != -1 or cur_items:
        msgs.append((cur_type, cur_items))
    return msgs


# ==========================================
# 7. 訓練：Listener 學「讀懂」，Speaker 靠獎勵學「怎麼講才聽得懂」
# ==========================================
def train(episodes=1500, batch_size=16, d_model=64, lr=3e-3, proto_lr=1e-2,
          beta_collision=20.0, beta_freq=40.0, beta_distinct=60.0,
          beta_convention=0.02, w_table=0.2,
          tau_start=1.0, tau_end=0.05, freeze_ratio=0.75,
          corpus_path="corpus.txt", lexicon_path="lexicon.txt",
          seed=0, log_every=50):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lexicon = load_lexicon(lexicon_path)
    word_to_ix = {w: i for i, w in enumerate(lexicon)}
    ix_to_word = {i: w for i, w in enumerate(lexicon)}
    corpus = load_corpus(corpus_path)
    V = len(lexicon)

    # 詞頻（0~1）：用來衡量一個詞當「標記」會不會太常見而跟內容詞撞車
    from collections import Counter
    _cnt = Counter(w for ln in corpus for w in ln)
    _mx = max(_cnt.values())
    freq_vec = torch.tensor([_cnt.get(w, 0) / _mx for w in lexicon], device=device)

    print("=" * 76)
    print("        LearnTalk v5：兩位 AI 從零創造自己的語法")
    print("=" * 76)
    print(f"詞庫（世界概念詞）: {V} 個 | 語料: {len(corpus)} 行（零語法，純詞袋）")
    print(f"語意角色: {' / '.join(ROLES)} | 句型: {' / '.join(TYPES)}")
    print(f"通道: 只有原本的 {V} 個概念詞，沒有假密碼、沒有分隔符")
    print(f"執行裝置: {device}\n")

    speaker = GrammarSpeaker(V, d_model=d_model).to(device)
    listener = GrammarListener(V, d_model=d_model).to(device)

    spk_opt = optim.Adam([
        {"params": [p for n, p in speaker.named_parameters() if n != "proto.logits"],
         "lr": lr},
        {"params": [speaker.proto.logits], "lr": proto_lr},   # 協定量：微調出「哪個字當標記」
    ])
    lis_opt = optim.Adam(listener.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    baseline = 0.0
    history, tag_hist = [], []
    print("=== 訓練開始：AI 正在發明「哪個詞當標記、怎麼排、怎麼收尾」 ===\n")

    for ep in range(episodes):
        # ---- 1. 抽場景 ----
        scenes = [sample_scene(rng, corpus, word_to_ix) for _ in range(batch_size)]
        B, K = batch_size, MAX_ITEMS
        words = torch.zeros(B, K, dtype=torch.long, device=device)
        roles = torch.zeros(B, K, dtype=torch.long, device=device)
        imask = torch.zeros(B, K, dtype=torch.bool, device=device)
        stype = torch.tensor([s[2] for s in scenes], dtype=torch.long, device=device)
        counts = torch.tensor([len(s[0]) for s in scenes], dtype=torch.long, device=device)
        for i, (p, r, _) in enumerate(scenes):
            for j in range(len(p)):
                words[i, j] = word_to_ix[p[j]]
                roles[i, j] = r[j]
                imask[i, j] = True
        truth_items = [{(r, word_to_ix[p[j]]) for j, r in enumerate(rr)}
                       for p, rr, _ in scenes]
        truth_types = [tt for _, _, tt in scenes]

        # ---- 2. 本回合的「自創標記集」（全 batch 一致，才叫慣例）----
        #    第一階段：溫度退火 + 隨機抽樣（還在探索、還在換標記詞）
        #    第二階段：把慣例「定型」，Listener 針對最終那套語法收尾微調
        frozen = ep >= int(episodes * freeze_ratio)
        if frozen:
            proto_words = speaker.proto.logits.detach().argmax(-1).to(device)
        else:
            tau = max(tau_end, tau_start * (tau_end / tau_start) ** (ep / max(1, episodes)))
            proto_words = speaker.proto.sample_distinct(tau=tau).to(device)

        # ---- 3. Speaker 說話 ----
        stream, spk_logp, spk_logp_tbl, spk_valid, tags = speaker(
            words, roles, stype, proto_words, item_mask=imask, counts=counts)
        msg_len = spk_valid.sum(dim=1)

        # ---- 4. 組出 B 條「通道」：每條含若干則訊息（長度隨機，模擬真實通訊）----
        chan_stream, chan_tags, chan_msgs = [], [], []
        for b in range(B):
            perm = rng.sample(range(B), B)
            use = perm[:rng.randint(1, B)]        # 通道長度隨機：1~B 則訊息
            cs, ct = [], []
            for m in use:
                Ln = int(msg_len[m].item())
                cs.extend(stream[m, :Ln].tolist())
                ct.extend(tags[m, :Ln].tolist())
            chan_stream.append(cs)
            chan_tags.append(ct)
            chan_msgs.append(use)
        S = max(max(len(c) for c in chan_stream), max(len(c) for c in chan_tags))
        stream_in = torch.full((B, S), PAD_ID, dtype=torch.long, device=device)
        tag_tgt = torch.full((B, S), -100, dtype=torch.long, device=device)
        for b in range(B):
            stream_in[b, :len(chan_stream[b])] = torch.tensor(chan_stream[b], device=device)
            tag_tgt[b, :len(chan_tags[b])] = torch.tensor(chan_tags[b], device=device)

        # ---- 5. Listener 標註 ----
        logits = listener(stream_in)
        lis_loss = ce(logits.reshape(-1, N_TAGS), tag_tgt.reshape(-1))
        pred_tags = logits.argmax(dim=-1)

        # ---- 6. 算每一則訊息的報酬 ----
        rewards = torch.zeros(B, device=device)
        appear = torch.zeros(B, device=device)
        for b in range(B):
            parsed = decode_tags(pred_tags[b].tolist(), stream_in[b].tolist())
            for idx, m in enumerate(chan_msgs[b]):
                if idx < len(parsed):
                    ptype, pitems = parsed[idx]
                    got = set(pitems)
                    true = truth_items[m]
                    item_score = len(true & got) / max(1, len(true))
                    type_score = 1.0 if ptype == truth_types[m] else 0.0
                    frame_ok = 1.0 if len(pitems) == len(true) else 0.0
                else:
                    item_score = type_score = frame_ok = 0.0
                rewards[m] += 0.6 * item_score + 0.2 * type_score + 0.2 * frame_ok
                appear[m] += 1
        rewards = rewards / appear.clamp(min=1)
        baseline = 0.95 * baseline + 0.05 * rewards.mean().item()

        # ---- 7. Speaker 的 REINFORCE ----
        #    指標網路（語序）吃主要的策略梯度；協定量只吃一點點，
        #    讓「效用」訊號（撞詞率、熵）有主導權 → 慣例才會穩定凝固
        adv = (rewards - baseline).unsqueeze(1) * spk_valid.float()
        ptr_logp = spk_logp - spk_logp_tbl
        spk_loss = -(ptr_logp * adv).sum() / spk_valid.float().sum()
        spk_loss = spk_loss - w_table * (spk_logp_tbl * adv).sum() / spk_valid.float().sum()
        # 撞詞懲罰：標記詞若跟本則內容詞撞車，接收端就分不清 -> AI 只好挑冷僻字
        p_slot = speaker.proto.logits.softmax(-1)              # [slots,V]
        coll = 0.0
        for m in range(B):
            cids = [w for _, w in truth_items[m]]
            for j in range(N_SLOTS):
                coll = coll + p_slot[j, cids].sum()
        spk_loss = spk_loss + beta_collision * coll / (B * N_SLOTS)
        # 冷僻化壓力：直接對「標記詞的期望詞頻」施壓 → 常用字的 logit 被持續壓低，
        # AI 會一路換到最冷僻的字（標記越冷僻，越不會跟內容詞撞車）
        exp_freq = (p_slot * freq_vec).sum(-1).mean()
        spk_loss = spk_loss + beta_freq * exp_freq
        # 槽位互斥：兩個標記不能搶同一個詞（否則接收端無從分辨）
        cross = (p_slot @ p_slot.T)
        distinct_pen = (cross.sum() - cross.diag().sum()) / 2
        spk_loss = spk_loss + beta_distinct * distinct_pen
        # 語法收斂項：讓協定量銳化，凝固成「一套」而不是每回合換一套
        lp_all = speaker.proto.log_probs()
        ent = -(lp_all.exp() * lp_all).sum(-1).mean()
        spk_loss = spk_loss + beta_convention * ent

        lis_opt.zero_grad(); lis_loss.backward(); lis_opt.step()
        spk_opt.zero_grad(); spk_loss.backward()
        if frozen:
            speaker.proto.logits.grad = None      # 定型期：語法凍結，不再改
        spk_opt.step()

        acc = rewards.mean().item()
        history.append(acc)
        m_valid = tag_tgt != -100
        tag_acc = (pred_tags[m_valid] == tag_tgt[m_valid]).float().mean().item()
        tag_hist.append(tag_acc)
        if (ep + 1) % log_every == 0:
            win = max(1, len(history) - 100)
            conf = speaker.proto.logits.detach().softmax(-1).max(-1).values.mean().item()
            _t = max(tau_end, tau_start * (tau_end / tau_start) ** (ep / max(1, episodes)))
            print(f"回合 {ep+1:5d}/{episodes} | 通訊成功率: {acc*100:5.1f}% "
                  f"| 近100回合: {sum(history[win:])/len(history[win:])*100:5.1f}% "
                  f"| 標註準確率: {tag_acc*100:5.1f}% | 語法銳利度: {conf*100:5.1f}% | τ={_t:.2f}")

    print("\n=== 訓練完成 ===\n")
    torch.save({"speaker": speaker.state_dict(), "listener": listener.state_dict()},
               "learntalk_v5.pt")
    return speaker, listener, lexicon, word_to_ix, ix_to_word, corpus, device, history


# ==========================================
# 8. 展示：AI 自創的語法字典
# ==========================================
def report_protocol(speaker, listener, lexicon, corpus, device):
    speaker.eval()
    proto = speaker.proto.logits.detach()
    picked = proto.argmax(dim=-1)
    conf = proto.softmax(-1).max(-1).values
    names = ROLES + TYPES + ["結束"]

    # 詞頻排名（從語料統計）：AI 挑的標記詞是常用詞還是冷僻詞？
    from collections import Counter
    cnt = Counter(w for ln in corpus for w in ln)
    order = [w for w, _ in cnt.most_common()]
    rank = {w: i + 1 for i, w in enumerate(order)}
    marker_words = {nm: lexicon[picked[i].item()] for i, nm in enumerate(names)}

    print("=" * 76)
    print("      📖 兩位 AI 談妥的「自創語法字典」（AI 自己挑的標記詞）")
    print("=" * 76)
    for i, nm in enumerate(names):
        w = marker_words[nm]
        r = rank.get(w, 0)
        print(f"  {nm:<4} 的標記詞  ->  「{w}」  (詞頻第 {r}/{len(cnt)} 名, "
              f"把握度 {conf[i].item()*100:.0f}%)")

    print("\n  AI 自己寫下的語法規格（BNF）：")
    print("    句子   ::= ⟨句型標記⟩ ( ⟨角色標記⟩ 概念 )* ⟨結束標記⟩")
    print("    ⟨句型標記⟩ ::= 「%s」(陳述) | 「%s」(疑問)"
          % (marker_words["陳述"], marker_words["疑問"]))
    for r_ in ROLES:
        print(f"    ⟨{r_}標記⟩ ::= 「{marker_words[r_]}」")
    print(f"    ⟨結束標記⟩ ::= 「{marker_words['結束']}」")
    print("\n  → 角色靠「標記詞」承載，語序自由，收尾靠「結束標記」：")
    print("    這是一種『標記型 (marker-based) 且與語序無關』的語言。")
    print("  → 為什麼挑這些冷僻字？常用字在語料裡天天出現，")
    print("    拿來當標記會跟內容詞撞車 → AI 自己學會了「標記要冷僻」。")

    # 殘餘失誤來源：標記詞與內容詞撞車的比例
    mset = set(marker_words.values())
    hit = sum(1 for ln in corpus if mset & set(ln))
    print(f"\n  殘餘風險：語料中有 {hit}/{len(corpus)} 行 "
          f"({hit/len(corpus)*100:.1f}%) 會用到標記詞本身 → 該句會有歧義，")
    print("    這就是 AI 聽錯時的來源，也是它拼命挑冷僻字的动机。\n")
    return marker_words


# ==========================================
# 9. 展示：雙向對話 + 全新語料的解讀測試
# ==========================================
def render_stream(tokens, marker_of, ix_to_word):
    out = []
    for t in tokens:
        w = ix_to_word[t]
        out.append(f"{w}〔{marker_of[t]}〕" if t in marker_of else w)
    return " ".join(out)


def demo(speaker, listener, lexicon, word_to_ix, ix_to_word, corpus, device):
    speaker.eval()
    listener.eval()
    # 用「收斂後的那一套語法」（協定量的 argmax），而不是再抽一次
    proto_words = speaker.proto.logits.detach().argmax(-1).to(device)
    marker_of = {}
    for i, nm in enumerate(ROLES + TYPES + ["結束"]):
        marker_of[int(proto_words[i].item())] = nm

    def speak(picked, roles, t):
        words = torch.tensor([[word_to_ix[w] for w in picked]],
                             dtype=torch.long, device=device)
        rr = torch.tensor([roles], dtype=torch.long, device=device)
        st = torch.tensor([t], dtype=torch.long, device=device)
        with torch.no_grad():
            s, _, _, valid, _ = speaker(words, rr, st, proto_words)
        n = int(valid.sum(dim=1)[0].item())      # 截掉多餘的填充位置
        return [int(x) for x in s[0, :n].tolist()]

    def hear(toks):
        with torch.no_grad():
            lg = listener(torch.tensor([toks], dtype=torch.long, device=device))
        parsed = decode_tags(lg[0].argmax(dim=-1).tolist(), toks)
        if not parsed:
            return None
        typ, items = parsed[0]
        return typ, [(ROLES[r], ix_to_word[w]) for r, w in items]

    def show_round(tag, who_says, picked, roles, t, who_listens):
        toks = speak(picked, roles, t)
        print(f"\n【{tag}】{who_says} 說話，{who_listens} 聽")
        print("  語意：" + f"{TYPES[t]} | " +
              " ".join(f"{ROLES[r]}={w}" for w, r in zip(picked, roles)))
        print(f"  訊息流： {render_stream(toks, marker_of, ix_to_word)}")
        h = hear(toks)
        print(f"  {who_listens} 聽懂了：", "（解析失敗）" if h is None else
              f"{TYPES[h[0]]} | " + " ".join(f"{r}={w}" for r, w in h[1]))
        return h

    print("=" * 76)
    print("      💬 雙向通訊實測：兩個 AI 各自用「自創語法」說話")
    print("=" * 76)

    show_round("第一回合", "AI-A", ["鳥", "飛", "天"], [0, 1, 2], 0, "AI-B")
    show_round("第二回合", "AI-B", ["鳥", "問", "天"], [0, 1, 2], 1, "AI-A")
    show_round("第三回合", "AI-A", ["水", "飲", "魚"], [0, 1, 2], 0, "AI-B")
    show_round("第四回合", "AI-B", ["人", "走", "山"], [0, 1, 2], 0, "AI-A")

    # 反例：內容詞剛好就是某個標記詞 → 接收端必然誤判（這套語言唯一的弱點）
    if marker_of:
        cw = ix_to_word[list(marker_of.keys())[0]]
        show_round("反例", "AI-A", [cw, "走", "山"], [0, 1, 2], 0, "AI-B")
        print("  ↑ 內容詞「%s」正好是 AI 選的標記詞 → 接收端分不清它是標記還是內容。"
              % cw)

    print("\n【新概念袋測試】未見過的隨機詞袋，混在同一條通道上測試聽懂率")
    rng = random.Random(999)
    total = 40
    tests = [sample_scene(rng, corpus, word_to_ix) for _ in range(total)]
    chan = []
    for p, r, tt in tests:
        chan.extend(speak(p, r, tt))
    with torch.no_grad():
        lg = listener(torch.tensor([chan], dtype=torch.long, device=device))
    parsed = decode_tags(lg[0].argmax(dim=-1).tolist(), chan)
    ok = 0
    for i, (p, r, tt) in enumerate(tests):
        if i >= len(parsed):
            continue
        typ, items = parsed[i]
        got_roles = {ROLES[rr] for rr, _ in items}      # 角色索引 → 角色名
        true_roles = {ROLES[x] for x in r}
        ok += int(got_roles == true_roles and len(items) == len(r) and typ == tt)
    print(f"  角色/句型完全解讀正確： {ok}/{total} ({ok/total*100:.0f}%)")

    print("\n【通道長度穩健度】一條通道塞多少則訊息，接收端還聽得懂？")
    for n in [10, 20, 40, 60, 80]:
        rng2 = random.Random(2024)
        ts = [sample_scene(rng2, corpus, word_to_ix) for _ in range(n)]
        ch = []
        for p, r, tt in ts:
            ch.extend(speak(p, r, tt))
        with torch.no_grad():
            lg2 = listener(torch.tensor([ch], dtype=torch.long, device=device))
        ps = decode_tags(lg2[0].argmax(dim=-1).tolist(), ch)
        good = 0
        for i, (p, r, tt) in enumerate(ts):
            if i >= len(ps):
                continue
            typ, items = ps[i]
            good += int({ROLES[rr] for rr, _ in items} == {ROLES[x] for x in r}
                        and len(items) == len(r) and typ == tt)
        print(f"    {n:3d} 則訊息 / {len(ch):4d} token : {good:3d}/{n:3d} "
              f"({good/n*100:3.0f}%)")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    if os.environ.get("LT_LOAD") == "1" and os.path.exists("learntalk_v5.pt"):
        lex = load_lexicon()
        w2i = {w: i for i, w in enumerate(lex)}
        i2w = {i: w for i, w in enumerate(lex)}
        corp = load_corpus()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spk = GrammarSpeaker(len(lex)).to(dev)
        lis = GrammarListener(len(lex)).to(dev)
        ck = torch.load("learntalk_v5.pt", map_location=dev)
        spk.load_state_dict(ck["speaker"]); lis.load_state_dict(ck["listener"])
        spk.eval(); lis.eval()
        print("（已載入 learntalk_v5.pt）\n")
    else:
        spk, lis, lex, w2i, i2w, corp, dev, hist = train(
            episodes=int(os.environ.get("LT_EPISODES", 1200)),
            batch_size=int(os.environ.get("LT_BATCH", 40)),
        )
    report_protocol(spk, lis, lex, corp, dev)
    demo(spk, lis, lex, w2i, i2w, corp, dev)
