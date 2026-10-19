import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# 準備資料集
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer 設置
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# 模型超參數（與原版完全相同）
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# 定義 PyTorch 版的 RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (ms + self.eps).rsqrt()

# 定義 GPT 模型架構
class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn_wq': nn.Linear(n_embd, n_embd, bias=False),
                'attn_wk': nn.Linear(n_embd, n_embd, bias=False),
                'attn_wv': nn.Linear(n_embd, n_embd, bias=False),
                'attn_wo': nn.Linear(n_embd, n_embd, bias=False),
                'mlp_fc1': nn.Linear(n_embd, 4 * n_embd, bias=False),
                'mlp_fc2': nn.Linear(4 * n_embd, n_embd, bias=False),
                'rmsnorm1': RMSNorm(n_embd),
                'rmsnorm2': RMSNorm(n_embd),
            }) for _ in range(n_layer)
        ])
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # 依照原版進行權重初始化 (std=0.08)
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, token_id, pos_id, keys, values):
        # 轉換成 Tensor (純量或維度為 1 的 Tensor)
        tok_id_t = torch.tensor([token_id], dtype=torch.long)
        pos_id_t = torch.tensor([pos_id], dtype=torch.long)
        
        tok_emb = self.wte(tok_id_t).squeeze(0) # (n_embd,)
        pos_emb = self.wpe(pos_id_t).squeeze(0) # (n_embd,)
        
        x = tok_emb + pos_emb
        x = x * ((x.pow(2).mean() + 1e-5) ** -0.5) # 初始 RMSNorm

        for li, layer in enumerate(self.layers):
            # 1) Multi-head Attention block
            x_residual = x
            x_norm = layer['rmsnorm1'](x)
            
            q = layer['attn_wq'](x_norm)
            k = layer['attn_wk'](x_norm)
            v = layer['attn_wv'](x_norm)
            
            keys[li].append(k)
            values[li].append(v)
            
            x_attn_heads = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs+head_dim]
                k_h = torch.stack([ki[hs:hs+head_dim] for ki in keys[li]]) # (T, head_dim)
                v_h = torch.stack([vi[hs:hs+head_dim] for vi in values[li]]) # (T, head_dim)
                
                # 注意力權重計算
                attn_logits = (k_h @ q_h) / (head_dim ** 0.5) # (T,)
                attn_weights = F.softmax(attn_logits, dim=-1) # (T,)
                head_out = attn_weights @ v_h # (head_dim,)
                x_attn_heads.append(head_out)
                
            x_attn = torch.cat(x_attn_heads, dim=-1)
            x = layer['attn_wo'](x_attn) + x_residual
            
            # 2) MLP block
            x_residual = x
            x_norm = layer['rmsnorm2'](x)
            x = layer['mlp_fc1'](x_norm)
            x = F.relu(x)
            x = layer['mlp_fc2'](x)
            x = x + x_residual

        logits = self.lm_head(x)
        return logits

model = MicroGPT()
print(f"num params: {sum(p.numel() for p in model.parameters())}")

# 使用 PyTorch 的 Adam 優化器
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)

# 訓練迴圈
num_steps = 1000
for step in range(num_steps):
    # 動態調整學習率（線性衰減）
    lr_t = learning_rate * (1 - step / num_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t

    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    optimizer.zero_grad()
    
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = model(token_id, pos_id, keys, values)
        loss_t = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], dtype=torch.long))
        losses.append(loss_t)
        
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# 推論生成
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
model.eval()
with torch.no_grad():
    for sample_idx in range(20):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = model(token_id, pos_id, keys, values)
            probs = F.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")