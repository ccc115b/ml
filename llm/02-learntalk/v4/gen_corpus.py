# generate_corpus.py
import random

# ==========================================
# 語法引擎 1：簡語-256 (全詞庫無偏誤取樣)
# ==========================================
JIAN_SUBJS = ["我", "你", "他", "吾", "人", "男", "女", "子", "友", "父", "母", "王", "鳥", "獸", "魚"]
JIAN_VERBS = ["行", "來", "去", "食", "飲", "視", "聽", "感", "愛", "知", "作", "助", "問", "答", "取", "得"]
JIAN_OBJS  = ["水", "火", "土", "金", "木", "天", "地", "海", "山", "友", "父", "子", "王", "心", "物"]
JIAN_MARK1 = ["做", "將", "昔", "不"]
JIAN_MARK2 = ["於", "至", "同", "以"]

def gen_jian_sentence():
    """產生 5 字的簡語規範句子 (以空格隔開)"""
    return f"{random.choice(JIAN_SUBJS)} {random.choice(JIAN_MARK1)} {random.choice(JIAN_VERBS)} {random.choice(JIAN_MARK2)} {random.choice(JIAN_OBJS)}"

# ==========================================
# 語法引擎 2：遞迴巢狀數學算式 (Recursive Math)
# ==========================================
OPS = ["+", "-", "*", "/"]
NUMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

def gen_math_expr(depth=0, max_depth=2):
    """產生具備遞迴括號的四則運算式，如: ( ( 3 + 5 ) * 2 )"""
    if depth >= max_depth or (depth > 0 and random.random() < 0.4):
        return random.choice(NUMS)
    op = random.choice(OPS)
    left = gen_math_expr(depth + 1, max_depth)
    right = gen_math_expr(depth + 1, max_depth)
    return f"( {left} {op} {right} )"

# ==========================================
# 語法引擎 3：微型程式語言 (Mini-Code Snippets)
# ==========================================
VARS = ["x", "y", "a", "b"]
COMP_OPS = [">", "<", "=="]

def gen_mini_code():
    """產生簡化的程式語句"""
    v1 = random.choice(VARS)
    v2 = random.choice(VARS)
    val = random.choice(NUMS)
    tmpl = random.choice([
        f"let {v1} = {val} ; print {v1}",
        f"if {v1} {random.choice(COMP_OPS)} {val} : return {v1}",
        f"while {v1} > 0 : {v1} = {v1} - 1",
        f"set {v1} = {v2} + {val} ; halt",
    ])
    return tmpl

# ==========================================
# 主流程：產生 2000 行語料到 corpus.txt
# ==========================================
def build_corpus(filename="corpus.txt", total_lines=2000):
    lines = []
    print("--- 正在建構通用語料庫 ---")
    for _ in range(total_lines):
        r = random.random()
        if r < 0.5:
            lines.append(gen_jian_sentence()) # 50% 簡語
        elif r < 0.8:
            lines.append(gen_math_expr())     # 30% 遞迴數學算式
        else:
            lines.append(gen_mini_code())     # 20% 微型程式語句

    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            
    print(f"✓ 成功寫入 {filename} (共 {len(lines)} 筆多樣化語法句子)！\n")
    print("語料範例節錄：")
    for s in lines[:6]:
        print(" ", s)
    print()

if __name__ == "__main__":
    build_corpus("corpus.txt")