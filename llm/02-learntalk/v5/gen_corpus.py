# gen_corpus.py  (LearnTalk v5)
# ==============================================================
# v5 的關鍵改動：語料「沒有語法」，只有這個世界的基本概念詞彙。
#   - 詞庫 (lexicon.txt)：256 個基本概念詞（人 / 物 / 動作 / 性質 / 數量 ...）
#   - 語料 (corpus.txt)：每行 = 從詞庫隨機抽出的 1~6 個詞
#     * 詞與詞之間沒有任何語法關係（沒有主詞/動詞/賓語的模板）
#     * 也沒有任何標記，只是「一袋」世界概念
#   句子的「結構」從哪來？不在語料裡，而留給兩位 AI 自己去發明。
# ==============================================================
import random

# ==========================================
# 詞庫：這個世界的基本概念（256 個獨立中文字）
# ==========================================
LEXICON_GROUPS = [
    # 1. 語法結構與邏輯 (16)
    "做 於 之 以 至 同 不 非 麼 昔 今 將 若 故 且 或",
    # 2. 代詞與指稱 (16)
    "我 吾 你 汝 他 此 彼 某 何 誰 焉 幾 自 互 餘 皆",
    # 3. 時間與空間方位 (32)
    "時 日 月 年 晨 午 夕 夜 刻 始 中 終 宇 地 上 下 前 後 左 右 內 外 東 西 南 北 邊 間 頂 底 近 遠",
    # 4. 自然、宇宙與物質 (32)
    "天 星 風 雲 雨 雪 雷 電 光 闇 水 火 土 石 金 木 氣 海 江 山 原 草 樹 花 葉 根 果 種 冰 物 泉 洞",
    # 5. 生命、身體與人倫 (32)
    "生 死 人 男 女 子 父 母 友 敵 王 眾 獸 鳥 魚 蟲 身 首 面 目 耳 口 鼻 舌 齒 手 足 心 血 骨 皮 毛",
    # 6. 感知、認知與情感 (24)
    "視 聽 嗅 味 觸 感 知 識 思 疑 悟 夢 愛 恨 喜 怒 哀 懼 樂 痛 願 意 信 驚",
    # 7. 動作、行為與互動 (48)
    "行 來 去 止 立 坐 臥 走 飛 游 升 降 食 飲 眠 醒 呼 吸 居 沐 作 取 置 持 移 開 閉 破 修 變 連 分 言 語 告 問 答 教 學 助 爭 和 買 賣 換 贈 尋 得",
    # 7. 性質、度量與評價 (40)
    "大 小 長 短 高 低 深 淺 重 輕 廣 狹 厚 薄 多 少 滿 空 全 半 溫 寒 熱 乾 濕 硬 軟 快 慢 明 暗 好 壞 美 醜 真 偽 易 難 貴",
    # 9. 數字與序數 (16)
    "零 一 二 三 四 五 六 七 八 九 十 百 千 萬 初 極",
]

LEXICON = []
for _g in LEXICON_GROUPS:
    LEXICON.extend(_g.split())
LEXICON = sorted(set(LEXICON), key=LEXICON.index)  # 保持分組順序、去重
assert len(LEXICON) == 256, f"詞庫長度必須是 256，實際 {len(LEXICON)}"


# ==========================================
# 隨機詞彙語料：純抽籤，零語法
# ==========================================
def gen_random_line(rng: random.Random, weights, cum=None, max_words: int = 6) -> str:
    """隨機抓一袋詞：沒有主詞/動詞/賓語的模板，純粹是概念堆積"""
    k = rng.randint(1, max_words)
    picked = set()
    guard = 0
    while len(picked) < k and guard < 50:
        guard += 1
        if cum is None:
            w = rng.choice(LEXICON)
        else:                       # 依 Zipf 詞頻抽籤（自然語言的典型長尾分布）
            r = rng.random()
            lo, hi = 0, len(cum) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            w = weights[lo]
        picked.add(w)
    return " ".join(picked)


def build_corpus(filename="corpus.txt", lexicon_file="lexicon.txt",
                 total_lines=8000, max_words=6, seed=20250, zipf=0.7):
    rng = random.Random(seed)

    # 詞庫落地成獨立檔案：learntalk.py 以此為唯一詞彙來源
    with open(lexicon_file, "w", encoding="utf-8") as f:
        for w in LEXICON:
            f.write(w + "\n")

    # 詞頻排序（固定隨機置換，與語意分組無關）：前排常用、後排冷僻
    ranks = list(range(len(LEXICON)))
    rng.shuffle(ranks)
    ordered = [LEXICON[r] for r in ranks]
    w = [1.0 / ((i + 1) ** zipf) for i in range(len(ordered))]
    tot = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / tot
        cum.append(acc)

    lines = [gen_random_line(rng, ordered, cum, max_words) for _ in range(total_lines)]

    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    from collections import Counter
    cnt = Counter(w for ln in lines for w in ln.split())
    top = cnt.most_common(8)

    print("--- LearnTalk v5：無語法語料庫建構完成 ---")
    print(f"詞庫規模: {len(LEXICON)} 個基本概念詞 -> {lexicon_file}")
    print(f"語料行數: {len(lines)} (每行 1~{max_words} 個隨機詞，零語法) -> {filename}")
    print(f"詞頻分布: Zipf(alpha={zipf})，最常用 8 個詞 -> "
          + " ".join(f"{k}({v})" for k, v in top))
    print("\n語料節錄（注意：毫無句法結構，只是概念袋）：")
    for s in lines[:8]:
        print("  ", s)
    print("\n→ 句子怎麼排列、角色怎麼標記、訊息怎麼收尾，"
          "全部留給兩位 AI 自己創造。\n")


if __name__ == "__main__":
    build_corpus("corpus.txt")
