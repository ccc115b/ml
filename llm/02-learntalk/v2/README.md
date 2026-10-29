# ccc: 請用數學寫下這個網路的 loss


在這個「中文簡語 ↔ 雙字母通訊」的強化學習架構中，系統由兩個具有不同目標的神經網路組成：**說話者（Speaker，參數記為 $\theta$）** 與 **聆聽者（Listener，參數記為 $\phi$）**。

其損失函數（Loss Function）是由 **策略梯度（REINFORCE）** 與 **交叉熵（Cross-Entropy）** 結合的雙層優化目標。

---

### 一、 符號與變數定義

1. **真實世界中文句子**：
   $$X = (x_1, x_2, x_3, x_4, x_5), \quad x_s \in \mathcal{V}_{\text{ch}}$$
   其中 $s \in \{1, 2, 3, 4, 5\}$ 代表句子的 5 個槽位（主語、做、動作、於、賓語），$\mathcal{V}_{\text{ch}}$ 為中文詞表（大小為 12）。

2. **說話者（Speaker）生成的單詞**：
   對於每個槽位 $s$，說話者策略 $\pi_\theta$ 自回歸生成兩個英文字母 $a_s = (c_{s,1}, c_{s,2})$，其中 $c_{s,1}, c_{s,2} \in \{a, b, \dots, z\}$：
   $$P_\theta(a_s \mid X) = \pi_\theta(c_{s,1} \mid X, s) \cdot \pi_\theta(c_{s,2} \mid X, s, c_{s,1})$$
   其對數機率（Log-Likelihood）為：
   $$\log P_\theta(a_s \mid X) = \log \pi_\theta(c_{s,1} \mid X, s) + \log \pi_\theta(c_{s,2} \mid X, s, c_{s,1})$$

3. **傳輸的英文字串訊息**：
   $$M = [a_1, \text{SPACE}, a_2, \text{SPACE}, a_3, \text{SPACE}, a_4, \text{SPACE}, a_5]$$

4. **聆聽者（Listener）的預測**：
   聆聽者讀取 $M$，輸出每個槽位屬於各中文字的條件機率分佈 $\hat{p}_\phi(\cdot \mid M, s)$：
   $$\hat{p}_\phi(w \mid M, s) = \text{Softmax}(z_{s, w})$$
   其最終決策取最大機率者（Argmax）：
   $$\hat{x}_s = \arg\max_{w \in \mathcal{V}_{\text{ch}}} \hat{p}_\phi(w \mid M, s)$$

---

### 二、 聆聽者損失函數：$\mathcal{L}_{\text{lis}}(\phi)$

聆聽者是一個監督式分類器，負責將接收到的英文字母 $M$ 還原為正確的真實中文字 $x_s$。它最小化 **多標籤交叉熵損失（Multi-Slot Cross-Entropy Loss）**：

$$\mathcal{L}_{\text{lis}}(\phi) = - \frac{1}{B} \sum_{i=1}^B \left( \frac{1}{5} \sum_{s=1}^5 \log \hat{p}_\phi(x_{i, s} \mid M_i, s) \right)$$

* $B$ 為 Batch Size。
* 物理意義：直接給予聆聽者穩定的梯度，讓它以最快速度「學會將收到的一組字母映射到正確的中文字」。

---

### 三、 獎勵信號與優勢函數（Credit Assignment）

每個槽位 $s$ 獨立評估是否溝通成功（0/1 獎勵）：
$$R_{i, s} = \mathbb{I}(\hat{x}_{i, s} = x_{i, s}) = \begin{cases} 1.0, & \text{若猜對中文字} \\ 0.0, & \text{若猜錯} \end{cases}$$

為了降低政策梯度估計的高方差（Variance），定義每個槽位的**優勢函數（Advantage）**：
$$A_{i, s} = R_{i, s} - b_s$$
其中 $b_s$ 為該槽位過去的滑動平均基準線（Moving Average Baseline）：
$$b_s \leftarrow 0.95 \, b_s + 0.05 \left( \frac{1}{B} \sum_{i=1}^B R_{i, s} \right)$$

---

### 四、 說話者損失函數：$\mathcal{L}_{\text{spk}}(\theta)$

說話者無法直接對離散的字母求導，必須透過 **REINFORCE 政策梯度法** 優化，同時加入 **熵正則項（Entropy Regularization）** 防止過早收斂：

$$\mathcal{L}_{\text{spk}}(\theta) = - \frac{1}{B} \sum_{i=1}^B \left( \frac{1}{5} \sum_{s=1}^5 \Big[ A_{i, s} \cdot \log P_\theta(a_{i, s} \mid X_i) + \beta \cdot \mathcal{H}\big(\pi_\theta(\cdot \mid X_i, s)\big) \Big] \right)$$

其中：
1. **策略梯度項（Policy Gradient）**：
   $$- A_{i, s} \cdot \log P_\theta(a_{i, s} \mid X_i)$$
   若某個單詞使得聆聽者猜對了（$A_{i,s} > 0$），則加大生成這兩個字母的機率；反之則抑制。
2. **探索熵（Entropy）**：
   $$\mathcal{H}(\pi) = - \sum_{k=1}^{26} \pi(k) \log \pi(k)$$
   $\mathcal{H}\big(\pi_\theta(\cdot \mid X_i, s)\big) = \mathcal{H}(c_{s,1}) + \mathcal{H}(c_{s,2})$，$\beta = 0.01$ 為超參數，避免模型在初期只偏愛某幾個字母。

---

### 五、 數學本質總結

整個系統的梯度更新架構為：

$$\min_{\phi} \mathcal{L}_{\text{lis}}(\phi) \quad \text{與} \quad \min_{\theta} \mathcal{L}_{\text{spk}}(\theta)$$

| 元件 | 演算法 | 目標 | 解決的數學難題 |
| :--- | :--- | :--- | :--- |
| **Listener ($\phi$)** | 交叉熵分類 (Cross-Entropy) | 建立 $M \to X$ 的解碼器 | 提供確定且平滑的梯度下降方向 |
| **Speaker ($\theta$)** | 槽位級 REINFORCE | 建立 $X \to M$ 的編碼器 | 穿透不可微的離散英文字元通道 |
| **槽位解耦 ($s=1\dots5$)** | 局部信用分配 (Local Credit) | 將 $26^{10}$ 全局盲猜分解為 $5 \times 26^2$ | **徹底粉碎維度災難，使成功率在數百回合收斂至 100%** |

