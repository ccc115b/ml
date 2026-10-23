# webgym0
零依賴、純前端 RL 測試平台。環境與策略都是 ES module，瀏覽器與 Node 共用。
- `npm test`  → 對每個環境跑 200 個種子，驗證 closed-form 策略通過
- `npm start` → http://localhost:8080 視覺化播放 / 比較隨機策略

| 環境 | closed-form 解法 | 通過條件 |
|---|---|---|
| CartPole-v1 | 線性狀態回饋 u=k·s | 500 步全存活 |
| MountainCar-v0 | 順速度方向推（能量法） | ≥ -150 |
| FrozenLake-v1 (slippery) | 已知模型解 Bellman 最適方程式 | 成功率 ≥ 0.7 (理論約 0.74) |
| FrozenLake-v1-noslip | 同上（等於最短路徑） | 1.0 |
| CliffWalking-v0 | 上→右×11→下 | -13 (最適) |
| Snake-v0 (8x8) | 哈密頓迴圈（沿環前進） | 吃滿棋盤，回報 63 |
| Braitenberg-v0 (仿生趨光車) | 交叉連線感測器→輪子 + 除法正規化 | 抵達光源 (≥ -80) |

新增環境：在 `src/envs.js` 實作 `reset/step/render` 並 `register(id, factory)`，再於 `src/closedForm.js` 加一條。

## TF.js 即時訓練
- 瀏覽器：按「🧠 TF.js 訓練」（CDN 載入 tf.js），即時顯示學習曲線，訓練完按「訓練後策略」播放並與 closed-form 比較。
- `npm i && npm run test:train`：Node 內以 REINFORCE 訓練 CartPole / FrozenLake(noslip) / CliffWalking 並驗證收斂。
- 目前支援離散動作 + 向量/整數觀察的環境；Snake(物件觀察)、Braitenberg(連續動作)、MountainCar(探索稀疏) 留給 closed-form。

## 模仿學習：closed-form 當老師 (`src/imitate.js`)
補上強化學習學不好的三個環境。**老師 = closed-form 策略，學生 = TF.js 網路，學生不看環境獎勵**，只學「老師在這個狀態會做什麼」(DAgger：學生自己玩、老師逐步糾正)。
- `npm run test:imitate`：MountainCar / Braitenberg / Snake 的學生皆在 1 輪內達到老師水準。
- Snake 學生只看蛇頭位置，等於自己學出了哈密頓迴圈；Braitenberg 學生只看左右眼亮度，學出交叉連線 + 正規化。

## MuJoCo 環境 (`src/mj.js`，MuJoCo WASM)
動力學由 MuJoCo 計算；obs / reward / 終止條件仿 Gymnasium 重寫（**非逐位元相同**，故命名 `Mj*`）。載入 WASM 約 11 MB，瀏覽器先試本機 `node_modules`，再退回 CDN。

| 環境 | 解析 / 模型導向控制器 | 通過條件 |
|---|---|---|
| MjReacher-v0 | 兩連桿解析逆運動學 (餘弦定理) + PD | 平均回報 ≥ -6、最終距離 ≤ 0.02 |
| MjInvertedDoublePendulum-v0 | 平衡點線性化 → 解 Riccati 的 LQR | 每回合撐滿 1000 步 (≥ 9100) |
| MjSwimmer-v0 | 行進波 CPG（時鐘 sin/cos 的線性組合）| 平均回報 ≥ 150（Swimmer 的週期/相位差由離線掃描定出，非純推導）|

- `npm run test:mujoco`：驗證解析控制器，並讓 TF.js 學生 (DAgger) 模仿老師後也通過。
- 這三個環境為連續動作，只提供「模仿學習」，不提供 REINFORCE。
- MuJoCo 物件在 WASM heap，不會被 JS GC：環境提供 `close()`。

### 運動環境 (Gymnasium 原始模型)
模型為 Gymnasium 的 `half_cheetah.xml` / `walker2d_v5.xml` / `humanoid.xml`（未修改，MIT，見 `LICENSE-gymnasium.txt`，內嵌於 `src/mjModels.js`）。obs / reward / 終止條件依 v5 設計重寫並簡化（Humanoid 的 obs 為 qpos[2:]+qvel 共 45 維，未含 cinert 等；前進速度取根節點 x 位置差），**非逐位元相同**。

| 環境 | 老師 | 說明 |
|---|---|---|
| MjHalfCheetah-v0 | 開環正弦步態 `clip(A·sin(c+φ)+b)` | 環境提供步態時鐘 sin/cos；**A、φ、b 與週期 0.25s 是離線隨機搜尋 + 爬山得到的，並非推導**；A>1 時近似方波，為 4 Hz 高頻步態，可能利用模擬器特性，不是自然步態。軀幹不翻倒。通過條件：平均回報 ≥ 3500（學生 ≥ 3000）|
| MjWalker2d-v0 | 無 | 只提供環境 + 隨機基準（隨機策略約 20 步摔倒）。沒有簡潔的 closed-form 平衡/步行控制器 |
| MjHumanoid-v0 | 無 | 同上（隨機策略約 20 多步倒下）|

Walker2d / Humanoid 需要「學習型老師」(PPO/SAC 等) 才能做模仿示範，留待之後。
