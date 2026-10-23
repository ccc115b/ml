// 各環境的 closed-form 策略：不訓練、不學習，直接由物理 / 模型推導
// 每個條目: policy(env) -> (obs)=>action ; pass = 通過條件(平均回報下限)
export const closedForm = {
  // 線性狀態回饋 (LQR 風格)：u = k·s
  'CartPole-v1': { policy: () => { const k = [0.5, 1.0, 20, 3]; return o => o.reduce((a, x, i) => a + k[i] * x, 0) > 0 ? 1 : 0; }, pass: 500 },
  // 能量法：永遠順著速度方向推，持續灌入能量
  'MountainCar-v0': { policy: () => o => o[1] < 0 ? 0 : 2, pass: -150 },
  // 已知轉移模型 → 解 Bellman 最適方程式 (value iteration 到不動點)
  'FrozenLake-v1': { policy: env => bellman(env), pass: 0.7 },
  'FrozenLake-v1-noslip': { policy: env => bellman(env), pass: 1.0 },
  // 幾何解：上 → 右×11 → 下，最適回報 -13
  'CliffWalking-v0': { policy: () => o => { const r = Math.floor(o / 12), c = o % 12; return c == 11 ? 2 : r == 3 ? 0 : 1; }, pass: -13 },
  // 哈密頓迴圈：沿著一條走遍所有格子的環一直走 → 永不撞自己，必定吃滿 (n²-1 顆)
  'Snake-v0': { policy: () => hamiltonian, pass: 63 },
  // Braitenberg 交叉連線 (2b)：左眼→右輪、右眼→左輪；加上除法正規化(生物視覺的 gain control)
  // 閉環下偏角 β 的轉向 ≈ 0.3β/步 → 指數收斂到光源
  'Braitenberg-v0': { policy: () => ([sl, sr]) => { const t = (sl - sr) / (sl + sr + 1e-9); return [0.6 * (1 - t), 0.6 * (1 + t)]; }, pass: -80 },
  // —— MuJoCo 環境（解析 / 模型導向控制器）——
  // Reacher：兩連桿解析逆運動學 (餘弦定理) + PD 力矩
  'MjReacher-v0': { policy: () => { const L1 = .1, L2 = .11, Kp = KP, Kd = KD; return o => { const [c0, c1, s0, s1, tx, ty, w0, w1] = o;
      const c = Math.max(-1, Math.min(1, (tx * tx + ty * ty - L1 * L1 - L2 * L2) / (2 * L1 * L2))), q1 = Math.acos(c), q0 = Math.atan2(ty, tx) - Math.atan2(L2 * Math.sin(q1), L1 + L2 * c);
      return [Kp * wrap(q0 - Math.atan2(s0, c0)) - Kd * w0, Kp * (q1 - Math.atan2(s1, c1)) - Kd * w1]; }; }, pass: -6 },
  // 雙擺：在平衡點線性化 → 解離散 Riccati 方程式得 LQR 增益 u = -K s
  'MjInvertedDoublePendulum-v0': { policy: env => { const [A, B] = env.linearize(), K = lqr(A, B, QDIAG.map((q, i) => QDIAG.map((_, j) => i == j ? q : 0)), RR);
      return o => { const s = [o[0], Math.atan2(o[1], o[3]), Math.atan2(o[2], o[4]), o[5], o[6], o[7]]; return [-K.reduce((a, k, i) => a + k * s[i], 0)]; }; }, pass: 9100 },
  // Swimmer：行進波 CPG。步態時鐘 (sin c, cos c) 由環境提供，第二關節落後第一關節 LAG 相位
  'MjSwimmer-v0': { policy: () => o => { const sn = o[8], cs = o[9]; return [AMP * sn, AMP * (sn * Math.cos(LAG) - cs * Math.sin(LAG))]; }, pass: 150 },
  // HalfCheetah：開環正弦步態 a_i = clip(A_i·sin(c+φ_i)+b_i)，c 為環境提供的步態時鐘 (sin c, cos c = obs[17], obs[18])。
  // 注意：A、φ、b 與時鐘週期 0.25s 是離線隨機搜尋 + 爬山得到的，並非推導；A>1 時被 clip 成近似方波 (4 Hz 的高頻步態)
  'MjHalfCheetah-v0': { policy: () => o => CHEETAH.map(([A, ph, b]) => Math.max(-1, Math.min(1, A * (o[17] * Math.cos(ph) + o[18] * Math.sin(ph)) + b))), pass: 3500, studentPass: 3000 },
};
const CHEETAH = [[-1.561, 6.013, .728], [2.496, 5.08, .166], [-1.194, 5.318, .232], [.901, 5.024, .281], [1.866, 5.386, .031], [1.487, 4.795, .226]];
const KP = .35, KD = .06, QDIAG = [1, 50, 50, 1, 5, 5], RR = .1, AMP = 1, LAG = .7; // Swimmer 的 LAG 與環境時鐘週期 1.8s 由離線掃描定出
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const mm = (a, b) => a.map(r => b[0].map((_, j) => r.reduce((s, v, k) => s + v * b[k][j], 0)));
const tr = a => a[0].map((_, j) => a.map(r => r[j]));
function lqr(A, B, Q, R) { // 單一輸入：(R+BᵀPB) 是純量；迭代 Riccati 至收斂
  let P = Q, K; for (let i = 0; i < 50000; i++) { const AtP = mm(tr(A), P), BtP = mm(tr(B), P), s = R + mm(BtP, B)[0][0];
    K = mm(BtP, A)[0].map(v => v / s); const Pn = mm(AtP, A).map((r, a) => r.map((v, b) => Q[a][b] + v - mm(AtP, B)[a][0] * K[b]));
    const d = Math.max(...Pn.flat().map((v, i) => Math.abs(v - P.flat()[i]))); P = Pn; if (d < 1e-9) break; } return K; }

// 偶數 n 的哈密頓迴圈編號：第 0 列往右 → 其餘列在 1..n-1 欄蛇行 → 第 0 欄往上回到起點
const cycleIdx = (r, c, n) => r == 0 ? c : c == 0 ? n + (n - 1) ** 2 + (n - 1 - r)
  : n + (r - 1) * (n - 1) + (r % 2 ? n - 1 - c : c - 1);
function hamiltonian({ snake: [[r, c]], n }) {
  const next = (cycleIdx(r, c, n) + 1) % (n * n);
  return [[-1, 0], [0, 1], [1, 0], [0, -1]].findIndex(([dr, dc]) => {
    const R = r + dr, C = c + dc; return R >= 0 && C >= 0 && R < n && C < n && cycleIdx(R, C, n) == next; });
}

function bellman(env, gamma = 0.99) {
  const V = Array(env.nS).fill(0);
  const Q = (s, a) => env.P(s, a).reduce((q, [p, n, r]) => q + p * (r + (env.term(n) ? 0 : gamma * V[n])), 0);
  for (let i = 0; i < 2000; i++) for (let s = 0; s < env.nS; s++) if (!env.term(s)) V[s] = Math.max(...[0, 1, 2, 3].map(a => Q(s, a)));
  const pi = V.map((_, s) => [0, 1, 2, 3].reduce((b, a) => Q(s, a) > Q(s, b) + 1e-12 ? a : b, 0));
  return o => pi[o];
}
