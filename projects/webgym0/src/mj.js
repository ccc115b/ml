// MuJoCo 環境：動力學由 MuJoCo (WASM) 計算；obs / reward / 終止條件仿 Gymnasium 重寫（非逐位元相同，故命名 Mj*）。
import { rng, register } from './core.js';
import { XML } from './mjModels.js';
let MJ = null;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sq = a => [].concat(a).reduce((s, x) => s + clamp(x, -1, 1) ** 2, 0);

class MjEnv {
  continuous = true; range = [-1, 1];
  init() { this.model = MJ.MjModel.from_xml_string(this.xml); this.data = new MJ.MjData(this.model); this.dt = this.model.opt.timestep * this.frameSkip; return this; }
  close() { this.data.delete(); this.model.delete(); } // WASM heap 不會被 JS GC 回收，需手動釋放
  begin(seed) { this.rng = rng(seed); MJ.mj_resetData(this.model, this.data); this.t = 0; }
  u(a, b) { return a + (b - a) * this.rng(); }
  gauss() { return Math.sqrt(-2 * Math.log(1 - this.rng())) * Math.cos(2 * Math.PI * this.rng()); }
  clipA(a) { return [].concat(a).map(v => clamp(v, this.range[0], this.range[1])); }
  cost(a) { return this.clipA(a).reduce((s, x) => s + x * x, 0); }
  sim(a) { const c = this.data.ctrl; this.clipA(a).forEach((v, i) => c[i] = v);
    for (let i = 0; i < this.frameSkip; i++) MJ.mj_step(this.model, this.data); this.t++; }
  // 通用側視渲染 (x-z 平面)：由 MuJoCo 的 geom 位姿畫膠囊/球，鏡頭跟隨根節點
  renderSide(g, W, H, k = 110) { const m = this.model, d = this.data, cx = this.q[0], X = x => W / 2 + (x - cx) * k, Z = z => H * .82 - z * k;
    g.strokeStyle = '#8886'; g.beginPath(); g.moveTo(0, Z(0)); g.lineTo(W, Z(0)); g.stroke();
    for (let i = Math.floor(cx - W / 2 / k); i <= cx + W / 2 / k; i++) { g.beginPath(); g.moveTo(X(i), Z(0)); g.lineTo(X(i), Z(0) + 8); g.stroke(); }
    g.lineCap = 'round';
    for (let i = 0; i < m.ngeom; i++) { const t = m.geom_type[i]; if (t != 2 && t != 3) continue; // 2 球 3 膠囊
      const P = d.geom_xpos, R = d.geom_xmat, r = m.geom_size[3 * i], h = t == 3 ? m.geom_size[3 * i + 1] : 0, ax = R[9 * i + 2], az = R[9 * i + 8];
      g.strokeStyle = i % 2 ? '#378add' : '#639922'; g.lineWidth = Math.max(2, 2 * r * k);
      g.beginPath(); g.moveTo(X(P[3 * i] - h * ax), Z(P[3 * i + 2] - h * az)); g.lineTo(X(P[3 * i] + h * ax), Z(P[3 * i + 2] + h * az)); g.stroke(); }
    g.lineWidth = 1; }
  get q() { return this.data.qpos; } get v() { return this.data.qvel; }
}

/* ---------- Reacher：2 連桿手臂碰目標 ---------- */
const REACHER = `<mujoco><compiler angle="radian" inertiafromgeom="true"/>
<default><joint armature="1" damping="1" limited="true"/><geom contype="0" conaffinity="0"/></default>
<option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/><worldbody>
<body pos="0 0 .01"><geom fromto="0 0 0 0.1 0 0" size=".01" type="capsule"/><joint axis="0 0 1" limited="false" name="j0" type="hinge"/>
<body pos="0.1 0 0"><joint axis="0 0 1" name="j1" range="-3 3" type="hinge"/><geom fromto="0 0 0 0.1 0 0" size=".01" type="capsule"/>
<body pos="0.11 0 0"><geom size=".01" type="sphere"/></body></body></body>
<body pos="0 0 .01"><joint armature="0" axis="1 0 0" damping="0" range="-.27 .27" type="slide"/><joint armature="0" axis="0 1 0" damping="0" range="-.27 .27" type="slide"/><geom size=".009" type="sphere"/></body>
</worldbody><actuator><motor ctrllimited="true" ctrlrange="-1 1" gear="200" joint="j0"/><motor ctrllimited="true" ctrlrange="-1 1" gear="200" joint="j1"/></actuator></mujoco>`;
class Reacher extends MjEnv {
  id = 'MjReacher-v0'; nA = 2; maxSteps = 50; frameSkip = 2; slow = 4; xml = REACHER;
  tip() { const q = this.q, a = q[0], b = q[0] + q[1]; return [.1 * Math.cos(a) + .11 * Math.cos(b), .1 * Math.sin(a) + .11 * Math.sin(b)]; }
  obs() { const q = this.q, v = this.v, [fx, fy] = this.tip();
    return [Math.cos(q[0]), Math.cos(q[1]), Math.sin(q[0]), Math.sin(q[1]), q[2], q[3], v[0], v[1], fx - q[2], fy - q[3]]; }
  reset(seed) { this.begin(seed); const q = this.q, v = this.v; q[0] = this.u(-.1, .1); q[1] = this.u(-.1, .1);
    let tx, ty; do { tx = this.u(-.2, .2); ty = this.u(-.2, .2); } while (Math.hypot(tx, ty) >= .2);
    q[2] = tx; q[3] = ty; v[0] = this.u(-.005, .005); v[1] = this.u(-.005, .005); MJ.mj_forward(this.model, this.data); return this.obs(); }
  step(a) { this.sim(a); const o = this.obs(), d = Math.hypot(o[8], o[9]);
    return { obs: o, reward: -d - sq(a), done: false, truncated: this.t >= this.maxSteps, dist: d }; }
  render(g, W, H) { const k = Math.min(W, H) / 2 / .3 * .95, X = x => W / 2 + x * k, Y = y => H / 2 - y * k, q = this.q, a = q[0], tp = this.tip();
    g.strokeStyle = '#8886'; g.beginPath(); g.arc(W / 2, H / 2, .21 * k, 0, 7); g.stroke();
    g.lineWidth = 8; g.lineCap = 'round'; g.strokeStyle = '#378add'; g.beginPath(); g.moveTo(X(0), Y(0));
    g.lineTo(X(.1 * Math.cos(a)), Y(.1 * Math.sin(a))); g.lineTo(X(tp[0]), Y(tp[1])); g.stroke(); g.lineWidth = 1;
    g.fillStyle = '#e24b4a'; g.beginPath(); g.arc(X(q[2]), Y(q[3]), 7, 0, 7); g.fill(); }
}

/* ---------- InvertedDoublePendulum：推車平衡雙擺 ---------- */
const IDP = `<mujoco><compiler angle="radian" inertiafromgeom="true"/><default><joint damping="0.05"/><geom contype="0" conaffinity="0"/></default>
<option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/><worldbody>
<body pos="0 0 0"><joint axis="1 0 0" limited="true" margin="0.01" name="slider" pos="0 0 0" range="-1 1" type="slide"/><geom pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
<body pos="0 0 0"><joint axis="0 1 0" pos="0 0 0" type="hinge"/><geom fromto="0 0 0 0 0 0.6" size="0.045 0.3" type="capsule"/>
<body pos="0 0 0.6"><joint axis="0 1 0" pos="0 0 0" type="hinge"/><geom fromto="0 0 0 0 0 0.6" size="0.045 0.3" type="capsule"/></body></body></body>
</worldbody><actuator><motor ctrllimited="true" ctrlrange="-1 1" gear="500" joint="slider"/></actuator></mujoco>`;
class InvertedDoublePendulum extends MjEnv {
  id = 'MjInvertedDoublePendulum-v0'; nA = 1; maxSteps = 1000; frameSkip = 5; xml = IDP;
  obs() { const q = this.q, v = this.v; return [q[0], Math.sin(q[1]), Math.sin(q[2]), Math.cos(q[1]), Math.cos(q[2]), clamp(v[0], -10, 10), clamp(v[1], -10, 10), clamp(v[2], -10, 10)]; }
  tip() { const q = this.q, a = q[1], b = q[1] + q[2]; return [q[0] + .6 * Math.sin(a) + .6 * Math.sin(b), .6 * Math.cos(a) + .6 * Math.cos(b)]; }
  reset(seed) { this.begin(seed); const q = this.q, v = this.v; for (let i = 0; i < 3; i++) { q[i] = this.u(-.1, .1); v[i] = .1 * this.gauss(); }
    MJ.mj_forward(this.model, this.data); return this.obs(); }
  step(a) { this.sim(a); const [x, z] = this.tip(), v = this.v, done = z <= 1;
    const reward = 10 - (.01 * x * x + (z - 2) ** 2) - (1e-3 * v[1] ** 2 + 5e-3 * v[2] ** 2);
    return { obs: this.obs(), reward, done, truncated: !done && this.t >= this.maxSteps }; }
  // 以有限差分在直立平衡點線性化 s'=A s+B u，s=[x,θ1,θ2,ẋ,ω1,ω2]；一步 = frameSkip 個 mj_step
  linearize(eps = 1e-5) { const m = this.model, d = new MJ.MjData(m);
    const f = (s, u) => { MJ.mj_resetData(m, d); for (let i = 0; i < 3; i++) { d.qpos[i] = s[i]; d.qvel[i] = s[i + 3]; } d.ctrl[0] = u;
      for (let i = 0; i < this.frameSkip; i++) MJ.mj_step(m, d); return [0, 1, 2].map(i => d.qpos[i]).concat([0, 1, 2].map(i => d.qvel[i])); };
    const z = Array(6).fill(0), col = j => { const p = z.slice(), n = z.slice(); p[j] = eps; n[j] = -eps; const a = f(p, 0), b = f(n, 0); return a.map((v, i) => (v - b[i]) / (2 * eps)); };
    const cols = z.map((_, j) => col(j)), A = z.map((_, i) => cols.map(c => c[i]));
    const bp = f(z, eps), bn = f(z, -eps), B = bp.map((v, i) => [(v - bn[i]) / (2 * eps)]); d.delete(); return [A, B]; }
  render(g, W, H) { const k = Math.min(W / 3, H / 1.7), X = x => W / 2 + x * k, Y = z => H * .78 - z * k, q = this.q, a = q[1], b = q[1] + q[2];
    const p0 = [q[0], 0], p1 = [q[0] + .6 * Math.sin(a), .6 * Math.cos(a)], p2 = this.tip();
    g.strokeStyle = '#888'; g.beginPath(); g.moveTo(X(-1), Y(0)); g.lineTo(X(1), Y(0)); g.stroke();
    g.fillStyle = '#378add'; g.fillRect(X(p0[0]) - 25, Y(0) - 12, 50, 24);
    g.lineWidth = 9; g.lineCap = 'round'; g.strokeStyle = '#d85a30'; g.beginPath(); g.moveTo(X(p0[0]), Y(0)); g.lineTo(X(p1[0]), Y(p1[1])); g.lineTo(X(p2[0]), Y(p2[1])); g.stroke(); g.lineWidth = 1; }
}

/* ---------- Swimmer：三節游泳者 (黏滯流體)；obs 帶「步態時鐘」sin/cos ---------- */
const SWIM = `<mujoco><compiler angle="degree" inertiafromgeom="true"/><option density="4000" integrator="RK4" timestep="0.01" viscosity="0.1"/>
<default><geom contype="0" conaffinity="0" condim="1"/><joint armature="0.1"/></default><worldbody>
<body pos="0 0 0"><geom density="1000" fromto="1.5 0 0 0.5 0 0" size="0.1" type="capsule"/><joint axis="1 0 0" type="slide"/><joint axis="0 1 0" type="slide"/><joint axis="0 0 1" type="hinge"/>
<body pos="0.5 0 0"><geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/><joint axis="0 0 1" limited="true" name="rot2" range="-100 100" type="hinge"/>
<body pos="-1 0 0"><geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/><joint axis="0 0 1" limited="true" name="rot3" range="-100 100" type="hinge"/></body></body></body>
</worldbody><actuator><motor ctrllimited="true" ctrlrange="-1 1" gear="150" joint="rot2"/><motor ctrllimited="true" ctrlrange="-1 1" gear="150" joint="rot3"/></actuator></mujoco>`;
class Swimmer extends MjEnv {
  id = 'MjSwimmer-v0'; nA = 2; maxSteps = 1000; frameSkip = 4; spf = 3; xml = SWIM; period = 1.8; // period: 步態時鐘週期(秒)
  obs() { const q = this.q, v = this.v, c = 2 * Math.PI * this.t * this.dt / this.period;
    return [q[2], q[3], q[4], v[0], v[1], v[2], v[3], v[4], Math.sin(c), Math.cos(c)]; }
  reset(seed) { this.begin(seed); const q = this.q, v = this.v; for (let i = 0; i < 5; i++) { q[i] = this.u(-.1, .1); v[i] = this.u(-.1, .1); }
    MJ.mj_forward(this.model, this.data); return this.obs(); }
  step(a) { const x0 = this.q[0]; this.sim(a); const vx = (this.q[0] - x0) / this.dt;
    return { obs: this.obs(), reward: vx - 1e-4 * sq(a), done: false, truncated: this.t >= this.maxSteps }; }
  render(g, W, H) { const q = this.q, [x, y, r, r2, r3] = [0, 1, 2, 3, 4].map(i => q[i]), R = (a, l) => [l * Math.cos(a), l * Math.sin(a)], ad = (p, v) => [p[0] + v[0], p[1] + v[1]];
    const head = ad([x, y], R(r, 1.5)), j1 = ad([x, y], R(r, .5)), a1 = r + r2, j2 = ad(j1, R(a1, -1)), tail = ad(j2, R(r + r2 + r3, -1));
    const k = 40, X = p => W / 2 + (p - j1[0]) * k, Y = p => H / 2 - (p - j1[1]) * k;
    g.strokeStyle = '#8883'; for (let i = Math.floor(j1[0] - 9); i <= j1[0] + 9; i++) { g.beginPath(); g.moveTo(X(i), 0); g.lineTo(X(i), H); g.stroke(); }
    for (let i = Math.floor(j1[1] - 4); i <= j1[1] + 4; i++) { g.beginPath(); g.moveTo(0, Y(i)); g.lineTo(W, Y(i)); g.stroke(); }
    g.lineWidth = 12; g.lineCap = 'round'; g.lineJoin = 'round'; g.strokeStyle = '#639922'; g.beginPath(); g.moveTo(X(head[0]), Y(head[1]));
    [j1, j2, tail].forEach(p => g.lineTo(X(p[0]), Y(p[1]))); g.stroke(); g.lineWidth = 1; }
}

/* ---------- 運動環境：HalfCheetah / Walker2d / Humanoid（Gymnasium 原始模型，obs/reward 依 v5 設計簡化重寫）---------- */
class Loco extends MjEnv {
  maxSteps = 1000; qSkip = 1; vClip = 0; noise = .01; gaussV = false; fwdW = 1; ctrlW = 0; alive = 0; term = false; clock = false; period = 1; k = 110;
  healthy() { return true; }
  obs() { const q = Array.from(this.q).slice(this.qSkip), v = Array.from(this.v, x => this.vClip ? clamp(x, -this.vClip, this.vClip) : x);
    if (!this.clock) return [...q, ...v]; const c = 2 * Math.PI * this.t * this.dt / this.period; return [...q, ...v, Math.sin(c), Math.cos(c)]; }
  reset(seed) { this.begin(seed); const q = this.q, v = this.v, s = this.noise;
    for (let i = 0; i < q.length; i++) q[i] += this.u(-s, s); for (let i = 0; i < v.length; i++) v[i] += this.gaussV ? .1 * this.gauss() : this.u(-s, s);
    MJ.mj_forward(this.model, this.data); return this.obs(); }
  step(a) { const x0 = this.q[0]; this.sim(a); const vx = (this.q[0] - x0) / this.dt, ok = this.healthy(), done = this.term && !ok;
    return { obs: this.obs(), reward: this.fwdW * vx + (ok ? this.alive : 0) - this.ctrlW * this.cost(a), done, truncated: !done && this.t >= this.maxSteps, vx }; }
  render(g, W, H) { this.renderSide(g, W, H, this.k); }
}
class HalfCheetah extends Loco { id = 'MjHalfCheetah-v0'; nA = 6; frameSkip = 5; spf = 2; xml = XML.cheetah; noise = .1; gaussV = true; ctrlW = .1; clock = true; period = PERIOD_CHEETAH; }
class Walker2d extends Loco { id = 'MjWalker2d-v0'; nA = 6; frameSkip = 4; xml = XML.walker; vClip = 10; noise = 5e-3; ctrlW = 1e-3; alive = 1; term = true;
  healthy() { const q = this.q; return q[1] > .8 && q[1] < 2 && Math.abs(q[2]) < 1; } }
class Humanoid extends Loco { id = 'MjHumanoid-v0'; nA = 17; frameSkip = 5; xml = XML.humanoid; qSkip = 2; noise = .01; fwdW = 1.25; ctrlW = .1; alive = 5; term = true; range = [-.4, .4]; k = 90;
  healthy() { const z = this.q[2]; return z > 1 && z < 2; } }
const PERIOD_CHEETAH = 0.25; // 步態時鐘週期(秒)，與 closedForm.js 的 CHEETAH 參數一組

export const mjIds = ['MjReacher-v0', 'MjInvertedDoublePendulum-v0', 'MjSwimmer-v0', 'MjHalfCheetah-v0', 'MjWalker2d-v0', 'MjHumanoid-v0'];
// loadMujoco: 回傳 MuJoCo 模組的 async 函式（Node: 套件 default export；瀏覽器: CDN 的 ESM）
export async function initMujoco(loadMujoco) {
  if (!MJ) MJ = await loadMujoco();
  const open = C => () => new C().init();
  register('MjReacher-v0', open(Reacher)); register('MjInvertedDoublePendulum-v0', open(InvertedDoublePendulum)); register('MjSwimmer-v0', open(Swimmer));
  register('MjHalfCheetah-v0', open(HalfCheetah)); register('MjWalker2d-v0', open(Walker2d)); register('MjHumanoid-v0', open(Humanoid));
}
