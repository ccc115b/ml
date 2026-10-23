import { rng, register } from './core.js';

/* ---------- CartPole-v1 (與 Gym 相同物理) ---------- */
class CartPole {
  id = 'CartPole-v1'; nA = 2; maxSteps = 500;
  reset(seed) { this.rng = rng(seed); this.s = [0, 0, 0, 0].map(() => this.rng() * 0.1 - 0.05); this.t = 0; return [...this.s]; }
  step(a) {
    let [x, v, th, w] = this.s; const f = a ? 10 : -10, c = Math.cos(th), sn = Math.sin(th);
    const tmp = (f + 0.05 * w * w * sn) / 1.1;
    const tha = (9.8 * sn - c * tmp) / (0.5 * (4 / 3 - 0.1 * c * c / 1.1)), xa = tmp - 0.05 * tha * c / 1.1;
    x += 0.02 * v; v += 0.02 * xa; th += 0.02 * w; w += 0.02 * tha; this.s = [x, v, th, w]; this.t++;
    const done = Math.abs(x) > 2.4 || Math.abs(th) > 0.2095;
    return { obs: [...this.s], reward: 1, done, truncated: !done && this.t >= this.maxSteps };
  }
  render(g, W, H) {
    const [x, , th] = this.s, cx = W / 2 + x * W / 5.2, cy = H * 0.65;
    g.strokeStyle = '#888'; g.beginPath(); g.moveTo(0, cy + 15); g.lineTo(W, cy + 15); g.stroke();
    g.fillStyle = '#378add'; g.fillRect(cx - 30, cy - 15, 60, 30);
    g.strokeStyle = '#d85a30'; g.lineWidth = 6; g.beginPath(); g.moveTo(cx, cy - 15);
    g.lineTo(cx + Math.sin(th) * 100, cy - 15 - Math.cos(th) * 100); g.stroke(); g.lineWidth = 1;
  }
}

/* ---------- FrozenLake 4x4 (slippery / 非 slippery) ---------- */
const MAP = ['SFFF', 'FHFH', 'FFFH', 'HFFG'];
class FrozenLake {
  nA = 4; maxSteps = 100; nS = 16; // 0左 1下 2右 3上
  constructor(slip) { this.slip = slip; this.id = slip ? 'FrozenLake-v1' : 'FrozenLake-v1-noslip'; }
  move(s, a) { let r = s >> 2, c = s & 3;
    if (a == 0) c = Math.max(c - 1, 0); if (a == 1) r = Math.min(r + 1, 3);
    if (a == 2) c = Math.min(c + 1, 3); if (a == 3) r = Math.max(r - 1, 0); return r * 4 + c; }
  term(s) { return 'HG'.includes(MAP[s >> 2][s & 3]); }
  P(s, a) { const as = this.slip ? [(a + 3) % 4, a, (a + 1) % 4] : [a]; // 轉移模型 [prob, next, reward]
    return as.map(x => { const n = this.move(s, x); return [1 / as.length, n, MAP[n >> 2][n & 3] == 'G' ? 1 : 0]; }); }
  reset(seed) { this.rng = rng(seed); this.s = 0; this.t = 0; return 0; }
  step(a) { const outs = this.P(this.s, a); let u = this.rng(), o = outs[outs.length - 1];
    for (const x of outs) { if (u < x[0]) { o = x; break; } u -= x[0]; }
    this.s = o[1]; this.t++; const done = this.term(this.s);
    return { obs: this.s, reward: o[2], done, truncated: !done && this.t >= this.maxSteps }; }
  render(g, W, H) { const k = Math.min(W, H) / 4 - 4;
    for (let s = 0; s < 16; s++) { const ch = MAP[s >> 2][s & 3], x = (s & 3) * (k + 4) + 20, y = (s >> 2) * (k + 4) + 10;
      g.fillStyle = { S: '#b5d4f4', F: '#e6f1fb', H: '#444', G: '#97c459' }[ch]; g.fillRect(x, y, k, k);
      if (s == this.s) { g.fillStyle = '#d85a30'; g.beginPath(); g.arc(x + k / 2, y + k / 2, k / 4, 0, 7); g.fill(); } }
  }
}

/* ---------- CliffWalking 4x12 ---------- */
class CliffWalking {
  id = 'CliffWalking-v0'; nA = 4; nS = 48; maxSteps = 200; // 0上 1右 2下 3左
  reset(seed) { this.s = [3, 0]; this.t = 0; return 36; }
  step(a) { let [r, c] = this.s;
    if (a == 0) r = Math.max(r - 1, 0); if (a == 1) c = Math.min(c + 1, 11);
    if (a == 2) r = Math.min(r + 1, 3); if (a == 3) c = Math.max(c - 1, 0);
    let reward = -1; if (r == 3 && c > 0 && c < 11) { reward = -100; r = 3; c = 0; }
    this.s = [r, c]; this.t++; const done = r == 3 && c == 11;
    return { obs: r * 12 + c, reward, done, truncated: !done && this.t >= this.maxSteps }; }
  render(g, W, H) { const k = Math.min(W / 12, H / 4) - 3;
    for (let r = 0; r < 4; r++) for (let c = 0; c < 12; c++) {
      g.fillStyle = r == 3 && c > 0 && c < 11 ? '#444' : r == 3 && c == 11 ? '#97c459' : '#e6f1fb';
      g.fillRect(c * (k + 3) + 5, r * (k + 3) + 10, k, k); }
    g.fillStyle = '#d85a30'; g.beginPath(); g.arc(this.s[1] * (k + 3) + 5 + k / 2, this.s[0] * (k + 3) + 10 + k / 2, k / 3, 0, 7); g.fill(); }
}

/* ---------- MountainCar-v0 ---------- */
class MountainCar {
  id = 'MountainCar-v0'; nA = 3; maxSteps = 200; // 0左推 1不動 2右推
  reset(seed) { this.rng = rng(seed); this.s = [-0.6 + this.rng() * 0.2, 0]; this.t = 0; return [...this.s]; }
  step(a) { let [p, v] = this.s; v += (a - 1) * 0.001 + Math.cos(3 * p) * -0.0025; v = Math.max(-0.07, Math.min(0.07, v));
    p = Math.max(-1.2, Math.min(0.6, p + v)); if (p == -1.2 && v < 0) v = 0; this.s = [p, v]; this.t++;
    const done = p >= 0.5; return { obs: [p, v], reward: -1, done, truncated: !done && this.t >= this.maxSteps }; }
  render(g, W, H) { const X = p => (p + 1.2) / 1.8 * (W - 40) + 20, Y = p => H * 0.8 - (Math.sin(3 * p) * 0.45 + 0.55) * H * 0.5;
    g.strokeStyle = '#888'; g.beginPath(); for (let p = -1.2; p <= 0.6; p += 0.02) g.lineTo(X(p), Y(p)); g.stroke();
    g.fillStyle = '#97c459'; g.fillRect(X(0.5), Y(0.5) - 30, 4, 30);
    g.fillStyle = '#d85a30'; g.beginPath(); g.arc(X(this.s[0]), Y(this.s[0]) - 8, 8, 0, 7); g.fill(); }
}

/* ---------- Snake 8x8 ---------- */
class Snake {
  id = 'Snake-v0'; nA = 4; maxSteps = 10000; spf = 8; n = 8; // 0上 1右 2下 3左；反向移動視為無效(沿原方向)
  reset(seed) { this.rng = rng(seed); this.snake = [[0, 0]]; this.dir = 1; this.t = 0; this.food = this.spawn(); return this.obs(); }
  obs() { return { snake: this.snake.map(c => [...c]), food: [...this.food], n: this.n }; }
  spawn() { const free = []; for (let r = 0; r < this.n; r++) for (let c = 0; c < this.n; c++)
      if (!this.snake.some(b => b[0] == r && b[1] == c)) free.push([r, c]); return free[Math.floor(this.rng() * free.length)]; }
  step(a) { if ((a + 2) % 4 != this.dir || this.snake.length == 1) this.dir = a;
    const [dr, dc] = [[-1, 0], [0, 1], [1, 0], [0, -1]][this.dir], h = this.snake[0], nh = [h[0] + dr, h[1] + dc];
    const eat = nh[0] == this.food[0] && nh[1] == this.food[1], body = eat ? this.snake : this.snake.slice(0, -1);
    this.t++; const dead = nh[0] < 0 || nh[1] < 0 || nh[0] >= this.n || nh[1] >= this.n || body.some(b => b[0] == nh[0] && b[1] == nh[1]);
    if (dead) return { obs: this.obs(), reward: 0, done: true, truncated: false };
    this.snake = [nh, ...body]; let win = false;
    if (eat) { win = this.snake.length == this.n * this.n; if (!win) this.food = this.spawn(); }
    return { obs: this.obs(), reward: eat ? 1 : 0, done: win, truncated: !win && this.t >= this.maxSteps }; }
  render(g, W, H) { const k = Math.min(W, H) / this.n - 2, ox = (W - this.n * (k + 2)) / 2;
    for (let r = 0; r < this.n; r++) for (let c = 0; c < this.n; c++) { g.fillStyle = '#8881'; g.fillRect(ox + c * (k + 2), r * (k + 2), k, k); }
    this.snake.forEach((b, i) => { g.fillStyle = i ? '#639922' : '#27500a'; g.fillRect(ox + b[1] * (k + 2), b[0] * (k + 2), k, k); });
    g.fillStyle = '#e24b4a'; g.beginPath(); g.arc(ox + this.food[1] * (k + 2) + k / 2, this.food[0] * (k + 2) + k / 2, k / 3, 0, 7); g.fill(); }
}

/* ---------- Braitenberg-v0：仿生趨光車 (連續動作 = [左輪, 右輪] ∈ [0,1]) ---------- */
class Braitenberg {
  id = 'Braitenberg-v0'; nA = 2; continuous = true; maxSteps = 600; spf = 2;
  reset(seed) { this.rng = rng(seed); const u = () => this.rng();
    this.p = [1 + 8 * u(), 1 + 8 * u()]; this.th = u() * 2 * Math.PI;
    do { this.light = [1 + 8 * u(), 1 + 8 * u()]; } while (Math.hypot(this.light[0] - this.p[0], this.light[1] - this.p[1]) < 4);
    this.t = 0; this.trail = [[...this.p]]; return this.sense(); }
  sense() { const [x, y] = this.p, d = Math.hypot(this.light[0] - x, this.light[1] - y), b = Math.atan2(this.light[1] - y, this.light[0] - x);
    const f = a => (1 + Math.cos(b - this.th - a)) / 2 / (1 + d); return [f(0.6), f(-0.6)]; } // 觀察值只有 [左眼, 右眼]
  step(a) { const [vl, vr] = a.map(v => Math.max(0, Math.min(1, v))); this.th += 0.8 * (vr - vl); const v = 0.3 * (vl + vr) / 2;
    this.p = this.p.map((q, i) => Math.max(0, Math.min(10, q + v * (i ? Math.sin(this.th) : Math.cos(this.th)))));
    this.trail.push([...this.p]); this.t++; const done = Math.hypot(this.light[0] - this.p[0], this.light[1] - this.p[1]) < 0.5;
    return { obs: this.sense(), reward: -1, done, truncated: !done && this.t >= this.maxSteps }; }
  render(g, W, H) { const S = H - 20, k = S / 10, ox = (W - S) / 2, X = x => ox + x * k, Y = y => H - 10 - y * k;
    g.strokeStyle = '#888'; g.strokeRect(ox, 10, S, S); g.strokeStyle = '#ef9f2788'; g.beginPath();
    this.trail.forEach((q, i) => i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))); g.stroke();
    g.fillStyle = '#ef9f27'; g.beginPath(); g.arc(X(this.light[0]), Y(this.light[1]), 10, 0, 7); g.fill();
    const [x, y] = this.p, c = Math.cos(this.th), s = Math.sin(this.th); g.fillStyle = '#378add'; g.beginPath();
    g.moveTo(X(x + .35 * c), Y(y + .35 * s)); g.lineTo(X(x - .25 * c - .2 * s), Y(y - .25 * s + .2 * c)); g.lineTo(X(x - .25 * c + .2 * s), Y(y - .25 * s - .2 * c)); g.fill(); }
}

register('CartPole-v1', () => new CartPole());
register('FrozenLake-v1', () => new FrozenLake(true));
register('FrozenLake-v1-noslip', () => new FrozenLake(false));
register('CliffWalking-v0', () => new CliffWalking());
register('MountainCar-v0', () => new MountainCar());
register('Snake-v0', () => new Snake());
register('Braitenberg-v0', () => new Braitenberg());
