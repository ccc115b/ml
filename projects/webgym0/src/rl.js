// REINFORCE (策略梯度) + entropy bonus。tf 由外部傳入：瀏覽器用 CDN 全域 tf，Node 用 npm 套件。
// 取樣用純 JS 前向傳播（快），梯度更新用 TF.js (autodiff + Adam)。
import { rng } from './core.js';
export const trainable = id => /CartPole|FrozenLake|CliffWalking/.test(id); // 離散動作 + 向量/整數觀察

export function createAgent(tf, env, { hidden = 32, lr = 0.02, batch = 8, gamma = 0.99, seed = 1 } = {}) {
  const enc = o => typeof o === 'number' ? Array.from({ length: env.nS }, (_, i) => +(i === o)) : o; // 整數狀態 → one-hot
  const D = enc(env.reset(0)).length, A = env.nA;
  const init = s => tf.initializers.glorotUniform({ seed: s });
  const model = tf.sequential({ layers: [
    tf.layers.dense({ inputShape: [D], units: hidden, activation: 'tanh', kernelInitializer: init(seed) }),
    tf.layers.dense({ units: A, kernelInitializer: init(seed + 1) })] }); // 輸出 logits
  const opt = tf.train.adam(lr); let W;
  const sync = () => W = model.getWeights().map(w => w.dataSync().slice());
  const probs = x => { const [k1, b1, k2, b2] = W, h = new Float32Array(hidden), l = [];
    for (let j = 0; j < hidden; j++) { let s = b1[j]; for (let i = 0; i < D; i++) s += x[i] * k1[i * hidden + j]; h[j] = Math.tanh(s); }
    for (let a = 0; a < A; a++) { let s = b2[a]; for (let j = 0; j < hidden; j++) s += h[j] * k2[j * A + a]; l.push(s); }
    const m = Math.max(...l), e = l.map(v => Math.exp(v - m)), z = e.reduce((p, c) => p + c); return e.map(v => v / z); };
  const r = rng(seed * 7919); let ep = 0; sync();
  const sample = x => { let u = r(); const p = probs(x); for (let a = 0; a < A; a++) { if (u < p[a]) return a; u -= p[a]; } return A - 1; };
  const greedy = o => { const p = probs(enc(o)); return p.indexOf(Math.max(...p)); };

  function trainIter() { // 收集 batch 個回合 → 一次梯度更新；回傳這批的平均回報
    sync(); const S = [], Ac = [], G = [], rets = [];
    for (let b = 0; b < batch; b++) {
      let o = env.reset(1000 + ep++), rs = [];
      for (;;) { const x = enc(o), a = sample(x), st = env.step(a); S.push(x); Ac.push(a); rs.push(st.reward); o = st.obs; if (st.done || st.truncated) break; }
      let g = 0; const gs = Array(rs.length); for (let t = rs.length - 1; t >= 0; t--) { g = rs[t] + gamma * g; gs[t] = g; }
      G.push(...gs); rets.push(rs.reduce((p, c) => p + c, 0));
    }
    const mean = G.reduce((p, c) => p + c, 0) / G.length, sd = Math.sqrt(G.reduce((p, c) => p + (c - mean) ** 2, 0) / G.length) + 1e-8;
    tf.tidy(() => { const xs = tf.tensor2d(S), as = tf.oneHot(tf.tensor1d(Ac, 'int32'), A), adv = tf.tensor1d(G.map(g => (g - mean) / sd));
      opt.minimize(() => { const p = tf.softmax(model.apply(xs)), lp = tf.log(p.mul(as).sum(1).add(1e-8));
        const negEnt = p.mul(tf.log(p.add(1e-8))).sum(1).mean();
        return lp.mul(adv).mean().neg().add(negEnt.mul(0.01)); }); });
    return rets.reduce((p, c) => p + c, 0) / rets.length;
  }
  return { greedy, trainIter, model };
}
