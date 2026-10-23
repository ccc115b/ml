// 模仿學習 (DAgger)：closed-form 策略 = 老師；TF.js 神經網路 = 學生。
// 學生「不看環境獎勵」，只拿 (觀察值 → 老師的動作) 當監督式標籤。
// 第 0 輪：老師示範；之後每輪：學生自己玩，經過的每個狀態都請老師重新標註，累積資料再訓練。
import { rng, rollout } from './core.js';
import { closedForm } from './closedForm.js';

const ADAPT = { // 學生看到的特徵 / 輸出維度 / 隱藏層
  'MountainCar-v0': { x: o => [(o[0] + 0.3) / 0.9, o[1] / 0.07], out: 3, hidden: [32, 32] },
  'Snake-v0': { x: o => { const v = Array(o.n * o.n).fill(0); v[o.snake[0][0] * o.n + o.snake[0][1]] = 1; return v; }, out: 4, hidden: [64] }, // 只看蛇頭位置
  'Braitenberg-v0': { x: o => [o[0] * 4, o[1] * 4], out: 2, cont: true, hidden: [32, 32] }, // 只看左右眼亮度
  // MuJoCo 環境（連續動作 ∈[-1,1]）
  'MjReacher-v0': { x: o => o, out: 2, cont: true, range: [-1, 1], hidden: [64, 64], opts: { episodes: 30, collectSteps: 50, epochs: 25, cap: 12000 } },
  'MjInvertedDoublePendulum-v0': { x: o => [o[0], Math.atan2(o[1], o[3]), Math.atan2(o[2], o[4]), o[5], o[6], o[7]], out: 1, cont: true, range: [-1, 1], hidden: [32, 32], opts: { episodes: 6, collectSteps: 400 } },
  'MjSwimmer-v0': { x: o => o, out: 2, cont: true, range: [-1, 1], hidden: [32, 32], opts: { episodes: 3, collectSteps: 300 } },
};
export const imitable = id => id in ADAPT;

export function createStudent(tf, env, userOpts = {}) {
  const ad = ADAPT[env.id], { lr = 0.01, episodes = 4, collectSteps = 300, epochs = 30, seed = 1, cap = 8000 } = { ...ad.opts, ...userOpts }, [lo, hi] = ad.range || [0, 1], teacher = closedForm[env.id].policy(env), cont = !!ad.cont, A = ad.out;
  const D = ad.x(env.reset(0)).length, norm = a => cont ? [].concat(a).map(v => (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) : a;
  const init = s => tf.initializers.glorotUniform({ seed: s });
  const model = tf.sequential({ layers: [
    ...ad.hidden.map((u, i) => tf.layers.dense({ ...(i ? {} : { inputShape: [D] }), units: u, activation: 'tanh', kernelInitializer: init(seed + i) })),
    tf.layers.dense({ units: A, activation: cont ? 'sigmoid' : 'softmax', kernelInitializer: init(seed + 9) })] });
  model.compile({ optimizer: tf.train.adam(lr), loss: cont ? 'meanSquaredError' : 'categoricalCrossentropy' });
  let X = [], Y = [], Ws = null, it = 0;
  const sync = () => Ws = model.getWeights().map(w => w.dataSync().slice());
  const fwd = x => { let h = x; for (let l = 0; l < Ws.length; l += 2) { const k = Ws[l], b = Ws[l + 1], n = b.length, y = new Float32Array(n);
      for (let j = 0; j < n; j++) { let s = b[j]; for (let i = 0; i < h.length; i++) s += h[i] * k[i * n + j]; y[j] = s; }
      h = l + 2 < Ws.length ? y.map(Math.tanh) : y; } return h; }; // 純 JS 前向傳播 (輸出為 softmax/sigmoid 前的值)
  const act = o => { const y = fwd(ad.x(o)); return cont ? Array.from(y, v => lo + (hi - lo) / (1 + Math.exp(-v))) : y.indexOf(Math.max(...y)); };

  async function iterate(evalEps = 3) {
    for (let e = 0; e < episodes; e++) { let o = env.reset(10000 + it * 100 + e);
      for (let s = 0; s < collectSteps; s++) { const raw = teacher(o), lab = norm(raw); X.push(ad.x(o)); Y.push(lab);
        const st = env.step(it === 0 ? raw : act(o)); o = st.obs; if (st.done || st.truncated) break; } }
    X = X.slice(-cap); Y = Y.slice(-cap);
    const xs = tf.tensor2d(X), ys = cont ? tf.tensor2d(Y) : tf.oneHot(tf.tensor1d(Y, 'int32'), A).toFloat();
    const h = await model.fit(xs, ys, { epochs, batchSize: 128, shuffle: true, verbose: 0 }); xs.dispose(); ys.dispose(); sync();
    let m = 0; X.forEach((x, i) => { const y = fwd(x); m += cont ? Y[i].reduce((a, t, j) => a + Math.abs(1 / (1 + Math.exp(-y[j])) - t), 0) / A : +(y.indexOf(Math.max(...y)) == Y[i]); });
    let R = 0; for (let i = 0; i < evalEps; i++) R += rollout(env, act, 777 + i).R;
    return { iter: ++it, n: X.length, loss: h.history.loss.at(-1), match: m / X.length, mae: m / X.length, studentReturn: R / evalEps };
  }
  return { iterate, act, cont };
}
