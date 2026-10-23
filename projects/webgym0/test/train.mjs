// 需要: npm i -D @tensorflow/tfjs   (純 JS 後端，不需編譯)
import * as tf from '@tensorflow/tfjs';
import { make, rollout } from '../src/core.js'; import '../src/envs.js'; import { createAgent } from '../src/rl.js';
const cases = [['CartPole-v1', 400, 475, 195], ['FrozenLake-v1-noslip', 300, 0.99, 1], ['CliffWalking-v0', 300, -16, -17]];
let fail = 0;
for (const [id, maxIt, stop, pass] of cases) {
  const ag = createAgent(tf, make(id)), h = []; let it = 0; const t0 = Date.now();
  for (; it < maxIt; it++) { h.push(ag.trainIter()); const m = h.slice(-5).reduce((a, b) => a + b) / Math.min(5, h.length); if (it >= 5 && m >= stop) break; }
  const env = make(id); let s = 0; for (let i = 0; i < 20; i++) s += rollout(env, ag.greedy, 5000 + i).R;
  const ok = s / 20 >= pass; if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${id.padEnd(22)} iters=${it} greedy平均=${(s / 20).toFixed(2)} (需 ≥ ${pass}) ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
process.exit(fail ? 1 : 0);
