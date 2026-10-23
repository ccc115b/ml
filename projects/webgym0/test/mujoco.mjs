// 需要 npm i -D @mujoco/mujoco @tensorflow/tfjs。
// 有老師的環境：(1) 解析控制器通過 (2) TF.js 學生模仿老師後也通過；無老師的環境 (Walker2d/Humanoid)：隨機基準合理性檢查
import loadMujoco from '@mujoco/mujoco'; import * as tf from '@tensorflow/tfjs';
import { make, rollout, rng } from '../src/core.js'; import { initMujoco, mjIds } from '../src/mj.js';
import { closedForm } from '../src/closedForm.js'; import { createStudent } from '../src/imitate.js';
await initMujoco(loadMujoco); let fail = 0;
const N = { 'MjReacher-v0': 30, 'MjInvertedDoublePendulum-v0': 10, 'MjSwimmer-v0': 5, 'MjHalfCheetah-v0': 3 };
for (const id of mjIds) {
  const env = make(id), cf = closedForm[id];
  if (!cf) { const r = rng(1); let steps = 0, ok = true; // 無老師：隨機策略應該很快摔倒 (done)，回報為有限值
    for (let s = 0; s < 5; s++) { const x = rollout(env, () => Array.from({ length: env.nA }, () => (r() * 2 - 1) * env.range[1]), s); steps += x.n; ok &&= x.done && isFinite(x.R) && x.n < 300; }
    if (!ok) fail++; console.log(`${ok ? 'PASS' : 'FAIL'} 環境 ${id.padEnd(28)} 隨機策略平均撐 ${(steps / 5).toFixed(0)} 步後終止（無 closed-form 老師，僅環境基準）`); continue; }
  const { policy, pass } = cf, pi = policy(env), sp = cf.studentPass ?? pass; let sum = 0, min = Infinity, maxD = 0;
  for (let s = 0; s < N[id]; s++) { let last; const { R } = rollout(env, pi, s, x => last = x); sum += R; min = Math.min(min, R); if (last.dist != null) maxD = Math.max(maxD, last.dist); }
  const mean = sum / N[id], ok = mean >= pass && min >= (id.includes('Pendulum') ? pass : -Infinity) && maxD <= .02; if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} 老師 ${id.padEnd(28)} 平均回報=${mean.toFixed(2)} 最差=${min.toFixed(2)}${maxD ? ' 最終距離最大=' + maxD.toFixed(4) : ''} (需 ≥ ${pass})`);
  const st = createStudent(tf, make(id)), t0 = Date.now(); let r;
  for (let i = 0; i < 12; i++) { r = await st.iterate(); if (r.studentReturn >= sp) break; }
  const ok2 = r.studentReturn >= sp; if (!ok2) fail++;
  console.log(`${ok2 ? 'PASS' : 'FAIL'} 學生 ${id.padEnd(28)} 輪數=${r.iter} 動作誤差(正規化)=${r.mae.toFixed(3)} 學生回報=${r.studentReturn.toFixed(2)} (需 ≥ ${sp}) ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
process.exit(fail ? 1 : 0);
