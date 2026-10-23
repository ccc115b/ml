// 需要 npm i -D @mujoco/mujoco @tensorflow/tfjs。驗證：(1) 解析控制器通過 (2) 學生模仿老師後也通過
import loadMujoco from '@mujoco/mujoco'; import * as tf from '@tensorflow/tfjs';
import { make, rollout } from '../src/core.js'; import { initMujoco, mjIds } from '../src/mj.js';
import { closedForm } from '../src/closedForm.js'; import { createStudent } from '../src/imitate.js';
await initMujoco(loadMujoco); let fail = 0;
const N = { 'MjReacher-v0': 30, 'MjInvertedDoublePendulum-v0': 10, 'MjSwimmer-v0': 5 };
for (const id of mjIds) {
  const env = make(id), { policy, pass } = closedForm[id], pi = policy(env); let sum = 0, min = Infinity, maxD = 0;
  for (let s = 0; s < N[id]; s++) { let last; const { R } = rollout(env, pi, s, x => last = x); sum += R; min = Math.min(min, R); if (last.dist != null) maxD = Math.max(maxD, last.dist); }
  const mean = sum / N[id], ok = mean >= pass && min >= (id.includes('Pendulum') ? pass : -Infinity) && maxD <= .02; if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} 老師 ${id.padEnd(28)} 平均回報=${mean.toFixed(2)} 最差=${min.toFixed(2)}${maxD ? ' 最終距離最大=' + maxD.toFixed(4) : ''} (需 ≥ ${pass})`);
  const st = createStudent(tf, make(id)), t0 = Date.now(); let r;
  for (let i = 0; i < 12; i++) { r = await st.iterate(); if (r.studentReturn >= pass) break; }
  const ok2 = r.studentReturn >= pass; if (!ok2) fail++;
  console.log(`${ok2 ? 'PASS' : 'FAIL'} 學生 ${id.padEnd(28)} 輪數=${r.iter} 動作誤差(正規化)=${r.mae.toFixed(3)} 學生回報=${r.studentReturn.toFixed(2)} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
process.exit(fail ? 1 : 0);
