// 老師 = closed-form 策略，學生 = TF.js 網路 (DAgger)。學生從未看過獎勵。
import * as tf from '@tensorflow/tfjs';
import { make } from '../src/core.js'; import '../src/envs.js'; import { closedForm } from '../src/closedForm.js'; import { createStudent } from '../src/imitate.js';
let fail = 0;
for (const id of ['MountainCar-v0', 'Braitenberg-v0', 'Snake-v0']) {
  const st = createStudent(tf, make(id)), pass = closedForm[id].pass, t0 = Date.now(); let r;
  for (let i = 0; i < 12; i++) { r = await st.iterate(); if (r.studentReturn >= pass) break; }
  const ok = r.studentReturn >= pass; if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${id.padEnd(16)} 輪數=${r.iter} ${st.cont ? 'MAE=' + r.mae.toFixed(3) : '與老師一致率=' + (r.match * 100).toFixed(1) + '%'} 學生回報=${r.studentReturn.toFixed(1)} (老師門檻 ${pass}) ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
process.exit(fail ? 1 : 0);
