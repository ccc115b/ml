import { list, make, rollout } from '../src/core.js';
import '../src/envs.js';
import { closedForm } from '../src/closedForm.js';
const N = 200; let fail = 0;
for (const id of list()) {
  const env = make(id), { policy, pass } = closedForm[id], pi = policy(env);
  let sum = 0, min = Infinity; for (let s = 0; s < N; s++) { const { R } = rollout(env, pi, s); sum += R; min = Math.min(min, R); }
  const strict = id === 'CartPole-v1' || id === 'MountainCar-v0' || id === 'Snake-v0' || id === 'Braitenberg-v0';
  const mean = sum / N, ok = mean >= pass && (!strict || min >= pass);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${id.padEnd(22)} mean=${mean.toFixed(3)} min=${min} (需 ≥ ${pass})`); if (!ok) fail++;
}
process.exit(fail ? 1 : 0);
