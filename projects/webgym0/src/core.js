// webgym0 核心：seeded RNG、環境註冊、rollout
export const rng = s => () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const reg = {};
export const register = (id, f) => reg[id] = f;
export const make = id => reg[id]();
export const list = () => Object.keys(reg);
// Env 介面: reset(seed)->obs ; step(a)->{obs,reward,done,truncated} ; render(ctx,w,h)
export function rollout(env, policy, seed = 0, onStep) {
  let o = env.reset(seed), R = 0, n = 0;
  for (;;) { const r = env.step(policy(o)); R += r.reward; n++; onStep && onStep(r); o = r.obs;
    if (r.done || r.truncated) return { R, n, done: r.done }; }
}
