"""
PyTorch PPO 訓練 MuJoCo 仿生人/雙足機器人，預設限時 10 分鐘。
--env 為必填 (例如 Walker2d-v5、Humanoid-v5)，存檔名會依環境自動命名，例如 humanoid_ppo.pt。

安裝:
    pip install torch "gymnasium[mujoco]" numpy

訓練:
    python mujoco_ppo.py --env Walker2d-v5
    python mujoco_ppo.py --env Humanoid-v5 --minutes 60

接續訓練 (從上次存檔繼續，不從頭開始):
    python mujoco_ppo.py --env Humanoid-v5 --resume --minutes 30

播放訓練好的模型 (會開視窗顯示動畫，鏡頭跟隨角色):
    python mujoco_ppo.py --env Humanoid-v5 --play
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class RunningNorm:
    """線上計算平均與變異數，用來正規化觀測值。"""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        delta = x - self.mean
        total = self.count + 1
        self.mean = self.mean + delta / total
        self.var = (self.var * self.count + delta * (x - self.mean)) / total
        self.count = total

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"環境: {args.env} | 存檔: {args.save} | 更新網路使用裝置: {device} (環境取樣固定用 CPU)")

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high

    net = ActorCritic(obs_dim, act_dim).to(device)          # 在 GPU(MPS/CUDA) 上做 PPO 更新
    cpu_net = ActorCritic(obs_dim, act_dim)                  # CPU 副本，專門用來跟環境互動
    cpu_net.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    obs_norm = RunningNorm(obs_dim)
    ret_norm = RunningNorm(())  # 用折現回報的標準差縮放 reward
    running_ret = 0.0

    start_steps = 0
    if args.resume:
        ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
        if ckpt.get("env", args.env) != args.env:
            raise SystemExit(f"存檔 {args.save} 是 {ckpt['env']} 訓練的，和 --env {args.env} 不符")
        net.load_state_dict(ckpt["model"])
        cpu_net.load_state_dict(ckpt["model"])
        obs_norm.mean, obs_norm.var = ckpt["mean"], ckpt["var"]
        obs_norm.count = ckpt.get("obs_count", 1e5)   # 舊版存檔沒有 count，用保守值
        if "ret_norm" in ckpt:
            ret_norm.mean, ret_norm.var, ret_norm.count = ckpt["ret_norm"]
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_steps = ckpt.get("total_steps", 0)
        print(f"已從 {args.save} 載入，接續訓練 (先前累計 {start_steps} 步)")

    def save_ckpt():
        torch.save({"model": cpu_net.state_dict(), "opt": opt.state_dict(),
                    "mean": obs_norm.mean, "var": obs_norm.var, "obs_count": obs_norm.count,
                    "ret_norm": (ret_norm.mean, ret_norm.var, ret_norm.count),
                    "total_steps": total_steps, "env": args.env}, args.save)

    T, gamma, lam = args.rollout, 0.99, 0.95
    time_limit = args.minutes * 60
    t0 = time.time()
    total_steps, update = start_steps, 0
    last_save = time.time()
    ep_return, ep_returns = 0.0, []

    raw, _ = env.reset(seed=args.seed)
    obs_norm.update(raw)
    obs = obs_norm(raw)

    while time.time() - t0 < time_limit:
        frac = min((time.time() - t0) / time_limit, 1.0)
        for g in opt.param_groups:  # 學習率隨時間線性衰減
            g["lr"] = args.lr * (1 - 0.9 * frac)

        b_obs = np.zeros((T, obs_dim), np.float32)
        b_act = np.zeros((T, act_dim), np.float32)
        b_logp = np.zeros(T, np.float32)
        b_rew = np.zeros(T, np.float32)
        b_val = np.zeros(T + 1, np.float32)
        b_done = np.zeros(T, np.float32)

        for t in range(T):
            with torch.no_grad():
                o = torch.as_tensor(obs).unsqueeze(0)
                d = cpu_net.dist(o)
                a = d.sample()
                logp = d.log_prob(a).sum(-1)
                v = cpu_net.value(o)
            act = a.squeeze(0).cpu().numpy()
            raw, r, terminated, truncated, _ = env.step(np.clip(act, low, high))
            ep_return += r
            total_steps += 1

            running_ret = running_ret * gamma + r
            ret_norm.update(running_ret)
            r_scaled = float(np.clip(r / np.sqrt(ret_norm.var + 1e-8), -10, 10))

            b_obs[t], b_act[t] = obs, act
            b_logp[t], b_val[t] = logp.item(), v.item()

            if terminated or truncated:
                if truncated and not terminated:  # 時間到而非跌倒: 用價值函數補上未來回報
                    final_obs = obs_norm(raw)
                    with torch.no_grad():
                        fv = cpu_net.value(torch.as_tensor(final_obs).unsqueeze(0)).item()
                    r_scaled += gamma * fv
                b_done[t] = 1.0
                ep_returns.append(ep_return)
                ep_return, running_ret = 0.0, 0.0
                raw, _ = env.reset()
            b_rew[t] = r_scaled
            obs_norm.update(raw)
            obs = obs_norm(raw)

        with torch.no_grad():
            b_val[T] = cpu_net.value(torch.as_tensor(obs).unsqueeze(0)).item()

        # GAE 優勢估計
        adv = np.zeros(T, np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + gamma * b_val[t + 1] * nonterm - b_val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + b_val[:T]

        t_obs = torch.as_tensor(b_obs, device=device)
        t_act = torch.as_tensor(b_act, device=device)
        t_logp = torch.as_tensor(b_logp, device=device)
        t_adv = torch.as_tensor(adv, device=device)
        t_ret = torch.as_tensor(ret, device=device)

        # PPO 更新
        for _ in range(args.epochs):
            perm = torch.randperm(T, device=device)
            for i in range(0, T, args.minibatch):
                idx = perm[i:i + args.minibatch]
                d = net.dist(t_obs[idx])
                new_logp = d.log_prob(t_act[idx]).sum(-1)
                ratio = (new_logp - t_logp[idx]).exp()
                a_mb = t_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg_loss = -torch.min(ratio * a_mb, ratio.clamp(0.8, 1.2) * a_mb).mean()
                v_loss = 0.5 * (net.value(t_obs[idx]) - t_ret[idx]).pow(2).mean()
                loss = pg_loss + 0.5 * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        cpu_net.load_state_dict(net.state_dict())  # 把更新後的權重同步回 CPU 副本
        update += 1
        if time.time() - last_save > 60:           # 每分鐘自動存檔，中途中斷也不怕
            save_ckpt()
            last_save = time.time()
        if update % 5 == 0 and ep_returns:
            print(f"update {update:4d} | {time.time() - t0:5.0f}s | 步數 {total_steps:8d} | "
                  f"近 10 回合平均 reward {np.mean(ep_returns[-10:]):7.1f}", flush=True)

    save_ckpt()
    print(f"訓練完成，已儲存到 {args.save}")


def follow_camera(env):
    """讓鏡頭跟著軀幹移動，角色就不會跑出畫面。"""
    try:
        viewer = env.unwrapped.mujoco_renderer.viewer
        if viewer is None:
            return
        pos = env.unwrapped.data.body("torso").xpos
        viewer.cam.lookat[0] = pos[0]
        viewer.cam.lookat[1] = pos[1]   # Humanoid 會往 y 方向移動，Walker2d 的 y 恆為 0
    except Exception:
        pass  # 取不到視窗或 gymnasium 版本不同時，退回固定鏡頭，不影響播放


def play(args):
    ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
    env = gym.make(
        ckpt.get("env", args.env),
        render_mode="human",
        default_camera_config={"distance": 5.0, "elevation": -10.0, "lookat": np.array([0.0, 0.0, 1.0])},
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    net = ActorCritic(obs_dim, act_dim)
    net.load_state_dict(ckpt["model"])
    net.eval()
    mean, var = ckpt["mean"], ckpt["var"]

    for ep in range(5):
        raw, _ = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            obs = np.clip((raw - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
            with torch.no_grad():
                act = net.actor(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()  # 播放時用平均動作
            raw, r, term, trunc, _ = env.step(np.clip(act, env.action_space.low, env.action_space.high))
            follow_camera(env)
            total += r
            steps += 1
            done = term or trunc
        print(f"回合 {ep + 1}: reward {total:.1f}, 存活 {steps} 步")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="必填，例如 Walker2d-v5 或 Humanoid-v5")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto / mps / cuda / cpu (auto 優先順序: mps > cuda > cpu)")
    p.add_argument("--save", default=None, help="存檔路徑 (預設依環境命名，例如 humanoid_ppo.pt)")
    p.add_argument("--resume", action="store_true", help="從 --save 指定的存檔接續訓練")
    p.add_argument("--play", action="store_true", help="載入模型並顯示動畫")
    args = p.parse_args()
    if args.save is None:  # 存檔名隨環境改變: Walker2d-v5 -> walker2d_ppo.pt, Humanoid-v5 -> humanoid_ppo.pt
        args.save = f"{args.env.split('-')[0].lower()}_ppo.pt"
    play(args) if args.play else train(args)