"""
通用 PyTorch PPO：支援 Gymnasium 內建環境、Atari、MuJoCo、Box2D 等。
--env 為必填，存檔名依環境自動命名 (例如 humanoid_ppo.pt、ale_pong_ppo.pt)。

安裝:
    pip install torch gymnasium numpy
    pip install "gymnasium[mujoco]"      # MuJoCo (Humanoid、Walker2d ...)
    pip install "gymnasium[atari]"       # Atari (ALE/Pong-v5 ...)
    pip install "gymnasium[box2d]"       # Box2D (LunarLander、CarRacing ...)

訓練:
    python gym_ppo.py --env CartPole-v1 --minutes 3
    python gym_ppo.py --env Humanoid-v5 --minutes 60
    python gym_ppo.py --env ALE/Pong-v5 --minutes 120

接續訓練:
    python gym_ppo.py --env Humanoid-v5 --resume --minutes 30

播放 (開視窗；MuJoCo 鏡頭會跟隨角色):
    python gym_ppo.py --env Humanoid-v5 --play

自動判斷:
    動作空間  Discrete -> Categorical；Box -> 高斯分布
    觀測空間  影像 (3 維 Box，含 Atari) -> CNN；向量 -> MLP + 觀測正規化；Discrete -> one-hot
    Atari     自動套用 灰階/84x84/跳 4 幀/疊 4 幀/reward 取正負號，並預設同時跑 8 個環境

不支援: Dict / Tuple / MultiDiscrete / MultiBinary 等複合空間，以及多智能體環境。
"""
import argparse
import os
import re
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

try:  # Atari 需要 ale-py；沒裝就略過，其他環境不受影響
    import ale_py
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
except ImportError:
    pass


# --------------------------------------------------------------------------- 環境相關
def env_info(env_id):
    """回傳 (是否為 Atari 影像環境, 是否為 MuJoCo)。"""
    try:
        spec = gym.spec(env_id)
    except Exception as e:
        raise SystemExit(
            f"找不到環境 {env_id}: {e}\n"
            "提示: Atari 需要 pip install \"gymnasium[atari]\"；MuJoCo 需要 \"gymnasium[mujoco]\"；"
            "Box2D 需要 \"gymnasium[box2d]\""
        )
    entry = str(spec.entry_point)
    atari = "ale_py" in entry
    ram = atari and "ram" in env_id.lower()   # RAM 版觀測是向量，不是影像
    return atari and not ram, "mujoco" in entry


def make_env(env_id, render=False):
    atari_img, mujoco = env_info(env_id)
    kw = {"render_mode": "human"} if render else {}
    if atari_img:
        env = gym.make(env_id, frameskip=1, **kw)  # 跳幀交給 AtariPreprocessing 處理
        env = gym.wrappers.AtariPreprocessing(
            env, noop_max=30, frame_skip=4, screen_size=84,
            terminal_on_life_loss=False, grayscale_obs=True)
        stack = getattr(gym.wrappers, "FrameStackObservation", None)
        return stack(env, 4) if stack else gym.wrappers.FrameStack(env, 4)
    if render and mujoco:
        try:
            return gym.make(env_id, default_camera_config={
                "distance": 5.0, "elevation": -10.0, "lookat": np.array([0.0, 0.0, 1.0])}, **kw)
        except TypeError:   # 舊版環境 (例如 v4) 不接受這個參數
            pass
    return gym.make(env_id, **kw)


def describe(env):
    """把 observation / action space 整理成簡單的 dict。"""
    o, a = env.observation_space, env.action_space
    if isinstance(o, gym.spaces.Box):
        if len(o.shape) == 3:
            if o.dtype != np.uint8:
                raise SystemExit("不支援非 uint8 的影像觀測")
            hwc = o.shape[-1] in (1, 3) and o.shape[0] > 3
            shape = (o.shape[2], o.shape[0], o.shape[1]) if hwc else tuple(o.shape)
            obs = dict(kind="image", shape=shape, hwc=hwc)
        else:
            obs = dict(kind="vector", shape=(int(np.prod(o.shape)),), hwc=False)
    elif isinstance(o, gym.spaces.Discrete):
        obs = dict(kind="onehot", shape=(int(o.n),), hwc=False, start=int(getattr(o, "start", 0)))
    else:
        raise SystemExit(f"不支援的觀測空間: {type(o).__name__} (只支援 Box / Discrete)")

    if isinstance(a, gym.spaces.Discrete):
        act = dict(discrete=True, dim=int(a.n), start=int(getattr(a, "start", 0)))
    elif isinstance(a, gym.spaces.Box):
        act = dict(discrete=False, dim=int(np.prod(a.shape)), shape=a.shape,
                   low=a.low.reshape(-1), high=a.high.reshape(-1))
    else:
        raise SystemExit(f"不支援的動作空間: {type(a).__name__} (只支援 Discrete / Box)")
    return obs, act


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


class ObsProcessor:
    """把環境原始觀測轉成網路輸入: 影像 -> CHW uint8；向量 -> 正規化 float32；Discrete -> one-hot。"""

    def __init__(self, obs):
        self.obs = obs
        self.norm = RunningNorm(obs["shape"]) if obs["kind"] == "vector" else None

    def encode(self, raw):
        k = self.obs["kind"]
        if k == "image":
            x = np.asarray(raw)
            if self.obs["hwc"]:
                x = x.transpose(2, 0, 1)
            return np.ascontiguousarray(x, dtype=np.uint8)
        if k == "vector":
            return np.asarray(raw, dtype=np.float64).reshape(-1)
        v = np.zeros(self.obs["shape"][0], dtype=np.float32)
        v[int(raw) - self.obs["start"]] = 1.0
        return v

    def update(self, raw):
        if self.norm is not None:
            self.norm.update(self.encode(raw))

    def __call__(self, raw):
        x = self.encode(raw)
        return self.norm(x) if self.norm is not None else x


# --------------------------------------------------------------------------- 網路
def layer(m, gain=np.sqrt(2)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class ActorCritic(nn.Module):
    def __init__(self, obs, act, hidden=64):
        super().__init__()
        self.discrete = act["discrete"]
        self.image = obs["kind"] == "image"
        n_out = act["dim"]
        if self.image:  # Nature-DQN 風格 CNN，actor/critic 共用特徵
            c, h, w = obs["shape"]
            convs = nn.Sequential(
                layer(nn.Conv2d(c, 32, 8, 4)), nn.ReLU(),
                layer(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
                layer(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                flat = convs(torch.zeros(1, c, h, w)).shape[1]
            self.enc = nn.Sequential(convs, layer(nn.Linear(flat, 512)), nn.ReLU())
            self.actor = layer(nn.Linear(512, n_out), 0.01)
            self.critic = layer(nn.Linear(512, 1), 1.0)
        else:
            d = obs["shape"][0]
            self.actor = nn.Sequential(
                layer(nn.Linear(d, hidden)), nn.Tanh(),
                layer(nn.Linear(hidden, hidden)), nn.Tanh(),
                layer(nn.Linear(hidden, n_out), 0.01))
            self.critic = nn.Sequential(
                layer(nn.Linear(d, hidden)), nn.Tanh(),
                layer(nn.Linear(hidden, hidden)), nn.Tanh(),
                layer(nn.Linear(hidden, 1), 1.0))
        if not self.discrete:
            self.log_std = nn.Parameter(torch.full((n_out,), -0.5))

    def forward(self, x):
        if self.image:
            h = self.enc(x.float() / 255.0)
            out, value = self.actor(h), self.critic(h)
        else:
            out, value = self.actor(x), self.critic(x)
        if self.discrete:
            dist = torch.distributions.Categorical(logits=out)
        else:
            dist = torch.distributions.Normal(out, self.log_std.exp())
        return dist, value.squeeze(-1)


def logp_ent(dist, a):
    lp, ent = dist.log_prob(a), dist.entropy()
    if lp.dim() > 1:  # 連續動作: 各維度加總
        lp, ent = lp.sum(-1), ent.sum(-1)
    return lp, ent


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- 訓練
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    env0 = make_env(args.env)
    obs, act = describe(env0)
    is_img = obs["kind"] == "image"

    # 依環境類型自動選超參數 (都可以用命令列覆蓋)
    N = args.num_envs or (8 if is_img else 1)
    T = args.rollout or (128 if is_img else 2048)            # 每個環境每輪收集的步數
    epochs = args.epochs or (4 if is_img else 10)
    minibatch = args.minibatch or (256 if is_img else 64)
    lr0 = args.lr or (2.5e-4 if is_img else 3e-4)
    clip = 0.1 if is_img else 0.2
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.01 if act["discrete"] else 0.0)
    gamma, lam = 0.99, 0.95

    envs = [env0] + [make_env(args.env) for _ in range(N - 1)]
    proc = ObsProcessor(obs)
    ret_norm = RunningNorm(())  # 非影像環境用折現回報的標準差縮放 reward

    net = ActorCritic(obs, act).to(device)       # 在 GPU(MPS/CUDA) 上做 PPO 更新
    cpu_net = ActorCritic(obs, act)              # CPU 副本，專門用來跟環境互動
    cpu_net.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr0, eps=1e-5)

    print(f"環境: {args.env} | 觀測: {obs['kind']}{obs['shape']} | "
          f"動作: {'Discrete' if act['discrete'] else 'Box'}({act['dim']}) | 同時環境數: {N}")
    print(f"存檔: {args.save} | 更新網路使用裝置: {device} (環境取樣固定用 CPU)")

    start_steps = 0
    if args.resume:
        ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
        if ckpt.get("env", args.env) != args.env:
            raise SystemExit(f"存檔 {args.save} 是 {ckpt['env']} 訓練的，和 --env {args.env} 不符")
        try:
            net.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise SystemExit(f"存檔的網路結構和目前不符 (可能是舊版程式存的): {e}")
        cpu_net.load_state_dict(ckpt["model"])
        if proc.norm is not None and ckpt.get("mean") is not None:
            proc.norm.mean, proc.norm.var = ckpt["mean"], ckpt["var"]
            proc.norm.count = ckpt.get("obs_count", 1e5)  # 舊版存檔沒有 count，用保守值
        if ckpt.get("ret_norm") is not None:
            ret_norm.mean, ret_norm.var, ret_norm.count = ckpt["ret_norm"]
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_steps = ckpt.get("total_steps", 0)
        print(f"已從 {args.save} 載入，接續訓練 (先前累計 {start_steps} 步)")
    elif os.path.exists(args.save):
        print(f"注意: {args.save} 已存在，訓練結束後會被覆蓋 (要接續訓練請加 --resume)")

    total_steps = start_steps

    def save_ckpt():
        n = proc.norm
        torch.save({"model": cpu_net.state_dict(), "opt": opt.state_dict(),
                    "mean": n.mean if n else None, "var": n.var if n else None,
                    "obs_count": n.count if n else None,
                    "ret_norm": (ret_norm.mean, ret_norm.var, ret_norm.count),
                    "total_steps": total_steps, "env": args.env}, args.save)

    obs_dtype = np.uint8 if is_img else np.float32
    act_dtype = np.int64 if act["discrete"] else np.float32
    act_shape = () if act["discrete"] else (act["dim"],)

    ep_ret = np.zeros(N)
    running_ret = np.zeros(N)
    ep_returns = []
    time_limit = args.minutes * 60
    t0 = last_log = last_save = time.time()
    update = 0

    raws = [e.reset(seed=args.seed + i)[0] for i, e in enumerate(envs)]
    for r in raws:
        proc.update(r)
    cur = np.stack([proc(r) for r in raws])

    while time.time() - t0 < time_limit:
        frac = min((time.time() - t0) / time_limit, 1.0)
        for g in opt.param_groups:  # 學習率隨時間線性衰減
            g["lr"] = lr0 * (1 - 0.9 * frac)

        b_obs = np.zeros((T, N) + tuple(obs["shape"]), obs_dtype)
        b_act = np.zeros((T, N) + act_shape, act_dtype)
        b_logp = np.zeros((T, N), np.float32)
        b_rew = np.zeros((T, N), np.float32)
        b_val = np.zeros((T + 1, N), np.float32)
        b_done = np.zeros((T, N), np.float32)

        for t in range(T):
            with torch.no_grad():
                d, v = cpu_net(torch.as_tensor(cur))
                a = d.sample()
                lp, _ = logp_ent(d, a)
            a_np = a.numpy()
            b_obs[t], b_act[t] = cur, a_np
            b_logp[t], b_val[t] = lp.numpy(), v.numpy()

            for i, env in enumerate(envs):
                if act["discrete"]:
                    env_act = int(a_np[i]) + act["start"]
                else:
                    env_act = np.clip(a_np[i], act["low"], act["high"]).reshape(act["shape"])
                raw, r, term, trunc, _ = env.step(env_act)
                ep_ret[i] += r
                total_steps += 1

                if is_img:
                    rs = float(np.sign(r))                 # Atari 標準作法: reward 取正負號
                else:
                    running_ret[i] = running_ret[i] * gamma + r
                    ret_norm.update(running_ret[i])
                    rs = float(np.clip(r / np.sqrt(ret_norm.var + 1e-8), -10, 10))

                if term or trunc:
                    if trunc and not term:  # 時間到而非失敗: 用價值函數補上未來回報
                        with torch.no_grad():
                            fv = cpu_net(torch.as_tensor(proc(raw)).unsqueeze(0))[1].item()
                        rs += gamma * fv
                    b_done[t, i] = 1.0
                    ep_returns.append(ep_ret[i])
                    ep_ret[i], running_ret[i] = 0.0, 0.0
                    raw, _ = env.reset()
                b_rew[t, i] = rs
                proc.update(raw)
                cur[i] = proc(raw)

        with torch.no_grad():
            b_val[T] = cpu_net(torch.as_tensor(cur))[1].numpy()

        # GAE 優勢估計 (所有環境一起算)
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + gamma * b_val[t + 1] * nonterm - b_val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + b_val[:T]

        B = T * N
        t_obs = torch.as_tensor(b_obs.reshape((B,) + tuple(obs["shape"])), device=device)
        t_act = torch.as_tensor(b_act.reshape((B,) + act_shape), device=device)
        t_logp = torch.as_tensor(b_logp.reshape(B), device=device)
        t_adv = torch.as_tensor(adv.reshape(B), device=device)
        t_ret = torch.as_tensor(ret.reshape(B), device=device)

        # PPO 更新
        for _ in range(epochs):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, minibatch):
                idx = perm[i:i + minibatch]
                d, v = net(t_obs[idx])
                new_logp, ent = logp_ent(d, t_act[idx])
                ratio = (new_logp - t_logp[idx]).exp()
                a_mb = t_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg_loss = -torch.min(ratio * a_mb, ratio.clamp(1 - clip, 1 + clip) * a_mb).mean()
                v_loss = 0.5 * (v - t_ret[idx]).pow(2).mean()
                loss = pg_loss + 0.5 * v_loss - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        cpu_net.load_state_dict(net.state_dict())  # 把更新後的權重同步回 CPU 副本
        update += 1
        now = time.time()
        if now - last_save > 60:                   # 每分鐘自動存檔，中途中斷也不怕
            save_ckpt()
            last_save = now
        if now - last_log > 10:
            recent = f"{np.mean(ep_returns[-10:]):9.1f}" if ep_returns else "      (無)"
            print(f"update {update:5d} | {now - t0:6.0f}s | 累計步數 {total_steps:9d} | "
                  f"近 10 回合平均 reward {recent}", flush=True)
            last_log = now

    save_ckpt()
    print(f"訓練完成，已儲存到 {args.save}")


# --------------------------------------------------------------------------- 播放
def follow_camera(env):
    """讓鏡頭跟著軀幹移動 (僅 MuJoCo 有效)，角色就不會跑出畫面。"""
    try:
        viewer = env.unwrapped.mujoco_renderer.viewer
        if viewer is None:
            return
        pos = env.unwrapped.data.body("torso").xpos
        viewer.cam.lookat[0] = pos[0]
        viewer.cam.lookat[1] = pos[1]
    except Exception:
        pass  # 取不到視窗或環境沒有 torso 時，退回固定鏡頭，不影響播放


def play(args):
    ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
    if ckpt.get("env", args.env) != args.env:
        raise SystemExit(f"存檔 {args.save} 是 {ckpt['env']} 訓練的，和 --env {args.env} 不符")
    env = make_env(args.env, render=True)
    obs, act = describe(env)
    net = ActorCritic(obs, act)
    net.load_state_dict(ckpt["model"])
    net.eval()
    proc = ObsProcessor(obs)
    if proc.norm is not None:
        proc.norm.mean, proc.norm.var = ckpt["mean"], ckpt["var"]

    for ep in range(args.episodes):
        raw, _ = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            with torch.no_grad():
                d, _ = net(torch.as_tensor(proc(raw)).unsqueeze(0))
                if act["discrete"]:
                    a = int(d.sample().item()) + act["start"]   # 離散動作用取樣，避免 Atari 卡住不動
                else:
                    a = np.clip(d.mean.squeeze(0).numpy(), act["low"], act["high"]).reshape(act["shape"])
            raw, r, term, trunc, _ = env.step(a)
            follow_camera(env)
            total += r
            steps += 1
            done = term or trunc
        print(f"回合 {ep + 1}: reward {total:.1f}, 存活 {steps} 步")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="必填，例如 Humanoid-v5、CartPole-v1、ALE/Pong-v5")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--num-envs", type=int, default=None, help="同時環境數 (預設: Atari/影像 8，其他 1)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--rollout", type=int, default=None, help="每個環境每輪收集步數")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--minibatch", type=int, default=None)
    p.add_argument("--ent-coef", type=float, default=None, help="熵獎勵 (預設: 離散動作 0.01，連續 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto / mps / cuda / cpu (auto 優先順序: mps > cuda > cpu)")
    p.add_argument("--save", default=None, help="存檔路徑 (預設依環境命名，例如 humanoid_ppo.pt)")
    p.add_argument("--resume", action="store_true", help="從 --save 指定的存檔接續訓練")
    p.add_argument("--play", action="store_true", help="載入模型並顯示動畫")
    p.add_argument("--episodes", type=int, default=5, help="--play 時播放的回合數")
    args = p.parse_args()
    if args.save is None:  # Humanoid-v5 -> humanoid_ppo.pt, ALE/Pong-v5 -> ale_pong_ppo.pt
        name = re.sub(r"[^A-Za-z0-9]+", "_", args.env.rsplit("-v", 1)[0]).strip("_").lower()
        args.save = f"{name}_ppo.pt"
    play(args) if args.play else train(args)