import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--use_target_network", action="store_true")
    p.add_argument("--use_replay_buffer",  action="store_true")
    p.add_argument("--run_name",  type=str,   default="run")
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--total_steps",        type=int,   default=1_000_000)
    p.add_argument("--num_envs",           type=int,   default=8)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--gamma",              type=float, default=0.99)
    p.add_argument("--batch_size",         type=int,   default=128)
    p.add_argument("--grad_clip",          type=float, default=10.0)
    p.add_argument("--epsilon_start",      type=float, default=1.0)
    p.add_argument("--epsilon_end",        type=float, default=0.01)
    p.add_argument("--epsilon_decay",      type=int,   default=200_000)
    p.add_argument("--update_every",       type=int,   default=4,
                   help="env steps between gradient updates (ER configs only)")
    p.add_argument("--target_update_freq", type=int,   default=5_000,
                   help="env steps between hard target-net syncs")
    p.add_argument("--buffer_size",        type=int,   default=100_000)
    p.add_argument("--warmup_steps",       type=int,   default=1_000,
                   help="minimum buffer size before first ER update")
    p.add_argument("--hidden_size",        type=int,   default=128)
    p.add_argument("--log_interval",       type=int,   default=500,
                   help="log mean episode return every N env steps")
    p.add_argument("--outdir",             type=str,   default="results")
    p.add_argument("--device",             type=str,   default="auto")

    return p.parse_args()

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
            nn.Linear(hidden,  n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, act, rew, next_obs, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs),  dtype=torch.float32, device=device),
            torch.tensor(list(act),      dtype=torch.long,    device=device),
            torch.tensor(list(rew),      dtype=torch.float32, device=device),
            torch.tensor(np.array(nobs), dtype=torch.float32, device=device),
            torch.tensor(list(done),     dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buf)

def get_epsilon(step: int, args) -> float:
    frac = min(1.0, step / args.epsilon_decay)
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

def gradient_update(q_net, target_net, optimizer, loss_fn,
                    obs_b, act_b, rew_b, nobs_b, done_b,
                    gamma: float, grad_clip: float, use_tn: bool):
    with torch.no_grad():
        bootstrap_net = target_net if use_tn else q_net
        next_q  = bootstrap_net(nobs_b).max(dim=1).values
        targets = rew_b + gamma * next_q * (1.0 - done_b)
        targets = targets.clamp(-500.0, 500.0)

    current_q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
    loss = loss_fn(current_q, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[{args.run_name} | seed={args.seed}] "
          f"use_tn={args.use_target_network}, use_er={args.use_replay_buffer} | "
          f"device={device}")

    out_dir = os.path.join(args.outdir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    envs = gym.make_vec("CartPole-v1", num_envs=args.num_envs,
                        vectorization_mode="sync")
    obs_dim   = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    q_net      = QNetwork(obs_dim, n_actions, args.hidden_size).to(device)
    target_net = QNetwork(obs_dim, n_actions, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    loss_fn   = nn.HuberLoss(delta=1.0)
    buffer = ReplayBuffer(args.buffer_size) if args.use_replay_buffer else None
    online_transitions = []

    steps_log      = []
    returns_log    = []
    window_returns = []
    next_log_step  = args.log_interval

    obs, _ = envs.reset(seed=args.seed)
    episode_rewards = np.zeros(args.num_envs)
    global_step = 0

    while global_step < args.total_steps:
        eps = get_epsilon(global_step, args)

        if random.random() < eps:
            actions = np.array([envs.single_action_space.sample()
                                 for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                obs_t   = torch.tensor(obs, dtype=torch.float32, device=device)
                actions = q_net(obs_t).argmax(dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        dones = terminated | truncated

        episode_rewards += rewards

        for i in range(args.num_envs):
            d = float(terminated[i])

            if args.use_replay_buffer:
                buffer.push(obs[i], actions[i], rewards[i], next_obs[i], d)
            else:
                online_transitions.append(
                    (obs[i], actions[i], rewards[i], next_obs[i], d)
                )

            if dones[i]:
                window_returns.append(episode_rewards[i])
                episode_rewards[i] = 0.0

        obs = next_obs
        global_step += args.num_envs

        if args.use_replay_buffer:
            if (global_step % args.update_every == 0 and
                    len(buffer) >= max(args.batch_size, args.warmup_steps)):
                obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(
                    args.batch_size, device)
                gradient_update(q_net, target_net, optimizer, loss_fn,
                                obs_b, act_b, rew_b, nobs_b, done_b,
                                args.gamma, args.grad_clip,
                                args.use_target_network)
        else:
            if len(online_transitions) >= args.batch_size:
                batch  = random.sample(online_transitions, args.batch_size)
                obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)
                obs_b  = torch.tensor(np.array(obs_b),  dtype=torch.float32, device=device)
                act_b  = torch.tensor(list(act_b),      dtype=torch.long,    device=device)
                rew_b  = torch.tensor(list(rew_b),      dtype=torch.float32, device=device)
                nobs_b = torch.tensor(np.array(nobs_b), dtype=torch.float32, device=device)
                done_b = torch.tensor(list(done_b),     dtype=torch.float32, device=device)
                gradient_update(q_net, target_net, optimizer, loss_fn,
                                obs_b, act_b, rew_b, nobs_b, done_b,
                                args.gamma, args.grad_clip,
                                args.use_target_network)
                online_transitions.clear()

        if args.use_target_network and (global_step % args.target_update_freq == 0):
            target_net.load_state_dict(q_net.state_dict())

        if global_step >= next_log_step:
            mean_ret = np.mean(window_returns) if window_returns else 0.0
            steps_log.append(global_step)
            returns_log.append(mean_ret)
            window_returns.clear()
            next_log_step += args.log_interval

            if len(steps_log) % 20 == 0:
                print(f"  step {global_step:>9d}/{args.total_steps} | "
                      f"ε={eps:.3f} | mean_return={mean_ret:.1f}")

    envs.close()
    out_path = os.path.join(out_dir, f"seed_{args.seed}.npz")
    np.savez(out_path, steps=np.array(steps_log), returns=np.array(returns_log))
    print(f"[done] saved → {out_path}")


if __name__ == "__main__":
    main()