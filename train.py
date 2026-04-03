"""
train.py — Training loop for CartPole DQN experiments.
Uses vectorized environments for speed (as suggested by the assignment).

Supports all 4 configurations via CLI flags:
  --use_target_network    enables Target Network      (task 2.3)
  --use_replay_buffer     enables Experience Replay   (task 2.3)
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym

from agent import DQNAgent


def _try_import_replay_buffer():
    try:
        from replay_buffer import ReplayBuffer
        return ReplayBuffer
    except ImportError:
        return None


def train(
    total_steps: int = 1_000_000,
    seed: int = 0,
    hidden_sizes=(64, 64),
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 100_000,
    update_every: int = 4,
    use_target_network: bool = False,
    use_replay_buffer: bool = False,
    target_update_freq: int = 1000,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    min_buffer_size: int = 1000,
    log_interval: int = 1000,
    run_name: str = "naive",
    results_dir: str = "results",
    num_envs: int = 8,
) -> dict:

    np.random.seed(seed)
    torch.manual_seed(seed)

    envs = gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync")
    envs.action_space.seed(seed)

    state_dim  = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=list(hidden_sizes),
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
    )

    target_net = None
    if use_target_network:
        from agent import QNetwork
        target_net = QNetwork(state_dim, action_dim, list(hidden_sizes)).to(agent.device)
        target_net.load_state_dict(agent.q_net.state_dict())
        target_net.eval()

    replay_buffer = None
    if use_replay_buffer:
        ReplayBuffer = _try_import_replay_buffer()
        if ReplayBuffer is None:
            raise ImportError("replay_buffer.py not found.")
        replay_buffer = ReplayBuffer(capacity=buffer_size)

    log_steps, log_returns, log_losses, log_epsilons = [], [], [], []
    states, _ = envs.reset(seed=seed)
    episode_returns = np.zeros(num_envs)
    completed_returns = []
    losses_window = []
    global_step = 0

    print(f"[{run_name}] seed={seed} | steps={total_steps:,} | "
          f"TN={use_target_network} | ER={use_replay_buffer} | "
          f"lr={lr} | update_every={update_every} | hidden={hidden_sizes} | "
          f"num_envs={num_envs}")

    while global_step < total_steps:
        actions = np.array([agent.select_action(states[i]) for i in range(num_envs)])
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        dones = terminated | truncated
        episode_returns += rewards

        for i in range(num_envs):
            if use_replay_buffer and replay_buffer is not None:
                replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            else:
                if global_step % update_every == 0:
                    loss = agent.update(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    losses_window.append(loss)
                else:
                    agent.increment_step()
            if dones[i]:
                completed_returns.append(episode_returns[i])
                episode_returns[i] = 0.0

        if use_replay_buffer and replay_buffer is not None:
            if global_step % update_every == 0 and len(replay_buffer) >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                loss = _update_from_batch(agent, batch, target_net, gamma)
                losses_window.append(loss)
            else:
                agent.step_count += 1

        if use_target_network and target_net is not None:
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(agent.q_net.state_dict())

        states = next_states
        global_step += num_envs

        if global_step % log_interval == 0:
            avg_return = np.mean(completed_returns[-100:]) if completed_returns else 0.0
            avg_loss   = np.mean(losses_window) if losses_window else 0.0
            losses_window = []
            log_steps.append(global_step)
            log_returns.append(avg_return)
            log_losses.append(avg_loss)
            log_epsilons.append(agent.epsilon)
            if global_step % (log_interval * 10) == 0:
                print(f"  step={global_step:>8,} | avg_return={avg_return:6.1f} | "
                      f"eps={agent.epsilon:.3f} | loss={avg_loss:.4f}")

    envs.close()
    os.makedirs(f"{results_dir}/{run_name}", exist_ok=True)
    save_path = f"{results_dir}/{run_name}/seed_{seed}.npz"
    np.savez(save_path,
             steps=np.array(log_steps), returns=np.array(log_returns),
             losses=np.array(log_losses), epsilons=np.array(log_epsilons))
    print(f"  Saved -> {save_path}")
    return {"steps": np.array(log_steps), "returns": np.array(log_returns),
            "losses": np.array(log_losses), "epsilons": np.array(log_epsilons)}


def _update_from_batch(agent, batch, target_net, gamma):
    states, actions, rewards, next_states, dones = batch
    s  = torch.FloatTensor(states).to(agent.device)
    a  = torch.LongTensor(actions).to(agent.device)
    r  = torch.FloatTensor(rewards).to(agent.device)
    ns = torch.FloatTensor(next_states).to(agent.device)
    d  = torch.FloatTensor(dones).to(agent.device)
    q_current = agent.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_net_for_target = target_net if target_net is not None else agent.q_net
        q_next = q_net_for_target(ns).max(dim=1).values
        target = (r + gamma * q_next * (1.0 - d)).clamp(0.0, 500.0)
    loss = agent.loss_fn(q_current, target)
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), max_norm=1.0)
    agent.optimizer.step()
    agent.step_count += 1
    return loss.item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",               type=int,   default=1_000_000)
    p.add_argument("--seed",                type=int,   default=0)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--update_every",        type=int,   default=4)
    p.add_argument("--hidden_sizes",        type=int,   nargs="+", default=[64, 64])
    p.add_argument("--epsilon_start",       type=float, default=1.0)
    p.add_argument("--epsilon_end",         type=float, default=0.05)
    p.add_argument("--epsilon_decay",       type=int,   default=100_000)
    p.add_argument("--num_envs",            type=int,   default=8)
    p.add_argument("--use_target_network",  action="store_true")
    p.add_argument("--use_replay_buffer",   action="store_true")
    p.add_argument("--target_update_freq",  type=int,   default=1000)
    p.add_argument("--buffer_size",         type=int,   default=50_000)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--min_buffer_size",     type=int,   default=1000)
    p.add_argument("--run_name",            type=str,   default="naive")
    p.add_argument("--results_dir",         type=str,   default="results")
    p.add_argument("--log_interval",        type=int,   default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_steps=args.steps, seed=args.seed,
        hidden_sizes=tuple(args.hidden_sizes), lr=args.lr, gamma=0.99,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay, update_every=args.update_every,
        use_target_network=args.use_target_network,
        use_replay_buffer=args.use_replay_buffer,
        target_update_freq=args.target_update_freq,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        min_buffer_size=args.min_buffer_size, log_interval=args.log_interval,
        run_name=args.run_name, results_dir=args.results_dir, num_envs=args.num_envs,
    )