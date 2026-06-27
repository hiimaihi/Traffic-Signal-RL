"""
2×2 网格路口 — 模型训练脚本
============================
训练 4 个独立 Agent (每个路口一个), 共享同一种算法。
由于每路口只吃 8 维局部观测, 可复用单路口模型架构。

用法:
  python grid_train.py --agent dueling --pattern uniform --episodes 300 --seeds 3
  python grid_train.py --agent dqn --pattern all --episodes 200 --seeds 2
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import json
import numpy as np
import torch

from grid_2x2_env import Grid2x2Env
from agent import create_agent
from pg_agent import create_pg_agent
from experiments import train_agent


def train_grid_agents(
    agent_type: str = "dueling",
    traffic_pattern: str = "uniform",
    episodes: int = 300,
    seeds: int = 3,
    max_steps: int = 500,
    state_dim: int = 8,
    update_freq: int = 4,
) -> dict:
    """
    在 2×2 网格环境中训练 4 个 Agent。

    训练策略: 每个 episode:
      1. 4 个 Agent 各自根据局部观测选动作
      2. 环境执行联合动作
      3. 每个 Agent 独立学习 (store + update)
      4. 奖励共享 (全局奖励), 模拟 CTDE 范式
    """
    is_pg = agent_type in ('a2c', 'ppo')

    all_results = {}
    best_model_path = None
    best_reward = -float('inf')

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "models")
    os.makedirs(models_dir, exist_ok=True)

    t0 = time.time()

    for seed in range(seeds):
        print(f"\n  Seed {seed + 1}/{seeds}", flush=True)

        env = Grid2x2Env(
            arrival_rates=(2.0,) * 8,
            traffic_pattern=traffic_pattern,
            state_dim=state_dim,
            max_steps=max_steps,
        )

        # 4 个 Agent 独立初始化
        agents = []
        for _ in range(4):
            if is_pg:
                ag = create_pg_agent(agent_type=agent_type, state_dim=state_dim)
            else:
                ag = create_agent(agent_type=agent_type, state_dim=state_dim)
            agents.append(ag)

        episode_rewards = np.zeros(episodes)
        episode_queues = np.zeros(episodes)
        episode_waits = np.zeros(episodes)
        episode_throughput = np.zeros(episodes)

        for ep in range(episodes):
            obs, _ = env.reset()
            total_r = 0.0
            total_q = 0.0
            total_w = 0.0
            steps = 0

            for _ in range(max_steps):
                # 收集 pre-step 局部观测
                prev_locals = [env.get_local_obs(i) for i in range(4)]
                actions = [agents[i].select_action(prev_locals[i]) for i in range(4)]

                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                for i in range(4):
                    next_local = env.get_local_obs(i)
                    agents[i].store_transition(prev_locals[i], actions[i], reward,
                                                next_local, done)
                    # 每 N 步更新一次（大幅提速 ~3x）
                    if not is_pg and steps % update_freq == 0:
                        agents[i].update()

                obs = next_obs
                total_r += reward
                total_q += info["total_queue"]
                total_w += info["avg_wait"]
                steps += 1
                if done:
                    break

            if is_pg:
                for ag in agents:
                    ag.update()

            episode_rewards[ep] = total_r
            episode_queues[ep] = total_q / max(steps, 1)
            episode_waits[ep] = total_w / max(steps, 1)
            episode_throughput[ep] = info["total_departed"]

            if ep == 0 or ep % 25 == 0:
                print(f"    Ep {ep:3d}/{episodes} | R={total_r:8.1f} "
                      f"Q={total_q/max(steps,1):5.2f} W={total_w/max(steps,1):5.1f} "
                      f"Dep={info['total_departed']:4d}", flush=True)

            # 保存最佳模型 (按奖励)
            if total_r > best_reward:
                best_reward = total_r
                best_model_path = os.path.join(
                    models_dir, f"grid_{agent_type}_{traffic_pattern}.pt"
                )
                ckpt = {
                    "online_net": agents[0].online_net.state_dict(),
                    "target_net": agents[0].target_net.state_dict(),
                    "optimizer": agents[0].optimizer.state_dict(),
                    "train_steps": ep,
                    "epsilon": getattr(agents[0], 'epsilon', 0),
                    "temperature": getattr(agents[0], 'temperature', 1.0),
                }
                torch.save(ckpt, best_model_path)

        final_r = episode_rewards[-50:].mean()
        final_q = episode_queues[-50:].mean()
        final_w = episode_waits[-50:].mean()

        all_results[f"seed_{seed}"] = {
            "final_reward_mean": float(final_r),
            "final_queue_mean": float(final_q),
            "final_wait_mean": float(final_w),
            "final_throughput": int(episode_throughput[-1]),
        }

    elapsed = time.time() - t0
    result = {
        "agent": agent_type,
        "pattern": traffic_pattern,
        "episodes": episodes,
        "seeds": seeds,
        "max_steps": max_steps,
        "seed_results": all_results,
        "best_model": best_model_path,
        "elapsed_sec": round(elapsed, 1),
    }

    print(f"\n  完成! 耗时 {elapsed:.0f}s, 最佳模型 → {best_model_path}", flush=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2×2 网格路口模型训练")
    parser.add_argument("--agent", type=str, default="dueling",
                        help="算法: dqn, double, dueling, noisy, boltzmann, per_dqn, a2c, ppo")
    parser.add_argument("--pattern", type=str, default="uniform",
                        help="交通模式: uniform, peak_hour, tidal, burst, low_traffic, all")
    parser.add_argument("--episodes", type=int, default=300,
                        help="训练 episode 数")
    parser.add_argument("--seeds", type=int, default=3,
                        help="随机种子数")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="每 episode 最大步数")
    parser.add_argument("--state-dim", type=int, default=8,
                        help="状态维度 (每路口)")
    parser.add_argument("--update-freq", type=int, default=4,
                        help="网络更新频率 (每 N 步更新一次)")
    args = parser.parse_args()

    PATTERNS = ["uniform", "low_traffic", "peak_hour", "tidal", "burst"]

    if args.pattern == "all":
        for p in PATTERNS:
            print(f"\n{'='*60}", flush=True)
            print(f"  2×2 Grid — {args.agent} @ {p}", flush=True)
            print(f"{'='*60}", flush=True)
            train_grid_agents(
                agent_type=args.agent,
                traffic_pattern=p,
                episodes=args.episodes,
                seeds=args.seeds,
                max_steps=args.max_steps,
                state_dim=args.state_dim,
                update_freq=args.update_freq,
            )
    else:
        train_grid_agents(
            agent_type=args.agent,
            traffic_pattern=args.pattern,
            episodes=args.episodes,
            seeds=args.seeds,
            max_steps=args.max_steps,
            state_dim=args.state_dim,
            update_freq=args.update_freq,
        )

    print("\n  ✅ All done!", flush=True)
