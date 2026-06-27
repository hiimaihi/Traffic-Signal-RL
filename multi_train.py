"""
多路口训练脚本 — 1×2 走廊独立 Q-Learning
==========================================
两路口各自使用独立 Agent, 从单路口预训练模型初始化后微调。

用法:
    python multi_train.py --agent dueling --episodes 200 --pattern low_traffic
    python multi_train.py --agent dqn --episodes 300 --pattern uniform
"""
import sys, os, argparse, json, time
import numpy as np
import torch
from collections import defaultdict

from multi_intersection_env import MultiIntersectionEnv
from agent import create_agent
from pg_agent import create_pg_agent


def load_pretrained(agent, path: str) -> bool:
    """加载预训练权重 (单路口模型 → 多路口初始化)。"""
    if not os.path.isfile(path):
        return False
    try:
        ckpt = torch.load(path, map_location=agent.device, weights_only=False)
        agent.online_net.load_state_dict(ckpt)
        agent.target_net.load_state_dict(ckpt)
        return True
    except Exception as e:
        print(f"  ⚠ 加载失败 {path}: {e}")
        return False


def resolve_pretrained(agent_type: str, pattern: str) -> str:
    """解析预训练模型路径。"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "models")
    candidates = [
        f"{agent_type}_{pattern}.pt",
        f"{agent_type}_uniform.pt",
        f"dueling_uniform.pt",
    ]
    for c in candidates:
        full = os.path.join(models_dir, c)
        if os.path.isfile(full):
            return full
    return ""


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚦 多路口训练: 1×2 走廊 | Agent={args.agent} | Pattern={args.pattern}")
    print(f"   Episodes={args.episodes} | MaxSteps={args.max_steps} | Device={device}")

    # ── 创建环境 ──
    env = MultiIntersectionEnv(
        arrival_rates=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        traffic_pattern=args.pattern,
        state_dim=8,
        max_steps=args.max_steps,
        travel_delay=args.travel_delay,
    )

    # ── 创建两个 Agent (独立学习者) ──
    agents = []
    for i in range(2):
        if args.agent in ('a2c', 'ppo'):
            ag = create_pg_agent(agent_type=args.agent, state_dim=8, lr=args.lr)
        else:
            ag = create_agent(
                agent_type=args.agent, state_dim=8, lr=args.lr,
                epsilon_start=args.eps_start, epsilon_end=args.eps_end,
                epsilon_decay=args.eps_decay,
            )
        agents.append(ag)

    # ── 加载预训练 ──
    pretrained_path = resolve_pretrained(args.agent, args.pattern)
    loaded = [False, False]
    if pretrained_path:
        for i in range(2):
            loaded[i] = load_pretrained(agents[i], pretrained_path)
        print(f"   📦 预训练: {pretrained_path} → I0={loaded[0]}, I1={loaded[1]}")

    # ── 训练循环 ──
    metrics = defaultdict(list)
    best_avg_reward = -float('inf')
    start_time = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_queues = []
        ep_switches = [0, 0]
        step_count = 0

        while not done:
            # 每路口独立决策
            prev_locals = [env.get_local_obs(i) for i in range(2)]
            actions = [agents[i].select_action(prev_locals[i], evaluate=False) for i in range(2)]
            for i in range(2):
                if actions[i] == 1:
                    ep_switches[i] += 1

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            for i in range(2):
                next_local = env.get_local_obs(i)
                agents[i].store_transition(prev_locals[i], actions[i], reward / 2.0,
                                           next_local, done)
                if args.agent not in ('a2c', 'ppo'):
                    agents[i].update()

            obs = next_obs
            ep_reward += reward
            ep_queues.append(info['total_queue'])
            step_count += 1

            if step_count >= args.max_steps:
                break

        avg_q = np.mean(ep_queues) if ep_queues else 0
        total_tp = sum(env.all_total_departed)
        metrics['reward'].append(ep_reward)
        metrics['avg_queue'].append(avg_q)
        metrics['throughput'].append(total_tp)

        # 日志
        if (ep + 1) % max(1, args.episodes // 20) == 0 or ep == 0:
            elapsed = time.time() - start_time
            q_str = f"I0={[q.length for q in env.all_queues[0]]} I1={[q.length for q in env.all_queues[1]]}"
            print(f"  Ep {ep+1:4d}/{args.episodes} | "
                  f"R={ep_reward:7.1f} | AvgQ={avg_q:5.1f} | "
                  f"TP={total_tp:4d} | Sw=[{ep_switches[0]},{ep_switches[1]}] | "
                  f"⏱ {elapsed:.0f}s | {q_str}")

        # 保存最佳模型
        if ep_reward > best_avg_reward:
            best_avg_reward = ep_reward
            if args.save:
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", "models")
                os.makedirs(save_dir, exist_ok=True)
                for i in range(2):
                    path = os.path.join(save_dir, f"multi_{args.agent}_{args.pattern}_i{i}.pt")
                    torch.save(agents[i].online_net.state_dict(), path)
                print(f"  💾 最佳模型已保存 (R={ep_reward:.1f})")

    # ── 报告 ──
    print(f"\n{'='*60}")
    print(f"训练完成! 总时间: {time.time()-start_time:.0f}s")
    print(f"  平均奖励: {np.mean(metrics['reward'][-50:]):.1f} ± {np.std(metrics['reward'][-50:]):.1f}")
    print(f"  平均排队: {np.mean(metrics['avg_queue'][-50:]):.1f}")
    print(f"  平均吞吐: {np.mean(metrics['throughput'][-50:]):.0f}")
    print(f"  最佳奖励: {best_avg_reward:.1f}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Intersection Training")
    parser.add_argument('--agent', default='dueling',
                        choices=['dqn','double','dueling','noisy','boltzmann','per_dqn',
                                 'dueling_per','a2c','ppo'])
    parser.add_argument('--pattern', default='low_traffic',
                        choices=['uniform','peak_hour','tidal','burst','low_traffic'])
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--travel-delay', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eps-start', type=float, default=0.3)
    parser.add_argument('--eps-end', type=float, default=0.02)
    parser.add_argument('--eps-decay', type=int, default=5000)
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--no-save', dest='save', action='store_false')
    args = parser.parse_args()
    train(args)
