"""
命令行入口，调度训练、实验和可视化。

用法: python main.py [--agent AGENT] [--episodes N] [--ablation TYPE] [--dashboard]
"""

import argparse
import os
import sys
import time
import json
import numpy as np

# 项目模块
from traffic_env import make_env, TrafficLightEnv, FixedTimeController, ENV_CONFIG
from agent import create_agent
from network import create_network
from pg_agent import create_pg_agent
from experiments import (
    train_agent, evaluate_fixed_time,
    run_algorithm_comparison, run_network_ablation,
    run_state_ablation, run_reward_ablation,
    run_hyperparameter_sensitivity, run_traffic_robustness,
    generate_latex_table, save_results,
    ExperimentResult, compute_95ci, paired_t_test,
)
from visualization import (
    plot_training_dashboard, plot_q_value_heatmap,
    plot_phase_decision_timeline, plot_radar_chart,
    plot_multi_seed_ribbon, plot_summary_grand_figure,
    plot_bar_comparison, plot_policy_evolution,
    moving_average,
    ALGO_LABELS,
)

OUTPUT_DIR = "results"


# ══════════════════════════════════════════════════════════════════════════════
# 单次训练+评估
# ══════════════════════════════════════════════════════════════════════════════

def run_single_training(
    agent_type: str = "dueling",
    state_dim: int = 8,
    traffic_pattern: str = "uniform",
    reward_type: str = "composite",
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    output_dir: str = OUTPUT_DIR,
    verbose: bool = True,
) -> dict:
    """
    单算法多 seed 训练, 返回所有历史数据。
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    all_rewards, all_queues, all_waits, all_throughputs, all_losses = [], [], [], [], []

    best_agent = None
    best_avg_reward = -float("inf")

    for seed in range(seeds):
        if verbose:
            print(f"\n  Seed {seed + 1}/{seeds} ...")
        env = make_env(
            traffic_pattern=traffic_pattern,
            state_dim=state_dim,
            reward_type=reward_type,
            max_steps=max_steps,
        )
        # PG agents 使用不同工厂
        if agent_type in ("a2c", "ppo"):
            agent = create_pg_agent(agent_type=agent_type, state_dim=state_dim)
        else:
            agent = create_agent(agent_type=agent_type, state_dim=state_dim)

        import torch, random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        r, q, w, tp = train_agent(agent, env, episodes, max_steps,
                                  verbose=(verbose and seeds <= 3))
        all_rewards.append(r)
        all_queues.append(q)
        all_waits.append(w)
        all_throughputs.append(tp)
        all_losses.append(np.array(agent.loss_history) if agent.loss_history else np.array([]))

        # 保留最优 seed 的 agent
        avg_final = np.mean(r[-50:]) if len(r) >= 50 else np.mean(r)
        if avg_final > best_avg_reward:
            best_avg_reward = avg_final
            best_agent = agent

    # 保存最优模型, 供前端驾驶舱加载
    if best_agent is not None:
        model_path = os.path.join(output_dir, "models", f"{agent_type}_{traffic_pattern}.pt")
        best_agent.save(model_path)
        if verbose:
            print(f"\n   Best model (avg reward={best_avg_reward:.1f}) → {model_path}")

    return {
        "agent_type": agent_type,
        "rewards": np.array(all_rewards),
        "queues": np.array(all_queues),
        "waits": np.array(all_waits),
        "throughputs": np.array(all_throughputs),
        "losses": all_losses,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard 模式
# ══════════════════════════════════════════════════════════════════════════════

def run_dashboard(
    agent_type: str = "dueling",
    traffic_pattern: str = "uniform",
    episodes: int = 300,
    seeds: int = 3,
    max_steps: int = 1000,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """训练+全部可视化仪表盘。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Dashboard: {agent_type} | {traffic_pattern} | {episodes}ep × {seeds}seeds")
    print(f"{'='*60}")

    t0 = time.time()
    data = run_single_training(
        agent_type=agent_type, traffic_pattern=traffic_pattern,
        episodes=episodes, seeds=seeds, max_steps=max_steps,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s")

    # ── 图表生成 ──
    print("\n--- Generating Visualizations ---")

    # (a) Dashboard
    mean_r = data["rewards"].mean(axis=0)
    mean_q = data["queues"].mean(axis=0)
    mean_w = data["waits"].mean(axis=0)
    if data["losses"] and len(data["losses"][0]) > 0:
        all_loss = np.concatenate([l for l in data["losses"] if len(l) > 0])
    else:
        all_loss = np.array([0.1])

    plot_training_dashboard(
        mean_r, mean_q, mean_w, all_loss,
        save_path=os.path.join(output_dir, "dashboard.png"),
        title=f"{ALGO_LABELS.get(agent_type, agent_type)} — Training Dashboard",
    )
    print("  ✓ Dashboard")

    # (b) Q-Value Heatmap — 仅 DQN 系列 (PG 跳过)
    is_pg = agent_type in ("a2c", "ppo")
    if not is_pg:
        env = make_env(traffic_pattern=traffic_pattern, max_steps=max_steps)
        agent = create_agent(agent_type=agent_type, state_dim=8)
        # quick warm-up
        obs, _ = env.reset()
        for _ in range(300):
            a = agent.select_action(obs)
            ns, r, t, tr, _ = env.step(a)
            agent.store_transition(obs, a, r, ns, t or tr)
            obs = env.reset()[0] if (t or tr) else ns
            agent.update()
        plot_q_value_heatmap(agent, env,
                             save_path=os.path.join(output_dir, "q_heatmap.png"))
        print("  ✓ Q-Value Heatmap")
    else:
        print("  ⏭ Q-Value Heatmap (skipped for PG agents)")

    # (c) Phase Timeline — 用训练好的 agent (best_agent from run_single_training)
    if is_pg:
        eval_agent = create_pg_agent(agent_type=agent_type, state_dim=8)
        # 尝试加载刚训练保存的模型
        model_path = os.path.join(output_dir, "models", f"{agent_type}_{traffic_pattern}.pt")
        if os.path.isfile(model_path):
            eval_agent.load(model_path)
    else:
        eval_agent = agent
    env2 = make_env(traffic_pattern=traffic_pattern, max_steps=300)
    obs, _ = env2.reset()
    q_h, a_h, p_h, r_h = [], [], [], []
    for _ in range(300):
        q_h.append([env2.queues[i].length for i in range(4)])
        a = eval_agent.select_action(obs, evaluate=True)
        ns, _, t, tr, _ = env2.step(a)
        a_h.append(a); p_h.append(env2.current_phase)
        r_h.append(env2.arrival_rates.copy())
        obs = env2.reset()[0] if (t or tr) else ns
    plot_phase_decision_timeline(
        np.array(q_h), np.array(a_h), np.array(p_h), np.array(r_h),
        save_path=os.path.join(output_dir, "phase_timeline.png"),
    )
    print("  ✓ Phase Timeline")

    # (g) Policy Evolution — PG Agent 专用 (用已训练模型)
    if is_pg:
        env_pg = make_env(traffic_pattern=traffic_pattern, max_steps=200)
        pg_agent_viz = create_pg_agent(agent_type=agent_type, state_dim=8)
        model_path = os.path.join(output_dir, "models", f"{agent_type}_{traffic_pattern}.pt")
        if os.path.isfile(model_path):
            pg_agent_viz.load(model_path)
        plot_policy_evolution(
            pg_agent_viz, env_pg, episodes=50, max_steps=200,
            save_path=os.path.join(output_dir, "policy_evolution.png"),
            title=f"{ALGO_LABELS.get(agent_type, agent_type)} — Policy Evolution",
        )
        print("  ✓ Policy Evolution")

    # (e) Multi-Seed Ribbon
    seed_dict = {ALGO_LABELS.get(agent_type, agent_type): data["rewards"]}
    plot_multi_seed_ribbon(
        seed_dict,
        save_path=os.path.join(output_dir, "multi_seed_ribbon.png"),
        title=f"{ALGO_LABELS.get(agent_type, agent_type)} — {seeds}-Seed Training",
    )
    print("  ✓ Multi-Seed Ribbon")

    print(f"\n  All figures saved to: {os.path.abspath(output_dir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# 消融实验调度
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_MAP = {
    "algorithm": run_algorithm_comparison,
    "network": run_network_ablation,
    "state": run_state_ablation,
    "reward": run_reward_ablation,
    "hyperparam": run_hyperparameter_sensitivity,
    "traffic": run_traffic_robustness,
}


def run_ablation(
    ablation_type: str,
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """运行单类消融实验并生成全套图表。"""
    os.makedirs(output_dir, exist_ok=True)

    if ablation_type not in ABLATION_MAP:
        print(f"Unknown ablation: {ablation_type}")
        print(f"  Choose from: {list(ABLATION_MAP.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  Ablation: {ablation_type}")
    print(f"{'='*60}")

    fn = ABLATION_MAP[ablation_type]
    if ablation_type == "hyperparam":
        results = fn(episodes=episodes, seeds=seeds, max_steps=max_steps)
        for hp_name, result in results.items():
            save_results(result, os.path.join(output_dir, f"ablation_{hp_name}.json"))
            generate_latex_table(result, save_path=os.path.join(
                output_dir, f"table_{hp_name}.tex"))
            print(f"  {hp_name}: best={max(result.configs, key=lambda c: result.summary[c]['reward'][0])}")
    else:
        result = fn(episodes=episodes, seeds=seeds, max_steps=max_steps)
        save_results(result, os.path.join(output_dir, f"ablation_{ablation_type}.json"))
        generate_latex_table(result, save_path=os.path.join(
            output_dir, f"table_{ablation_type}.tex"))

        # 可视化
        if result.rewards:
            seed_curves = {}
            for cfg in result.configs:
                if cfg in result.rewards:
                    label = ALGO_LABELS.get(cfg, cfg)
                    seed_curves[label] = result.rewards[cfg]
            if seed_curves:
                plot_multi_seed_ribbon(
                    seed_curves,
                    save_path=os.path.join(output_dir, f"ablation_{ablation_type}_ribbon.png"),
                    title=result.name,
                )

        print(f"\n  Summary ({result.name}):")
        for cfg in result.configs:
            if cfg in result.summary:
                m, ci = result.summary[cfg]["reward"]
                print(f"    {cfg:25s}: R={m:.1f} ± {(ci[1] - ci[0]) / 2:.1f}")

    print(f"\n  Results saved to: {os.path.abspath(output_dir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# 完整实验
# ══════════════════════════════════════════════════════════════════════════════

def run_full_experiment(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """运行全部 6 类消融实验 + Fixed-Time baseline。"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  FULL EXPERIMENT SUITE")
    print("=" * 70)
    total_t0 = time.time()

    # ── 0. Fixed-Time Baseline ──
    print("\n" + "=" * 50)
    print("  0. Fixed-Time Baseline")
    print("=" * 50)
    for pattern in ["uniform", "peak_hour", "tidal", "burst", "low_traffic"]:
        stats = evaluate_fixed_time(pattern, max_steps, episodes=seeds)
        print(f"  {pattern:12s}: R={stats['reward_mean']:8.1f}, "
              f"Q={stats['queue_mean']:6.2f}, W={stats['wait_mean']:5.1f}, "
              f"Dep={stats['throughput_mean']:5.0f}")

    # ── 1-6. 消融实验 ──
    for ab_type in ["algorithm", "network", "state", "reward", "traffic"]:
        print(f"\n{'='*50}")
        print(f"  {ab_type.upper()} Ablation")
        print(f"{'='*50}")
        try:
            run_ablation(ab_type, episodes=episodes, seeds=seeds,
                        max_steps=max_steps, output_dir=output_dir)
        except Exception as e:
            print(f"  ⚠ {ab_type} failed: {e}")

    # ── Final Summary Figure ──
    print(f"\n{'='*50}")
    print(f"  Generating Grand Summary Figure")
    print(f"{'='*50}")

    total_elapsed = time.time() - total_t0
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"  All results saved to: {os.path.abspath(output_dir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Traffic Signal RL Adaptive Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --full --episodes 500 --seeds 5
  python main.py --dashboard --agent dueling_per
  python main.py --ablation algorithm --seeds 3
  python main.py --train-only --agent noisy --episodes 200
  python main.py --ablation all --episodes 200 --seeds 3
        """,
    )
    # 模式
    parser.add_argument("--full", action="store_true",
                        help="Run all 6 ablation experiments")
    parser.add_argument("--dashboard", action="store_true",
                        help="Single algorithm training + full dashboard")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["algorithm","network","state","reward","hyperparam","traffic","all"],
                        help="Run specific ablation experiment")
    parser.add_argument("--train-only", action="store_true",
                        help="Train only, no visualizations")

    # 参数
    parser.add_argument("--agent", type=str, default="dueling",
                        help="Agent type: dqn|double|dueling|noisy|boltzmann|per_dqn|dueling_per|a2c|ppo (default: dueling)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Training episodes (default: 300)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (default: 3)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--state-dim", type=int, default=8,
                        choices=[4, 8, 9],
                        help="State dimension (default: 8)")
    parser.add_argument("--reward-type", type=str, default="composite",
                        choices=["queue_only","queue_switch","queue_wait","composite"],
                        help="Reward type (default: composite)")
    parser.add_argument("--pattern", type=str, default="uniform",
                        choices=["uniform","peak_hour","tidal","burst","low_traffic"],
                        help="Traffic pattern (default: uniform)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")

    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Traffic Signal RL Adaptive Control System")
    print("=" * 60)
    print(f"  Output: {os.path.abspath(output_dir)}/")

    # PG agents 训练慢, 自动缩减默认参数
    if args.agent in ("a2c", "ppo") and args.episodes == 300 and args.max_steps == 1000:
        if not args.dashboard and not args.train_only and not args.full:
            args.episodes = 200
        print(f"  ⚠ PG agent detected → episodes={args.episodes}, max_steps={args.max_steps}")
        print(f"     Tip: use --train-only for faster pure training")

    if args.full:
        run_full_experiment(
            episodes=args.episodes, seeds=args.seeds,
            max_steps=args.max_steps, output_dir=output_dir,
        )
    elif args.ablation:
        if args.ablation == "all":
            run_full_experiment(
                episodes=args.episodes, seeds=args.seeds,
                max_steps=args.max_steps, output_dir=output_dir,
            )
        else:
            run_ablation(
                args.ablation,
                episodes=args.episodes, seeds=args.seeds,
                max_steps=args.max_steps, output_dir=output_dir,
            )
    elif args.dashboard:
        run_dashboard(
            agent_type=args.agent,
            traffic_pattern=args.pattern,
            episodes=args.episodes, seeds=args.seeds,
            max_steps=args.max_steps, output_dir=output_dir,
        )
    elif args.train_only:
        data = run_single_training(
            agent_type=args.agent,
            state_dim=args.state_dim,
            traffic_pattern=args.pattern,
            reward_type=args.reward_type,
            episodes=args.episodes, seeds=args.seeds,
            max_steps=args.max_steps, output_dir=output_dir,
        )
        print(f"\n  Training done. Final reward: {data['rewards'][:,-50:].mean():.1f}")
    else:
        # 默认: 快速 dashboard
        print("\n  Running default quick dashboard...")
        run_dashboard(
            agent_type=args.agent,
            traffic_pattern=args.pattern,
            episodes=args.episodes, seeds=args.seeds,
            max_steps=args.max_steps, output_dir=output_dir,
        )

    print(f"\n{'='*60}")
    print(f"  ✅ Done! Results: {os.path.abspath(output_dir)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
