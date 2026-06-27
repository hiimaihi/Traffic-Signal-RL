#!/usr/bin/env python
"""
批量训练所有 RL Agent × 交通模式组合, 保存模型供前端驾驶舱加载。
===================================================================
用法:
  python train_all_models.py                  # 训练全部模型 (跳过已有)
  python train_all_models.py --force           # 覆盖已有模型
  python train_all_models.py --dry-run         # 仅列出将训练的模型

输出:
  results/models/{agent}_{pattern}.pt          # 每个组合一个模型文件
  results/train_all_report.json                # 训练报告
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import List, Dict, Optional

import numpy as np
import torch
import random as py_random

from traffic_env import make_env
from agent import create_agent
from pg_agent import create_pg_agent

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# Agent 列表 (与前端下拉菜单一致)
DQN_AGENTS = [
    "dqn", "double", "dueling", "noisy", "noisy_double",
    "boltzmann", "boltzmann_double", "per_dqn", "dueling_per",
]
PG_AGENTS = ["a2c", "ppo"]

# 交通模式
PATTERNS = ["uniform", "peak_hour", "tidal", "burst", "low_traffic"]

# 快速训练参数 (为前端展示而训练, 不需要完全收敛)
TRAIN_EPISODES = 50
TRAIN_SEEDS = 1
TRAIN_MAX_STEPS = 200
OUTPUT_DIR = "results"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


# ══════════════════════════════════════════════════════════════════════════════
# 核心训练函数
# ══════════════════════════════════════════════════════════════════════════════

def train_one_model(
    agent_type: str,
    pattern: str,
    episodes: int = TRAIN_EPISODES,
    max_steps: int = TRAIN_MAX_STEPS,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    训练一个 Agent + Pattern 组合, 保存模型, 返回训练摘要。
    """
    model_path = os.path.join(MODELS_DIR, f"{agent_type}_{pattern}.pt")
    start_time = time.time()

    # 创建环境和 Agent
    try:
        env = make_env(
            traffic_pattern=pattern,
            state_dim=8,
            reward_type="composite",
            max_steps=max_steps,
        )
    except Exception as e:
        return {"error": f"env creation: {e}"}

    try:
        if agent_type in ("a2c", "ppo"):
            agent = create_pg_agent(agent_type=agent_type, state_dim=8)
        else:
            agent = create_agent(agent_type=agent_type, state_dim=8)
    except Exception as e:
        return {"error": f"agent creation: {e}"}

    # 固定种子
    np.random.seed(42)
    py_random.seed(42)
    torch.manual_seed(42)

    is_on_policy = agent_type in ("a2c", "ppo")

    # ── 训练 ──
    ep_rewards = []
    ep_losses = []
    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        total_r = 0.0

        for step in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)

            if not is_on_policy:
                agent.update()

            obs = next_obs
            total_r += reward
            total_steps += 1
            if done:
                break

        if is_on_policy:
            loss = agent.update()
        else:
            loss = agent.loss_history[-1] if agent.loss_history else 0.0

        ep_rewards.append(float(total_r))
        ep_losses.append(float(loss))

    # ── 保存 ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.save(model_path)

    elapsed = time.time() - start_time
    avg_r_first10 = np.mean(ep_rewards[:10])
    avg_r_last10 = np.mean(ep_rewards[-10:])

    summary = {
        "agent": agent_type,
        "pattern": pattern,
        "model_path": model_path,
        "episodes": episodes,
        "max_steps": max_steps,
        "reward_first10": round(avg_r_first10, 1),
        "reward_last10": round(avg_r_last10, 1),
        "reward_improved": bool(avg_r_last10 > avg_r_first10),
        "final_loss": round(ep_losses[-1], 4),
        "elapsed_s": round(elapsed, 1),
    }

    if verbose:
        arrow = "↑" if summary["reward_improved"] else "→"
        print(f"  ✓ {agent_type:18s} | {pattern:12s} | "
              f"R: {avg_r_first10:.0f} → {avg_r_last10:.0f} {arrow} | "
              f"loss={ep_losses[-1]:.3f} | {elapsed:.1f}s")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量训练全部 RL Agent, 供前端驾驶舱使用"
    )
    parser.add_argument("--force", action="store_true",
                        help="覆盖已有模型文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出将训练的模型, 不实际训练")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES,
                        help=f"训练 episode 数 (默认: {TRAIN_EPISODES})")
    parser.add_argument("--max-steps", type=int, default=TRAIN_MAX_STEPS,
                        help=f"单 episode 最大步数 (默认: {TRAIN_MAX_STEPS})")
    parser.add_argument("--agent", type=str, default=None,
                        help="仅训练指定 agent (逗号分隔)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="仅训练指定 pattern (逗号分隔)")
    args = parser.parse_args()

    # 筛选
    agents_to_train = list(DQN_AGENTS) + list(PG_AGENTS)
    patterns_to_train = list(PATTERNS)

    if args.agent:
        agents_to_train = [a.strip() for a in args.agent.split(",")]
        # 验证
        all_valid = set(DQN_AGENTS + PG_AGENTS)
        for a in agents_to_train:
            if a not in all_valid:
                print(f"⚠ Unknown agent: {a}")
                print(f"  Valid: {sorted(all_valid)}")
                return

    if args.pattern:
        patterns_to_train = [p.strip() for p in args.pattern.split(",")]
        for p in patterns_to_train:
            if p not in PATTERNS:
                print(f"⚠ Unknown pattern: {p}")
                print(f"  Valid: {PATTERNS}")
                return

    # 构建任务列表
    tasks = []
    skipped = []
    for agent in agents_to_train:
        for pattern in patterns_to_train:
            model_path = os.path.join(MODELS_DIR, f"{agent}_{pattern}.pt")
            if os.path.isfile(model_path) and not args.force:
                skipped.append((agent, pattern))
                continue
            tasks.append((agent, pattern))

    # ── 打印计划 ──
    print("=" * 70)
    print("  🚦 Traffic Signal RL — 批量模型训练")
    print("=" * 70)
    print(f"  Agents:  {len(agents_to_train)} ({', '.join(agents_to_train)})")
    print(f"  Patterns: {len(patterns_to_train)} ({', '.join(patterns_to_train)})")
    print(f"  Episodes: {args.episodes} | Max steps: {args.max_steps}")
    print(f"  Force overwrite: {args.force}")
    print(f"  Output: {os.path.abspath(MODELS_DIR)}/")
    print()

    if skipped:
        print(f"  ⏭  Skipping {len(skipped)} existing models "
              f"(use --force to overwrite)")
    print(f"  🎯 {len(tasks)} models to train")

    if args.dry_run:
        print("\n  Planned:")
        for agent, pattern in tasks:
            print(f"    {agent:18s} × {pattern}")
        print(f"\n  Dry run — no training performed.")
        return

    if not tasks:
        print("\n  ✅ All models already trained! Nothing to do.")
        return

    # ── 训练 ──
    print(f"\n{'='*70}")
    print(f"  Training {len(tasks)} models...")
    print(f"{'='*70}\n")

    t0 = time.time()
    results = []
    errors = []

    for i, (agent, pattern) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {agent} × {pattern} ", end="", flush=True)
        try:
            summary = train_one_model(
                agent_type=agent,
                pattern=pattern,
                episodes=args.episodes,
                max_steps=args.max_steps,
                verbose=False,
            )
            if summary and "error" not in summary:
                results.append(summary)
                arrow = "↑" if summary["reward_improved"] else "→"
                print(f"R: {summary['reward_first10']:.0f} → {summary['reward_last10']:.0f} "
                      f"{arrow} | loss={summary['final_loss']:.3f} | {summary['elapsed_s']:.1f}s")
            else:
                errors.append({"agent": agent, "pattern": pattern,
                               "error": summary.get("error", "unknown") if summary else "None"})
                print(f"⚠ ERROR: {errors[-1]['error']}")
        except Exception as e:
            errors.append({"agent": agent, "pattern": pattern, "error": str(e)})
            print(f"✗ CRASH: {e}")
            traceback.print_exc()

    total_elapsed = time.time() - t0

    # ── 报告 ──
    print(f"\n{'='*70}")
    print(f"  📊 Training Report")
    print(f"{'='*70}")
    print(f"  Completed: {len(results)}/{len(tasks)}")
    print(f"  Errors:    {len(errors)}")
    print(f"  Total:     {total_elapsed/60:.1f} min")
    print()

    if results:
        # 按 reward 改善排序
        improved = [r for r in results if r["reward_improved"]]
        print(f"  Reward improved: {len(improved)}/{len(results)}")
        print()

        print(f"  {'Agent':18s} {'Pattern':12s} {'R-start':>8s} {'R-end':>8s} {'Loss':>8s} {'Time':>7s}")
        print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
        for r in sorted(results, key=lambda x: x["reward_last10"] - x["reward_first10"], reverse=True):
            arrow = "↑" if r["reward_improved"] else "→"
            print(f"  {r['agent']:18s} {r['pattern']:12s} "
                  f"{r['reward_first10']:>8.0f} {r['reward_last10']:>8.0f} {arrow} "
                  f"{r['final_loss']:>8.3f} {r['elapsed_s']:>6.1f}s")

    if errors:
        print(f"\n  ⚠ Errors:")
        for e in errors:
            print(f"    {e['agent']} × {e['pattern']}: {e['error']}")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "train_all_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed": len(results),
        "total_tasks": len(tasks),
        "errors": errors,
        "total_elapsed_s": round(total_elapsed, 1),
        "params": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "agents": agents_to_train,
            "patterns": patterns_to_train,
        },
        "results": results,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Report saved: {os.path.abspath(report_path)}")

    # 最终状态
    final_count = sum(
        1 for a in agents_to_train for p in patterns_to_train
        if os.path.isfile(os.path.join(MODELS_DIR, f"{a}_{p}.pt"))
    )
    print(f"  🧠 Models on disk: {final_count}/{len(agents_to_train) * len(patterns_to_train)}")

    if errors:
        print(f"\n  ⚠ {len(errors)} models failed. Re-run to retry failed ones.")

    print(f"\n  ✅ Done! Start frontend: python backend/app.py")
    print(f"     Open http://localhost:5000 in browser\n")


if __name__ == "__main__":
    main()
