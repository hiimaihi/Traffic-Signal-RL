"""
交通信号灯自适应控制 — 实验框架
=================================
6 类消融/对比实验, 含统计检验。

实验类型:
  1. AlgorithmComparison       — 6 算法 × 5 seeds 对比
  2. NetworkAblation           — 4 网络架构 × 5 seeds 消融
  3. StateRepresentationAblation — 3 状态变体 × 5 seeds
  4. RewardFunctionAblation    — 4 奖励函数 × 5 seeds
  5. HyperparameterSensitivity — 3 超参 × 4 values × 5 seeds
  6. TrafficRobustness         — 4 模式 × 2 算法 × 5 seeds

统计检验:
  - 配对 t 检验 (最佳 vs 次佳)
  - 95% 置信区间 (CI)
  - LaTeX 表格自动生成
  - 效应量 (Cohen's d)
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import stats as scipy_stats

from traffic_env import make_env, FixedTimeController
from agent import create_agent
from network import create_network
from pg_agent import create_pg_agent


# ══════════════════════════════════════════════════════════════════════════════
# 统计工具
# ══════════════════════════════════════════════════════════════════════════════

def compute_95ci(data: np.ndarray) -> Tuple[float, float]:
    """计算均值的 95% 置信区间。"""
    n = len(data)
    if n < 2:
        return float(np.mean(data)), float(np.mean(data))
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    ci = 1.96 * sem
    return mean - ci, mean + ci


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """配对 t 检验, 返回 (t_stat, p_value)。"""
    t_stat, p_val = scipy_stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d 效应量。"""
    diff = np.mean(a) - np.mean(b)
    n1, n2 = len(a), len(b)
    s_pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
                       / (n1 + n2 - 2))
    if s_pooled < 1e-8:
        return 0.0
    return float(diff / s_pooled)


@dataclass
class ExperimentResult:
    """单个实验的完整结果。"""
    name: str
    seeds: int
    episodes: int
    configs: List[str] = field(default_factory=list)
    # 每个 config 的逐 episode 数据: shape (seeds, episodes)
    rewards: Dict[str, np.ndarray] = field(default_factory=dict)
    queues: Dict[str, np.ndarray] = field(default_factory=dict)
    waits: Dict[str, np.ndarray] = field(default_factory=dict)
    throughput: Dict[str, np.ndarray] = field(default_factory=dict)
    # 汇总统计
    summary: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=dict)  # {config: {metric: (mean, ci)}}
    t_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict:
        result = {
            "name": self.name,
            "seeds": self.seeds,
            "episodes": self.episodes,
            "configs": self.configs,
            "elapsed_time": self.elapsed_time,
            "summary": {},
            "t_tests": self.t_tests,
        }
        for cfg in self.configs:
            result["summary"][cfg] = {
                k: {"mean": m, "ci": c}
                for k, (m, c) in self.summary.get(cfg, {}).items()
            }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 训练 / 评估工具
# ══════════════════════════════════════════════════════════════════════════════

def train_agent(
    agent,
    env,
    episodes: int = 300,
    max_steps: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    训练一个 agent, 返回 (rewards, queues, waits, throughputs)。

    支持两类 Agent:
      - Off-policy (DQN 系列): 每步更新
      - On-policy (PG 系列: A2C/PPO): episode 结束时才更新
    """
    # 检测是否为 on-policy agent (有 RolloutBuffer)
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'states')

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
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)

            if not is_on_policy:
                agent.update()

            obs = next_obs
            total_r += reward
            total_q += info["total_queue"]
            total_w += info["avg_wait"]
            steps += 1
            if done:
                break

        # On-policy: 整个 episode 完成后一次性更新
        if is_on_policy:
            agent.update()

        episode_rewards[ep] = total_r
        episode_queues[ep] = total_q / max(steps, 1)
        episode_waits[ep] = total_w / max(steps, 1)
        episode_throughput[ep] = env.total_departed

        if verbose and ep % 50 == 0:
            print(f"    Ep {ep:3d}/{episodes} | R={total_r:8.1f} "
                  f"Q={total_q / max(steps, 1):5.2f} W={total_w / max(steps, 1):5.1f} "
                  f"Dep={env.total_departed:4d}")

    return episode_rewards, episode_queues, episode_waits, episode_throughput


def evaluate_fixed_time(
    traffic_pattern: str = "uniform",
    max_steps: int = 1000,
    fixed_interval: int = 30,
    episodes: int = 5,
) -> Dict[str, float]:
    """评估固定配时基线。"""
    env = make_env(traffic_pattern=traffic_pattern, max_steps=max_steps)
    ft = FixedTimeController(env, fixed_interval=fixed_interval)

    rewards, queues, waits, throughput = [], [], [], []
    for _ in range(episodes):
        stats = ft.run_episode()
        rewards.append(stats["total_reward"])
        queues.append(stats["avg_queue"])
        waits.append(stats["avg_wait"])
        throughput.append(stats["total_departed"])

    return {
        "reward_mean": np.mean(rewards), "reward_ci": compute_95ci(np.array(rewards))[1] - np.mean(rewards),
        "queue_mean": np.mean(queues), "queue_ci": compute_95ci(np.array(queues))[1] - np.mean(queues),
        "wait_mean": np.mean(waits), "wait_ci": compute_95ci(np.array(waits))[1] - np.mean(waits),
        "throughput_mean": np.mean(throughput),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 实验 1: 算法对比
# ══════════════════════════════════════════════════════════════════════════════

def run_algorithm_comparison(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    traffic_pattern: str = "uniform",
    verbose: bool = True,
) -> ExperimentResult:
    """
    对比 8 种算法: 6 种 DQN 变体 + A2C + PPO。

    DQN 系列: DQN, Double, Dueling, Noisy, Boltzmann, PER
    PG 系列: A2C, PPO
    """
    agent_types = [
        "dqn", "double", "dueling", "noisy", "boltzmann", "per_dqn",
        "a2c", "ppo",
    ]
    # PG agents need a different factory
    pg_types = {"a2c", "ppo"}

    result = ExperimentResult(
        name="Algorithm Comparison",
        seeds=seeds, episodes=episodes,
        configs=agent_types,
    )
    t0 = time.time()

    for at in agent_types:
        if verbose:
            print(f"\n{'='*50}\n  Training: {at}\n{'='*50}")
        all_r, all_q, all_w, all_tp = [], [], [], []
        for seed in range(seeds):
            env = make_env(traffic_pattern=traffic_pattern, max_steps=max_steps)
            if at in pg_types:
                agent = create_pg_agent(agent_type=at, state_dim=8)
            else:
                agent = create_agent(agent_type=at, state_dim=8)
            np.random.seed(seed)
            import random
            random.seed(seed)
            import torch
            torch.manual_seed(seed)
            r, q, w, tp = train_agent(agent, env, episodes, max_steps,
                                      verbose=(verbose and seed == 0))
            all_r.append(r); all_q.append(q); all_w.append(w); all_tp.append(tp)
        result.rewards[at] = np.array(all_r)
        result.queues[at] = np.array(all_q)
        result.waits[at] = np.array(all_w)
        result.throughput[at] = np.array(all_tp)

        # 汇总
        final_r = result.rewards[at][:, -50:].mean(axis=1)
        final_q = result.queues[at][:, -50:].mean(axis=1)
        final_w = result.waits[at][:, -50:].mean(axis=1)
        final_tp = result.throughput[at][:, -1]
        result.summary[at] = {
            "reward": (np.mean(final_r), compute_95ci(final_r)),
            "queue": (np.mean(final_q), compute_95ci(final_q)),
            "wait": (np.mean(final_w), compute_95ci(final_w)),
            "throughput": (np.mean(final_tp), compute_95ci(final_tp)),
        }

    # 统计检验 (最佳 vs 次佳)
    best = max(agent_types, key=lambda a: result.summary[a]["reward"][0])
    others = [a for a in agent_types if a != best]
    second_best = max(others, key=lambda a: result.summary[a]["reward"][0])
    best_r = result.rewards[best][:, -50:].mean(axis=1)
    second_r = result.rewards[second_best][:, -50:].mean(axis=1)
    t_stat, p_val = paired_t_test(best_r, second_r)
    d = cohens_d(best_r, second_r)
    result.t_tests = {
        "best_vs_second": {"best": best, "second": second_best,
                           "t_stat": t_stat, "p_value": p_val, "cohens_d": d},
    }

    result.elapsed_time = time.time() - t0
    if verbose:
        print(f"\n  Best: {best} (R={result.summary[best]['reward'][0]:.1f})")
        print(f"  vs {second_best} (R={result.summary[second_best]['reward'][0]:.1f})")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")
        print(f"  Elapsed: {result.elapsed_time:.1f}s")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 实验 2: 网络架构消融
# ══════════════════════════════════════════════════════════════════════════════

def run_network_ablation(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    verbose: bool = True,
) -> ExperimentResult:
    """
    对比 4 种网络: shallow(32), standard(128), dueling(V/A), deep(256×4).
    """
    configs = ["shallow", "standard", "dueling", "deep"]
    result = ExperimentResult(
        name="Network Architecture Ablation",
        seeds=seeds, episodes=episodes,
        configs=configs,
    )
    t0 = time.time()

    for cfg in configs:
        if verbose:
            print(f"\n{'='*50}\n  Network: {cfg}\n{'='*50}")
        all_r, all_q, all_w, all_tp = [], [], [], []
        for seed in range(seeds):
            env = make_env(max_steps=max_steps)
            if cfg == "shallow":
                agent = create_agent("dqn", state_dim=8, hidden_dim=32)
            elif cfg == "standard":
                agent = create_agent("dqn", state_dim=8)
            elif cfg == "dueling":
                agent = create_agent("dueling", state_dim=8)
            elif cfg == "deep":
                agent = create_agent("dqn", state_dim=8)
                from network import DeepDQN
                agent.online_net = DeepDQN(input_dim=8, output_dim=2).to(agent.device)
                agent.target_net = DeepDQN(input_dim=8, output_dim=2).to(agent.device)
                agent.target_net.load_state_dict(agent.online_net.state_dict())
            import torch, random
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            r, q, w, tp = train_agent(agent, env, episodes, max_steps,
                                      verbose=(verbose and seed == 0))
            all_r.append(r); all_q.append(q); all_w.append(w); all_tp.append(tp)
        result.rewards[cfg] = np.array(all_r)
        result.queues[cfg] = np.array(all_q)
        result.waits[cfg] = np.array(all_w)
        result.throughput[cfg] = np.array(all_tp)

        final_r = result.rewards[cfg][:, -50:].mean(axis=1)
        result.summary[cfg] = {
            "reward": (np.mean(final_r), compute_95ci(final_r)),
        }

    result.elapsed_time = time.time() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 实验 3: 状态表示消融
# ══════════════════════════════════════════════════════════════════════════════

def run_state_ablation(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    verbose: bool = True,
) -> ExperimentResult:
    """
    对比 3 种状态: S4(仅排队), S8(排队+等待), S9(排队+等待+相位).
    """
    configs = ["S4 (queue)", "S8 (queue+wait)", "S9 (+phase)"]
    state_dims = [4, 8, 9]
    result = ExperimentResult(
        name="State Representation Ablation",
        seeds=seeds, episodes=episodes,
        configs=configs,
    )
    t0 = time.time()

    for cfg, sd in zip(configs, state_dims):
        if verbose:
            print(f"\n{'='*50}\n  State: {cfg} (dim={sd})\n{'='*50}")
        all_r, all_q, all_w, all_tp = [], [], [], []
        for seed in range(seeds):
            env = make_env(state_dim=sd, max_steps=max_steps)
            agent = create_agent("dueling", state_dim=sd)
            import torch, random
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            r, q, w, tp = train_agent(agent, env, episodes, max_steps,
                                      verbose=(verbose and seed == 0))
            all_r.append(r); all_q.append(q); all_w.append(w); all_tp.append(tp)
        result.rewards[cfg] = np.array(all_r)
        result.queues[cfg] = np.array(all_q)
        result.waits[cfg] = np.array(all_w)
        result.throughput[cfg] = np.array(all_tp)

        final_r = result.rewards[cfg][:, -50:].mean(axis=1)
        result.summary[cfg] = {
            "reward": (np.mean(final_r), compute_95ci(final_r)),
        }

    result.elapsed_time = time.time() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 实验 4: 奖励函数消融
# ══════════════════════════════════════════════════════════════════════════════

def run_reward_ablation(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    verbose: bool = True,
) -> ExperimentResult:
    """
    对比 4 种奖励: R1(queue), R2(queue+switch), R3(queue+wait), R4(composite).
    """
    configs = ["R1: queue", "R2: queue+penalty", "R3: queue+wait", "R4: composite"]
    reward_types = ["queue_only", "queue_switch", "queue_wait", "composite"]
    result = ExperimentResult(
        name="Reward Function Ablation",
        seeds=seeds, episodes=episodes,
        configs=configs,
    )
    t0 = time.time()

    for cfg, rt in zip(configs, reward_types):
        if verbose:
            print(f"\n{'='*50}\n  Reward: {cfg}\n{'='*50}")
        all_r, all_q, all_w, all_tp = [], [], [], []
        for seed in range(seeds):
            env = make_env(reward_type=rt, max_steps=max_steps)
            agent = create_agent("dueling", state_dim=8)
            import torch, random
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            r, q, w, tp = train_agent(agent, env, episodes, max_steps,
                                      verbose=(verbose and seed == 0))
            all_r.append(r); all_q.append(q); all_w.append(w); all_tp.append(tp)
        result.rewards[cfg] = np.array(all_r)
        result.queues[cfg] = np.array(all_q)
        result.waits[cfg] = np.array(all_w)
        result.throughput[cfg] = np.array(all_tp)

        final_r = result.rewards[cfg][:, -50:].mean(axis=1)
        result.summary[cfg] = {
            "reward": (np.mean(final_r), compute_95ci(final_r)),
        }

    result.elapsed_time = time.time() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 实验 5: 超参数敏感性
# ══════════════════════════════════════════════════════════════════════════════

def run_hyperparameter_sensitivity(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    verbose: bool = True,
) -> Dict[str, ExperimentResult]:
    """
    3 组超参数 × 4 值: lr, gamma, tau.
    """
    hp_ranges = {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3],
        "gamma": [0.9, 0.95, 0.99, 0.999],
        "tau": [0.001, 0.005, 0.01, 0.05],
    }
    results = {}
    for hp_name, hp_values in hp_ranges.items():
        configs = [f"{hp_name}={v}" for v in hp_values]
        exp = ExperimentResult(
            name=f"Hyperparameter Sensitivity: {hp_name}",
            seeds=seeds, episodes=episodes,
            configs=configs,
        )
        t0 = time.time()
        for cfg, val in zip(configs, hp_values):
            if verbose:
                print(f"\n  HP: {cfg}")
            all_r = []
            for seed in range(seeds):
                kwargs = {"state_dim": 8, "agent_type": "dueling",
                          hp_name: val}
                agent = create_agent(**kwargs)
                env = make_env(max_steps=max_steps)
                import torch, random
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                r, q, w, tp = train_agent(agent, env, episodes, max_steps, verbose=False)
                all_r.append(r)
            exp.rewards[cfg] = np.array(all_r)
            final_r = exp.rewards[cfg][:, -50:].mean(axis=1)
            exp.summary[cfg] = {
                "reward": (np.mean(final_r), compute_95ci(final_r)),
            }
        exp.elapsed_time = time.time() - t0
        results[hp_name] = exp
        if verbose:
            print(f"    Best {hp_name}: {max(configs, key=lambda c: exp.summary[c]['reward'][0])}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 实验 6: 交通鲁棒性
# ══════════════════════════════════════════════════════════════════════════════

def run_traffic_robustness(
    episodes: int = 300,
    seeds: int = 5,
    max_steps: int = 1000,
    verbose: bool = True,
) -> ExperimentResult:
    """
    4 种交通模式 × 2 算法 (Dueling DQN vs DQN).
    """
    patterns = ["uniform", "peak_hour", "tidal", "burst"]
    algos = ["dueling", "dqn"]
    configs = [f"{p}|{a}" for p in patterns for a in algos]
    result = ExperimentResult(
        name="Traffic Pattern Robustness",
        seeds=seeds, episodes=episodes,
        configs=configs,
    )
    t0 = time.time()

    for p in patterns:
        for a in algos:
            cfg = f"{p}|{a}"
            if verbose:
                print(f"\n  Pattern={p}, Algo={a}")
            all_r = []
            for seed in range(seeds):
                env = make_env(traffic_pattern=p, max_steps=max_steps)
                agent = create_agent(agent_type=a, state_dim=8)
                import torch, random
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                r, q, w, tp = train_agent(agent, env, episodes, max_steps, verbose=False)
                all_r.append(r)
            result.rewards[cfg] = np.array(all_r)
            final_r = result.rewards[cfg][:, -50:].mean(axis=1)
            result.summary[cfg] = {
                "reward": (np.mean(final_r), compute_95ci(final_r)),
            }

    result.elapsed_time = time.time() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX 表格生成
# ══════════════════════════════════════════════════════════════════════════════

def generate_latex_table(
    result: ExperimentResult,
    metric: str = "reward",
    save_path: Optional[str] = None,
) -> str:
    """从 ExperimentResult 生成 LaTeX 表格。"""
    lines = [r"\begin{table}[htbp]",
             r"\centering",
             r"\caption{" + result.name + r"}",
             r"\label{tab:" + result.name.lower().replace(" ", "_") + r"}",
             r"\begin{tabular}{l" + "c" * len(result.configs) + r"}",
             r"\toprule"]
    header = "Algorithm & " + " & ".join(result.configs) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    row_mean = "Mean "
    row_ci = "95\\% CI "
    for cfg in result.configs:
        if cfg in result.summary and metric in result.summary[cfg]:
            mean_ci = result.summary[cfg][metric]
            row_mean += f" & {mean_ci[0]:.1f}"
            row_ci += f" & $\\pm${abs(mean_ci[1][1] - mean_ci[1][0]) / 2:.1f}"
        else:
            row_mean += " & —"
            row_ci += " & —"
    row_mean += r" \\"
    row_ci += r" \\"
    lines.append(row_mean)
    lines.append(row_ci)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
    return latex_str


# ══════════════════════════════════════════════════════════════════════════════
# 保存/加载
# ══════════════════════════════════════════════════════════════════════════════

def save_results(result: ExperimentResult, save_path: str) -> None:
    """保存实验结果 (JSON)。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 提取关键数据
    data = result.to_dict()
    # 只保存最后 50 ep 的平均
    data["final_rewards"] = {}
    for cfg in result.configs:
        if cfg in result.rewards:
            arr = result.rewards[cfg][:, -50:].mean(axis=1)
            data["final_rewards"][cfg] = arr.tolist()
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Experiments Module — Quick Tests")
    print("=" * 60)

    # 统计工具
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])
    ci = compute_95ci(a)
    t, p = paired_t_test(a, b)
    d = cohens_d(a, b)
    print(f"\n  Stats: CI={ci}, t={t:.3f}, p={p:.4f}, d={d:.3f}")

    # Fixed-time baseline
    print("\n--- Fixed-Time Baseline ---")
    ft_stats = evaluate_fixed_time("uniform", max_steps=200, episodes=3)
    print(f"  Reward={ft_stats['reward_mean']:.1f}, Queue={ft_stats['queue_mean']:.2f}, "
          f"Wait={ft_stats['wait_mean']:.2f}, Throughput={ft_stats['throughput_mean']:.0f}")

    # Quick algorithm comparison
    print("\n--- Quick Algorithm Comparison (2 algos × 2 seeds × 50 ep) ---")
    result = run_algorithm_comparison(episodes=50, seeds=2, max_steps=200, verbose=True)
    print(f"\n  Summary:")
    for cfg in result.configs:
        mean_ci = result.summary[cfg]["reward"]
        print(f"    {cfg:15s}: R={mean_ci[0]:.1f} ± {abs(mean_ci[1][1] - mean_ci[1][0]) / 2:.1f}")

    # LaTeX
    print("\n--- LaTeX Table ---")
    latex = generate_latex_table(result, save_path="results/test_table.tex")
    print(latex)

    print("\n✅ All experiment tests passed!")
