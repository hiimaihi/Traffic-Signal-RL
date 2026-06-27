"""
交通信号灯自适应控制 — 学术级可视化模块
========================================
6 大类可视化, 适配硕士/博士论文发表标准:
  (a) training_dashboard     — 4 面板训练曲线摘要
  (b) q_value_heatmap        — Q 值决策边界热力图
  (c) phase_decision_timeline— 相位决策时间线 (3 面板)
  (d) radar_chart            — 6 维雷达图算法对比
  (e) multi_seed_ribbon      — 多种子均值±95% CI 包络线
  (f) summary_grand_figure   — 综合大图 (2×3 面板)

输出: 300 DPI, 中文字体支持, 学术风格配色。
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from typing import Dict, List, Optional, Tuple, Any

# ══════════════════════════════════════════════════════════════════════════════
# 全局样式设置
# ══════════════════════════════════════════════════════════════════════════════

# 学术配色
COLORS = {
    "blue":   "#2196F3",
    "red":    "#E53935",
    "green":  "#43A047",
    "orange": "#FB8C00",
    "purple": "#8E24AA",
    "teal":   "#00897B",
    "grey":   "#757575",
    "dark":   "#212121",
}

AGENT_COLORS = {
    "DQN":           COLORS["blue"],
    "DoubleDQN":     COLORS["red"],
    "DuelingDQN":    COLORS["green"],
    "NoisyDQN":      COLORS["orange"],
    "BoltzmannDQN":  COLORS["purple"],
    "PER_DQN":        COLORS["teal"],
    "DuelingPER":   COLORS["dark"],
    "A2C":           "#FF6F00",
    "PPO":           "#795548",
}

ALGO_LABELS = {
    "dqn": "DQN", "double": "Double DQN", "dueling": "Dueling DQN",
    "noisy": "Noisy DQN", "noisy_double": "Noisy+Double",
    "boltzmann": "Boltzmann DQN", "boltzmann_double": "Boltz+Double",
    "per_dqn": "PER DQN", "dueling_per": "Dueling+PER",
    "a2c": "A2C", "ppo": "PPO",
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def setup_style() -> None:
    """应用全局样式。"""
    for k, v in STYLE.items():
        plt.rcParams[k] = v


def moving_average(data: np.ndarray, window: int = 50) -> np.ndarray:
    """滑动窗口平均。"""
    if len(data) < window:
        return np.array([np.mean(data[:i + 1]) for i in range(len(data))])
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode="valid")
    return np.concatenate([data[:window - 1], smoothed])


# ══════════════════════════════════════════════════════════════════════════════
# (a) 训练仪表盘 — 4 面板
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_dashboard(
    rewards: np.ndarray,
    queue_lengths: np.ndarray,
    wait_times: np.ndarray,
    losses: np.ndarray,
    save_path: str = "results/training_dashboard.png",
    title: str = "Training Dashboard",
    smooth_window: int = 50,
) -> str:
    """
    4 面板训练仪表盘:
      Top-Left:  Episode Reward
      Top-Right: Episode Loss
      Bottom-Left: Avg Queue Length
      Bottom-Right: Avg Wait Time
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    episodes = np.arange(1, len(rewards) + 1)

    # Panel 1: Reward
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.15, color=COLORS["blue"], linewidth=0.5)
    ax.plot(episodes, moving_average(rewards, smooth_window),
            color=COLORS["blue"], linewidth=1.8, label=f"MA({smooth_window})")
    ax.axhline(y=0, color=COLORS["grey"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Loss
    ax = axes[0, 1]
    if len(losses) > 0:
        ax.plot(losses, alpha=0.15, color=COLORS["red"], linewidth=0.5)
        ax.plot(moving_average(np.array(losses), smooth_window),
                color=COLORS["red"], linewidth=1.8, label=f"MA({smooth_window})")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("TD Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 3: Queue
    ax = axes[1, 0]
    ax.plot(episodes, queue_lengths, alpha=0.2, color=COLORS["green"], linewidth=0.5)
    ax.plot(episodes, moving_average(queue_lengths, smooth_window),
            color=COLORS["green"], linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Queue Length")
    ax.set_title("Average Queue Length")
    ax.grid(True, alpha=0.3)

    # Panel 4: Wait
    ax = axes[1, 1]
    ax.plot(episodes, wait_times, alpha=0.2, color=COLORS["orange"], linewidth=0.5)
    ax.plot(episodes, moving_average(wait_times, smooth_window),
            color=COLORS["orange"], linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Wait Time (steps)")
    ax.set_title("Average Wait Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# (b) Q 值热力图 — 决策边界可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_q_value_heatmap(
    agent,
    env,
    save_path: str = "results/q_value_heatmap.png",
    n_samples: int = 500,
) -> str:
    """
    通过采样 (q_NS, q_EW) 对画 Q 值决策边界热力图。
    横轴: 南北排队长度, 纵轴: 东西排队长度。
    """
    setup_style()
    import random
    random.seed(42)
    np.random.seed(42)

    q_ns_list, q_ew_list, queue_ns_list, queue_ew_list = [], [], [], []
    obs, _ = env.reset()
    agent.online_net.eval()

    for _ in range(n_samples):
        with torch.no_grad():
            st = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            q_vals = agent.online_net(st).cpu().numpy().flatten()
        q_ns_list.append(q_vals[0])  # Q(s, hold=NS)
        q_ew_list.append(q_vals[1])  # Q(s, switch=EW)
        raw = env.get_state_raw()
        queue_ns_list.append(raw["queues"][0] + raw["queues"][1])  # N+S
        queue_ew_list.append(raw["queues"][2] + raw["queues"][3])  # E+W
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Q-Value Decision Boundary Analysis", fontsize=14, fontweight="bold")

    # 子图 1: ΔQ = Q_hold - Q_switch
    ax = axes[0]
    delta_q = np.array(q_ns_list) - np.array(q_ew_list)
    sc = ax.scatter(queue_ns_list, queue_ew_list, c=delta_q, cmap="RdBu_r",
                    alpha=0.6, s=30, edgecolors="none", vmin=-2, vmax=2)
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("N+S Queue Length")
    ax.set_ylabel("E+W Queue Length")
    ax.set_title(r"$\Delta Q = Q_{hold} - Q_{switch}$")
    plt.colorbar(sc, ax=ax, label=r"$\Delta Q$")

    # 子图 2: 决策类别
    ax = axes[1]
    decisions = ["Hold (NS)" if dq > 0 else "Switch (EW)" for dq in delta_q]
    colors = [COLORS["blue"] if d == "Hold (NS)" else COLORS["red"] for d in decisions]
    ax.scatter(queue_ns_list, queue_ew_list, c=colors, alpha=0.4, s=25, edgecolors="none")
    ax.set_xlabel("N+S Queue Length")
    ax.set_ylabel("E+W Queue Length")
    ax.set_title("Policy Decision Map")
    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["blue"],
               markersize=8, label="Hold (NS)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["red"],
               markersize=8, label="Switch (EW)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# (c) 相位决策时间线 — 3 面板
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_decision_timeline(
    queues_history: np.ndarray,        # shape (steps, 4)
    actions_history: np.ndarray,       # shape (steps,)
    phases_history: np.ndarray,        # shape (steps,)
    arrival_rates: np.ndarray,         # shape (steps, 4)
    save_path: str = "results/phase_timeline.png",
    title: str = "Phase Decision Timeline",
) -> str:
    """
    3 面板: [Queue + Actions | Phase State | Arrival Rates]
    """
    setup_style()
    steps = range(len(actions_history))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Queue lengths & actions
    ax = axes[0]
    labels = ["N", "S", "E", "W"]
    colors_q = [COLORS["blue"], COLORS["blue"], COLORS["red"], COLORS["red"]]
    linestyles = ["-", "--", "-", "--"]
    for i in range(4):
        ax.plot(steps, queues_history[:, i], color=colors_q[i],
                linestyle=linestyles[i], linewidth=0.8, alpha=0.8, label=labels[i])
    # mark switch actions
    switch_steps = np.where(actions_history == 1)[0]
    for s in switch_steps:
        ax.axvline(x=s, color=COLORS["green"], alpha=0.15, linewidth=0.5)
    ax.set_ylabel("Queue Length")
    ax.set_title("Queue Lengths & Switch Actions (green lines)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: Phase state
    ax = axes[1]
    ax.fill_between(steps, 0, 1,
                    where=(phases_history == 0),
                    color=COLORS["blue"], alpha=0.4, label="Phase NS",
                    step="post")
    ax.fill_between(steps, 0, 1,
                    where=(phases_history == 1),
                    color=COLORS["red"], alpha=0.4, label="Phase EW",
                    step="post")
    ax.set_ylabel("Phase")
    ax.set_title("Current Phase (Blue=NS, Red=EW)")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["NS", "EW"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 3: Arrival rates
    ax = axes[2]
    for i in range(4):
        ax.plot(steps, arrival_rates[:, i], color=colors_q[i],
                linestyle=linestyles[i], linewidth=0.8, alpha=0.8, label=labels[i])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Arrival Rate (λ)")
    ax.set_title("Arrival Rates Over Time")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# (d) 雷达图 — 多算法 6 维性能对比
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(
    metrics: Dict[str, Dict[str, float]],  # {algo_name: {metric: value}}
    save_path: str = "results/radar_chart.png",
    title: str = "Algorithm Performance Radar",
) -> str:
    """
    6 维雷达图: [Avg Reward, Queue Reduction, Wait Reduction,
                  Throughput, Stability, Efficiency]
    所有指标归一化到 [0, 1]。
    """
    setup_style()
    metric_names = ["Avg Reward", "Queue ↓", "Wait ↓",
                    "Throughput", "Stability", "Efficiency"]
    n_metrics = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

    algo_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
                   COLORS["orange"], COLORS["purple"], COLORS["teal"], COLORS["dark"]]
    color_idx = 0

    for algo_name, algo_metrics in metrics.items():
        values = [algo_metrics.get(m, 0) for m in metric_names]
        values += values[:1]
        color = algo_colors[color_idx % len(algo_colors)]
        ax.fill(angles, values, alpha=0.1, color=color)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=algo_name,
                markersize=5)
        color_idx += 1

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# (e) 多种子包络线 — 均值 ± 95% CI
# ══════════════════════════════════════════════════════════════════════════════

def plot_multi_seed_ribbon(
    seed_curves: Dict[str, np.ndarray],  # {algo_label: shape (n_seeds, n_episodes)}
    save_path: str = "results/multi_seed_ribbon.png",
    title: str = "Multi-Seed Training Curves",
    ylabel: str = "Episode Reward",
    smooth_window: int = 20,
    ci_alpha: float = 0.3,
) -> str:
    """
    多种子训练曲线: 均值实线 + 95% CI 半透明填充区。
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    algo_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
                   COLORS["orange"], COLORS["purple"], COLORS["teal"]]
    color_idx = 0

    for algo_label, curves in seed_curves.items():
        color = algo_colors[color_idx % len(algo_colors)]
        n_seeds, n_eps = curves.shape

        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0, ddof=1)
        # 95% CI: mean ± 1.96 * std/sqrt(n)
        ci = 1.96 * std_curve / np.sqrt(n_seeds)

        smoothed_mean = moving_average(mean_curve, smooth_window)
        smoothed_ci_low = moving_average(mean_curve - ci, smooth_window)
        smoothed_ci_high = moving_average(mean_curve + ci, smooth_window)
        episodes = np.arange(1, len(smoothed_mean) + 1)

        ax.plot(episodes, smoothed_mean, color=color, linewidth=2,
                label=f"{algo_label} (n={n_seeds})")
        ax.fill_between(episodes, smoothed_ci_low, smoothed_ci_high,
                        color=color, alpha=ci_alpha)
        color_idx += 1

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# (f) 综合大图 — 2×3 面板
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_grand_figure(
    algo_names: List[str],
    avg_rewards: List[float],
    avg_queues: List[float],
    avg_waits: List[float],
    avg_throughput: List[float],
    training_curves: Optional[Dict[str, np.ndarray]] = None,
    save_path: str = "results/summary_grand_figure.png",
    title: str = "Comprehensive Algorithm Comparison",
) -> str:
    """
    2×3 综合大图:
      (1) Bar: Avg Reward
      (2) Bar: Avg Queue
      (3) Bar: Avg Wait
      (4) Training Curves (可选)
      (5) Bar: Throughput
      (6) Summary Table
    """
    setup_style()
    n_algos = len(algo_names)
    x = np.arange(n_algos)
    bar_colors = [AGENT_COLORS.get(a, COLORS["grey"]) for a in algo_names]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    # (1) Avg Reward
    ax = axes[0, 0]
    bars = ax.bar(x, avg_rewards, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg Reward")
    ax.set_title("Average Episode Reward")
    ax.grid(True, alpha=0.3, axis="y")

    # (2) Avg Queue
    ax = axes[0, 1]
    ax.bar(x, avg_queues, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg Queue Length")
    ax.set_title("Average Queue Length (lower=better)")
    ax.grid(True, alpha=0.3, axis="y")

    # (3) Avg Wait
    ax = axes[0, 2]
    ax.bar(x, avg_waits, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg Wait (steps)")
    ax.set_title("Average Wait Time (lower=better)")
    ax.grid(True, alpha=0.3, axis="y")

    # (4) Training Curves
    ax = axes[1, 0]
    if training_curves is not None:
        for i, (name, curve) in enumerate(training_curves.items()):
            color = AGENT_COLORS.get(name, COLORS["grey"])
            ax.plot(moving_average(curve, 30), color=color, linewidth=1.5, label=name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Curves (Smoothed)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "(No training data)", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["grey"])
        ax.set_title("Training Curves")

    # (5) Throughput
    ax = axes[1, 1]
    ax.bar(x, avg_throughput, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Total Throughput")
    ax.set_title("Total Vehicles Departed (higher=better)")
    ax.grid(True, alpha=0.3, axis="y")

    # (6) Summary Table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    col_labels = ["Algorithm", "Reward", "Queue", "Wait", "Thruput"]
    for i, name in enumerate(algo_names):
        table_data.append([
            name,
            f"{avg_rewards[i]:.1f}",
            f"{avg_queues[i]:.2f}",
            f"{avg_waits[i]:.2f}",
            f"{avg_throughput[i]:.0f}",
        ])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title("Numerical Summary")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# (g) 策略熵演化 (Policy Gradient 专用)
# ══════════════════════════════════════════════════════════════════════════════

def plot_policy_evolution(
    agent,
    env,
    episodes: int = 200,
    max_steps: int = 500,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    记录 PG Agent 策略熵随训练变化。

    策略熵 H(π(·|s)) = -Σ π(a|s) log π(a|s)
    高熵 = 探索, 低熵 = 利用。

    Returns:
        save_path 或空字符串
    """
    import torch
    from torch.distributions import Categorical

    entropy_history = []
    reward_history = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        ep_entropy = 0.0
        steps = 0
        for _ in range(max_steps):
            st = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                logits = agent.actor(st)
                dist = Categorical(logits=logits)
                ep_entropy += float(dist.entropy().item())

            action = agent.select_action(obs)
            ns, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store_transition(obs, action, rew, ns, done)
            ep_r += rew
            obs = ns
            steps += 1
            if done:
                break

        agent.update()
        entropy_history.append(ep_entropy / max(steps, 1))
        reward_history.append(ep_r)

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    # (1) 策略熵
    ax = axes[0]
    ax.plot(entropy_history, color=COLORS["orange"], alpha=0.8, linewidth=0.8)
    ax.plot(moving_average(np.array(entropy_history), 10),
            color=COLORS["red"], linewidth=1.5, label="MA(10)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Entropy H(π)")
    ax.set_title("Policy Entropy Evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) 奖励
    ax = axes[1]
    ax.plot(reward_history, color=COLORS["blue"], alpha=0.4, linewidth=0.6)
    ax.plot(moving_average(np.array(reward_history), 10),
            color=COLORS["blue"], linewidth=1.5, label="MA(10)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Figure saved: {save_path}")
        return save_path
    plt.close(fig)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# 辅助: 柱状图对比 (消融实验)
# ══════════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(
    labels: List[str],
    values: List[float],
    errors: Optional[List[float]] = None,
    save_path: str = "results/bar_comparison.png",
    title: str = "Performance Comparison",
    ylabel: str = "Avg Episode Reward",
    color: str = COLORS["blue"],
) -> str:
    """柱状图 (含误差棒)。"""
    setup_style()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x, values, color=color, edgecolor="white", linewidth=0.8,
           yerr=errors, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # 数值标注
    for i, v in enumerate(values):
        ax.text(i, v + (errors[i] if errors else 0) + max(values) * 0.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 延迟导入 (避免在无 torch 环境下崩溃)
# ══════════════════════════════════════════════════════════════════════════════

def _import_torch():
    import torch
    return torch


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    print("=" * 60)
    print("Visualization Module Tests")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    # ── Test (a): Dashboard ──
    print("\n--- (a) Training Dashboard ---")
    rng = np.random.default_rng(42)
    fake_rewards = np.cumsum(rng.normal(0, 1, 200)) + np.arange(200) * 0.05 - 0 * np.arange(200)**1.1
    fake_queue = np.maximum(30 - np.arange(200) * 0.1 + rng.normal(0, 2, 200), 0)
    fake_wait = np.maximum(50 - np.arange(200) * 0.2 + rng.normal(0, 3, 200), 0)
    fake_loss = np.maximum(2 * np.exp(-np.arange(500) * 0.005) + rng.normal(0, 0.05, 500), 0.001)
    path = plot_training_dashboard(fake_rewards, fake_queue, fake_wait, fake_loss,
                                   "results/test_dashboard.png")
    print(f"  Saved: {path}")

    # ── Test (b): Q Heatmap ──
    print("\n--- (b) Q-Value Heatmap ---")
    from traffic_env import make_env
    from agent import create_agent
    env = make_env(max_steps=200)
    agent = create_agent("dueling", state_dim=8)
    # quick train
    obs, _ = env.reset()
    for _ in range(200):
        a = agent.select_action(obs)
        ns, r, t, tr, _ = env.step(a)
        agent.store_transition(obs, a, r, ns, t or tr)
        if t or tr:
            obs, _ = env.reset()
        else:
            obs = ns
        agent.update()
    path = plot_q_value_heatmap(agent, env, "results/test_q_heatmap.png", n_samples=200)
    print(f"  Saved: {path}")

    # ── Test (c): Phase Timeline ──
    print("\n--- (c) Phase Decision Timeline ---")
    env2 = make_env(traffic_pattern="tidal", max_steps=200)
    obs, _ = env2.reset()
    q_hist, a_hist, p_hist, r_hist = [], [], [], []
    for _ in range(200):
        q_hist.append([env2.queues[i].length for i in range(4)])
        a = env2.action_space.sample()
        ns, _, t, tr, _ = env2.step(a)
        a_hist.append(a)
        p_hist.append(env2.current_phase)
        r_hist.append(env2.arrival_rates.copy())
        if t or tr:
            break
    path = plot_phase_decision_timeline(
        np.array(q_hist), np.array(a_hist), np.array(p_hist),
        np.array(r_hist), "results/test_timeline.png")
    print(f"  Saved: {path}")

    # ── Test (d): Radar Chart ──
    print("\n--- (d) Radar Chart ---")
    fake_metrics = {
        "DQN": {"Avg Reward": 0.55, "Queue ↓": 0.62, "Wait ↓": 0.58,
                "Throughput": 0.70, "Stability": 0.50, "Efficiency": 0.60},
        "DuelingDQN": {"Avg Reward": 0.72, "Queue ↓": 0.78, "Wait ↓": 0.75,
                       "Throughput": 0.82, "Stability": 0.74, "Efficiency": 0.80},
        "PER_DQN": {"Avg Reward": 0.68, "Queue ↓": 0.73, "Wait ↓": 0.70,
                    "Throughput": 0.79, "Stability": 0.65, "Efficiency": 0.73},
    }
    path = plot_radar_chart(fake_metrics, "results/test_radar.png")
    print(f"  Saved: {path}")

    # ── Test (e): Multi-Seed Ribbon ──
    print("\n--- (e) Multi-Seed Ribbon ---")
    fake_seeds = {
        "DuelingDQN": np.array([np.sin(np.linspace(0, 3*np.pi, 200)) * 500 - i * 100
                                + rng.normal(0, 30, 200) for i in range(5)]),
        "DQN": np.array([np.sin(np.linspace(0, 3*np.pi, 200)) * 400 - i * 120
                         + rng.normal(0, 40, 200) for i in range(5)]),
    }
    path = plot_multi_seed_ribbon(fake_seeds, "results/test_ribbon.png")
    print(f"  Saved: {path}")

    # ── Test (f): Grand Figure ──
    print("\n--- (f) Summary Grand Figure ---")
    names = ["DQN", "DoubleDQN", "DuelingDQN", "NoisyDQN", "PER_DQN"]
    rewards = [-8000, -7200, -5800, -6200, -5500]
    queues = [8.5, 7.8, 5.2, 6.4, 4.8]
    waits = [45, 38, 28, 32, 25]
    throughput = [850, 920, 1050, 980, 1100]
    fake_curves = {
        n: np.cumsum(rng.normal(0, 1, 200)) + np.arange(200) * (0.3 + i * 0.1) - 4000
        for i, n in enumerate(names)
    }
    path = plot_summary_grand_figure(
        names, rewards, queues, waits, throughput,
        fake_curves, "results/test_grand_figure.png")
    print(f"  Saved: {path}")

    print("\n✅ All visualization tests passed!")
