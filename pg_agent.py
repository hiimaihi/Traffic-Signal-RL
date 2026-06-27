"""
交通信号灯自适应控制 — Policy Gradient 算法族
================================================
实现:
  - A2CAgent:    Advantage Actor-Critic (n-step TD + entropy bonus)
  - PPOAgent:    Proximal Policy Optimization (clip + value clipping)

统一接口: select_action() / store_transition() / update() (在 episode 结束时调用)
与 DQN 系列 Agent 的关键区别:
  - 在线学习 (on-policy): 每个 episode 收集 rollout → 计算 returns → 一次/多次梯度更新
  - 输出 Categorical 分布 (离散动作)
  - 使用 Actor-Critic 架构 (Actor=策略网络, Critic=价值网络)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, List, Tuple, Dict
import random


# ══════════════════════════════════════════════════════════════════════════════
# 通用 Rollout Buffer (on-policy)
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """存储一个 episode 的完整轨迹, 用于 on-policy 更新。"""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, state: np.ndarray, action: int, reward: float,
            log_prob: float, value: float, done: bool) -> None:
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def compute_returns(self, gamma: float, gae_lambda: float,
                        last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 discounted returns 和 GAE advantages。

        Returns:
            returns: [T+1] 含 bootstrap
            advantages: [T] GAE
        """
        T = len(self.rewards)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones + [False], dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return (torch.FloatTensor(returns),
                torch.FloatTensor(advantages))


# ══════════════════════════════════════════════════════════════════════════════
# A2C (Advantage Actor-Critic)
# ══════════════════════════════════════════════════════════════════════════════

class A2CAgent:
    """
    Advantage Actor-Critic (A2C).

    每个 episode 收集 rollout, 计算 GAE advantages + returns,
    对 Actor 做 policy gradient 更新, 对 Critic 做 MSE 回归。

    关键超参:
      - gamma: 折扣因子
      - gae_lambda: GAE λ 参数
      - entropy_coef: 熵正则系数 (鼓励探索)
      - value_coef: Critic loss 权重
      - lr_actor / lr_critic: 独立学习率
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 2,
                 hidden_dim: int = 128,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device) if device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from network import ActorNetwork, CriticNetwork
        self.actor = ActorNetwork(input_dim=state_dim, output_dim=action_dim,
                                  hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(input_dim=state_dim,
                                    hidden_dim=hidden_dim).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()
        self.loss_history: List[float] = []
        self.train_steps: int = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        A2C: 从 Categorical 分布中采样动作。

        训练时对每个动作记录 log_prob + value (存入 buffer)。
        评估时返回最大概率动作。
        """
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(st)
            value = self.critic(st)
            dist = Categorical(logits=logits)

        if evaluate:
            return int(logits.argmax(dim=1).item())

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item())

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """
        重算当前状态的 log_prob 和 value 存入 buffer。

        注: A2C 是 on-policy 方法, 必须在 select_action 时同步记录 log_prob/value。
        为与 DQN 接口兼容, 这里重新前向计算 (损失精度但保持接口统一)。
        """
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(st)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor([action], device=self.device))
            value = self.critic(st).item()

        self.buffer.add(state, action, reward, float(log_prob.item()), float(value), done)

    def update(self) -> float:
        """
        A2C 在 episode 结束时调用一次, 对整个 rollout 做参数更新。
        返回总 loss。
        """
        if len(self.buffer.rewards) == 0:
            return 0.0

        T = len(self.buffer.rewards)

        # 计算最后状态的 value (用于 GAE bootstrap)
        if self.buffer.dones[-1]:
            last_value = 0.0
        else:
            last_st = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.critic(last_st).item()

        returns, advantages = self.buffer.compute_returns(
            self.gamma, self.gae_lambda, last_value)

        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # ── Actor loss ──
        logits = self.actor(states_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages).mean()
        entropy_bonus = -self.entropy_coef * entropy
        actor_loss = policy_loss + entropy_bonus

        # ── Critic loss ──
        values = self.critic(states_t).squeeze(-1)
        critic_loss = self.value_coef * F.mse_loss(values, returns)

        # ── 总 loss ──
        total_loss = actor_loss + critic_loss

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.train_steps += 1
        self.buffer.clear()

        total_val = float(total_loss.item())
        self.loss_history.append(total_val)
        return total_val

    def update_target(self) -> None:
        """A2C 无需 target 网络。"""
        pass

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ══════════════════════════════════════════════════════════════════════════════
# PPO (Proximal Policy Optimization)
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) — Clip 版本。

    比 A2C 的核心改进:
      - 每个 episode 的 rollout 做 K 个 epoch 的 mini-batch 更新
      - clip ratio 限制策略更新幅度 (防止 catastrophic forgetting)
      - value clipping 防止价值函数过度更新

    关键超参:
      - clip_epsilon: PPO clip 范围
      - ppo_epochs: 每 rollout 训练的 epoch 数
      - mini_batch_size: mini-batch 大小
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 2,
                 hidden_dim: int = 128,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 lr: float = 3e-4, clip_epsilon: float = 0.2,
                 ppo_epochs: int = 10, mini_batch_size: int = 64,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device) if device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from network import ActorNetwork, CriticNetwork
        self.actor = ActorNetwork(input_dim=state_dim, output_dim=action_dim,
                                  hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(input_dim=state_dim,
                                    hidden_dim=hidden_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr)

        self.buffer = RolloutBuffer()
        self.loss_history: List[float] = []
        self.train_steps: int = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(st)
            dist = Categorical(logits=logits)

        if evaluate:
            return int(logits.argmax(dim=1).item())

        action = dist.sample()
        return int(action.item())

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(st)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor([action], device=self.device))
            value = self.critic(st).item()

        self.buffer.add(state, action, reward, float(log_prob.item()), float(value), done)

    def update(self) -> float:
        """
        PPO 在每个 episode 结束时更新, 对 rollout 做多 epoch 的 mini-batch 训练。
        """
        if len(self.buffer.rewards) == 0:
            return 0.0

        # 计算 GAE
        if self.buffer.dones[-1]:
            last_value = 0.0
        else:
            last_st = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.critic(last_st).item()

        returns, advantages = self.buffer.compute_returns(
            self.gamma, self.gae_lambda, last_value)

        # 标准化 advantages (稳定 Actor 梯度)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 标准化 returns (稳定 Critic 梯度, 防止初始化时 loss 爆炸)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        total_loss_accum = 0.0
        n_batches = 0
        dataset_size = len(self.buffer.rewards)

        for _ in range(self.ppo_epochs):
            # mini-batch 随机打乱
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.mini_batch_size):
                batch_idx = indices[start:start + self.mini_batch_size]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # ── Actor (PPO clip) loss ──
                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                   1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Critic (value clip) loss ──
                values = self.critic(batch_states).squeeze(-1)
                old_values = batch_returns - batch_advantages  # 近似

                v_clipped = old_values + torch.clamp(
                    values - old_values, -self.clip_epsilon, self.clip_epsilon)
                v_loss_unclipped = (values - batch_returns) ** 2
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # ── 总 loss ──
                total_loss = (policy_loss
                              + self.value_coef * value_loss
                              - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += float(total_loss.item())
                n_batches += 1

        self.train_steps += 1
        self.buffer.clear()

        avg_loss = total_loss_accum / max(n_batches, 1)
        self.loss_history.append(avg_loss)
        return avg_loss

    def update_target(self) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ══════════════════════════════════════════════════════════════════════════════
# Agent 工厂
# ══════════════════════════════════════════════════════════════════════════════

def create_pg_agent(agent_type: str = "ppo", state_dim: int = 8,
                    action_dim: int = 2, **kwargs):
    """
    Policy Gradient Agent 工厂。

    Args:
        agent_type: "a2c" | "ppo"
        state_dim: 状态维度
        action_dim: 动作维度
        **kwargs: 额外超参

    Returns:
        A2CAgent 或 PPOAgent 实例
    """
    pg_map = {
        "a2c": A2CAgent,
        "ppo": PPOAgent,
    }
    if agent_type not in pg_map:
        raise ValueError(f"Unknown PG agent_type: {agent_type}. "
                         f"Choose from {list(pg_map.keys())}.")
    return pg_map[agent_type](state_dim=state_dim, action_dim=action_dim, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Policy Gradient Agent Tests")
    print("=" * 60)

    dummy_state = np.random.randn(8).astype(np.float32)
    action_dim = 2

    # ── A2C ──
    print("\n--- A2C Agent ---")
    a2c = A2CAgent(state_dim=8, action_dim=action_dim)
    print(f"  Actor params: {sum(p.numel() for p in a2c.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in a2c.critic.parameters()):,}")

    # 模拟一个 episode
    for i in range(64):
        action = a2c.select_action(dummy_state)
        a2c.store_transition(dummy_state, action, -10.0, dummy_state, i == 63)
    loss = a2c.update()
    print(f"  Episode loss: {loss:.4f}")
    print(f"  Buffer cleared: {len(a2c.buffer.rewards) == 0}")

    # ── PPO ──
    print("\n--- PPO Agent ---")
    ppo = PPOAgent(state_dim=8, action_dim=action_dim)
    print(f"  Actor params: {sum(p.numel() for p in ppo.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in ppo.critic.parameters()):,}")

    for i in range(200):
        action = ppo.select_action(dummy_state)
        ppo.store_transition(dummy_state, action, -10.0, dummy_state, i == 199)
    loss = ppo.update()
    print(f"  Episode loss: {loss:.4f}")
    print(f"  Buffer cleared: {len(ppo.buffer.rewards) == 0}")

    # ── 工厂 ──
    print("\n--- Factory ---")
    for at in ["a2c", "ppo"]:
        agent = create_pg_agent(agent_type=at, state_dim=8, action_dim=2)
        print(f"  {at.upper()}: created, type={type(agent).__name__}")

    # ── 评估模式 ──
    print("\n--- Evaluation Mode ---")
    a2c_eval_action = a2c.select_action(dummy_state, evaluate=True)
    ppo_eval_action = ppo.select_action(dummy_state, evaluate=True)
    print(f"  A2C eval action: {a2c_eval_action}")
    print(f"  PPO eval action: {ppo_eval_action}")

    print("\n✅ All PG agent tests passed!")
