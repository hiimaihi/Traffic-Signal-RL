"""
交通信号灯自适应控制 — 强化学习算法核心逻辑
==============================================
实现:

  - ReplayBuffer:              经验回放池 (均匀随机采样)
  - PrioritizedReplayBuffer:   优先经验回放 (SumTree, PER)
  - SumTree:                   线段树数据结构 (O(log N) 采样/更新)
  - DQNAgent:                  标准 DQN (Vanilla DQN)
  - DoubleDQNAgent:            Double DQN (解耦动作选择与价值评估)
  - DuelingDQNAgent:           Dueling DQN (V/A 分离架构 + Double DQN 更新)
  - NoisyDQNAgent:             NoisyNet 探索 (自学习噪声)
  - BoltzmannDQNAgent:         Boltzmann/softmax 探索 (温度衰减)
  - PERDQNAgent:               优先经验回放 DQN
  - DuelingPERDQNAgent:        Dueling + Double + PER (最强组合)

所有 Agent 统一接口: select_action() / store_transition() / update() / update_target()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import Optional, Tuple, List, Deque
import random

# 数据类型
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ══════════════════════════════════════════════════════════════════════════════
# SumTree — 线段树 (用于优先经验回放)
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    线段树数据结构, 支持 O(log N) 的加权采样和优先级更新。
    叶子节点存储每个 transition 的优先级 p_i^α,
    内部节点存储子树优先级之和, 供二分采样使用。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: Transition) -> int:
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def get(self, s: float) -> Tuple[int, float, Transition]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    def __len__(self) -> int:
        return self.n_entries


# ══════════════════════════════════════════════════════════════════════════════
# 经验回放池 — 均匀采样
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """固定容量的经验回放池, 采用均匀随机采样。"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states      = torch.FloatTensor(np.array([t.state for t in batch]))
        actions     = torch.LongTensor(np.array([t.action for t in batch])).unsqueeze(1)
        rewards     = torch.FloatTensor(np.array([t.reward for t in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones       = torch.FloatTensor(np.array([t.done for t in batch])).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


# ══════════════════════════════════════════════════════════════════════════════
# 优先经验回放池 (PER)
# ══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """
    优先经验回放 (Schaul et al., 2016).
    P(i) ∝ p_i^α / Σ p_j^α,  p_i = |TD_error| + ε.
    IS 权重: w_i = (N·P(i))^{-β} / max_j w_j
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_steps: int = 50000, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        transition = Transition(state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch_size = min(batch_size, len(self.tree))
        batch, indices, priorities = [], [], []
        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, transition = self.tree.get(s)
            batch.append(transition)
            indices.append(idx)
            priorities.append(priority)

        sampling_probs = np.array(priorities) / self.tree.total_priority
        weights = (len(self.tree) * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(self.beta_end,
                        self.beta + (self.beta_end - self.beta_start) / self.beta_anneal_steps)

        states      = torch.FloatTensor(np.array([t.state for t in batch]))
        actions     = torch.LongTensor(np.array([t.action for t in batch])).unsqueeze(1)
        rewards     = torch.FloatTensor(np.array([t.reward for t in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones       = torch.FloatTensor(np.array([t.done for t in batch])).unsqueeze(1)
        weights     = torch.FloatTensor(weights).unsqueeze(1)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.tree)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.tree) >= batch_size


# ══════════════════════════════════════════════════════════════════════════════
# Base Agent — 公共逻辑
# ══════════════════════════════════════════════════════════════════════════════

class BaseAgent:
    """
    DQN 系列算法的抽象基类。
    支持三种探索策略: "epsilon" | "boltzmann" | "noisy"
    支持 PER (use_per=True)
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 2,
                 hidden_dim: int = 128, lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 1000,
                 exploration_type: str = "epsilon",
                 temperature_start: float = 10.0, temperature_end: float = 0.1,
                 temperature_decay: float = 1000,
                 tau: float = 0.005, buffer_capacity: int = 10000,
                 batch_size: int = 64, use_per: bool = False,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0,
                 device: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_type = exploration_type

        self.device = torch.device(device) if device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 探索参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.temperature = temperature_start
        self.train_steps = 0

        # 网络
        self.online_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # 回放池
        self.use_per = use_per
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha,
                beta_start=per_beta_start, beta_end=per_beta_end)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.loss_history: List[float] = []

    # ── 子类覆盖 ──
    def _build_network(self) -> nn.Module:
        raise NotImplementedError

    def _compute_targets(self, next_states: torch.Tensor, rewards: torch.Tensor,
                         dones: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ── 动作选择 ──
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if evaluate:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return int(self.online_net(st).argmax(dim=1).item())
        if self.exploration_type == "epsilon":
            return self._select_epsilon_greedy(state)
        elif self.exploration_type == "boltzmann":
            return self._select_boltzmann(state)
        elif self.exploration_type == "noisy":
            return self._select_noisy(state)
        return self._select_epsilon_greedy(state)

    def _select_epsilon_greedy(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(st).argmax(dim=1).item())

    def _select_boltzmann(self, state: np.ndarray) -> int:
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(st)
            probs = F.softmax(q / self.temperature, dim=1).cpu().numpy().flatten()
            return int(np.random.choice(self.action_dim, p=probs))

    def _select_noisy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(st).argmax(dim=1).item())

    # ── 存储 & 更新 ──
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        return self._update_per() if self.use_per else self._update_uniform()

    def _update_uniform(self) -> Optional[float]:
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        return self._do_update(states, actions, rewards, next_states, dones)

    def _update_per(self) -> Optional[float]:
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device); actions = actions.to(self.device)
        rewards = rewards.to(self.device); next_states = next_states.to(self.device)
        dones = dones.to(self.device); weights = weights.to(self.device)

        current_q = self.online_net(states).gather(1, actions)
        target_q = self._compute_targets(next_states, rewards, dones)
        td_errors = (target_q - current_q).detach().cpu().numpy().flatten()

        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors)
        self._soft_update_target()
        self._decay_exploration()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self.train_steps += 1
        return loss_val

    def _do_update(self, states, actions, rewards, next_states, dones) -> float:
        states = states.to(self.device); actions = actions.to(self.device)
        rewards = rewards.to(self.device); next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.online_net(states).gather(1, actions)
        target_q = self._compute_targets(next_states, rewards, dones)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._soft_update_target()
        self._decay_exploration()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self.train_steps += 1
        return loss_val

    # ── 目标网络更新 ──
    def _soft_update_target(self) -> None:
        for tp, op in zip(self.target_net.parameters(),
                          self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def update_target_hard(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── 探索衰减 ──
    def _decay_exploration(self) -> None:
        if self.exploration_type == "epsilon":
            self._decay_epsilon()
        elif self.exploration_type == "boltzmann":
            self._decay_temperature()

    def _decay_epsilon(self) -> None:
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
            * np.exp(-self.train_steps / self.epsilon_decay)

    def _decay_temperature(self) -> None:
        self.temperature = self.temperature_end + \
            (self.temperature_start - self.temperature_end) \
            * np.exp(-self.train_steps / self.temperature_decay)

    # ── 保存/加载 ──
    def save(self, path: str) -> None:
        torch.save({"online_net": self.online_net.state_dict(),
                     "target_net": self.target_net.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "train_steps": self.train_steps,
                     "epsilon": self.epsilon,
                     "temperature": self.temperature}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_steps = ckpt["train_steps"]
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.temperature = ckpt.get("temperature", self.temperature)


# ══════════════════════════════════════════════════════════════════════════════
# Vanilla DQN
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent(BaseAgent):
    """标准 DQN.  TD target: y = r + γ·max_a' Q_target(s', a')"""

    def _build_network(self) -> nn.Module:
        from network import StandardDQN
        return StandardDQN(input_dim=self.state_dim, output_dim=self.action_dim)

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            return rewards + self.gamma * next_q * (1.0 - dones)


# ══════════════════════════════════════════════════════════════════════════════
# Double DQN
# ══════════════════════════════════════════════════════════════════════════════

class DoubleDQNAgent(BaseAgent):
    """Double DQN: a*=argmax_a Q_online(s',a); y=r+γ·Q_target(s',a*)"""

    def _build_network(self) -> nn.Module:
        from network import StandardDQN
        return StandardDQN(input_dim=self.state_dim, output_dim=self.action_dim)

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            best = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best)
            return rewards + self.gamma * next_q * (1.0 - dones)


# ══════════════════════════════════════════════════════════════════════════════
# Dueling DQN
# ══════════════════════════════════════════════════════════════════════════════

class DuelingDQNAgent(BaseAgent):
    """Dueling DQN (V/A split) + Double DQN target"""

    def __init__(self, shared_hidden: int = 128, stream_hidden: int = 64, **kwargs):
        self.shared_hidden = shared_hidden
        self.stream_hidden = stream_hidden
        super().__init__(**kwargs)

    def _build_network(self) -> nn.Module:
        from network import DuelingDQN
        return DuelingDQN(input_dim=self.state_dim, output_dim=self.action_dim,
                          shared_hidden=self.shared_hidden,
                          stream_hidden=self.stream_hidden)

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            best = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best)
            return rewards + self.gamma * next_q * (1.0 - dones)


# ══════════════════════════════════════════════════════════════════════════════
# NoisyNet DQN
# ══════════════════════════════════════════════════════════════════════════════

class NoisyDQNAgent(BaseAgent):
    """NoisyNet DQN — 使用可学习噪声替代 ε-greedy"""

    def _build_network(self) -> nn.Module:
        from network import NoisyDQN
        return NoisyDQN(input_dim=self.state_dim, output_dim=self.action_dim)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        self.online_net.sample_noise()
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                q = self.online_net.get_mean_q(st)
            else:
                q = self.online_net(st)
            return int(q.argmax(dim=1).item())

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            return rewards + self.gamma * next_q * (1.0 - dones)

    def _decay_exploration(self) -> None:
        pass  # NoisyNet 无需手动衰减


# ══════════════════════════════════════════════════════════════════════════════
# NoisyNet + Double DQN
# ══════════════════════════════════════════════════════════════════════════════

class NoisyDoubleDQNAgent(NoisyDQNAgent):
    """NoisyNet + Double DQN 结合"""

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            best = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best)
            return rewards + self.gamma * next_q * (1.0 - dones)


# ══════════════════════════════════════════════════════════════════════════════
# Boltzmann DQN
# ══════════════════════════════════════════════════════════════════════════════

class BoltzmannDQNAgent(BaseAgent):
    """Boltzmann/Softmax 探索 — P(a|s) ∝ exp(Q(s,a)/T)"""

    def __init__(self, **kwargs):
        kwargs.setdefault("exploration_type", "boltzmann")
        super().__init__(**kwargs)

    def _build_network(self) -> nn.Module:
        from network import StandardDQN
        return StandardDQN(input_dim=self.state_dim, output_dim=self.action_dim)

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            return rewards + self.gamma * next_q * (1.0 - dones)


# ══════════════════════════════════════════════════════════════════════════════
# Boltzmann + Double DQN
# ══════════════════════════════════════════════════════════════════════════════

class BoltzmannDoubleDQNAgent(BoltzmannDQNAgent):
    """Boltzmann 探索 + Double DQN"""

    def _compute_targets(self, next_states, rewards, dones):
        with torch.no_grad():
            best = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best)
            return rewards + self.gamma * next_q * (1.0 - dones)


# ══════════════════════════════════════════════════════════════════════════════
# PER 变体
# ══════════════════════════════════════════════════════════════════════════════

class PERDQNAgent(DoubleDQNAgent):
    """Double DQN + 优先经验回放"""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_per", True)
        super().__init__(**kwargs)


class DuelingPERDQNAgent(DuelingDQNAgent):
    """Dueling DQN + Double + PER"""
    def __init__(self, **kwargs):
        kwargs.setdefault("use_per", True)
        super().__init__(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 工厂
# ══════════════════════════════════════════════════════════════════════════════

def create_agent(agent_type: str = "dueling", state_dim: int = 8,
                 action_dim: int = 2, **kwargs) -> BaseAgent:
    agent_map = {
        "dqn": DQNAgent,
        "double": DoubleDQNAgent,
        "dueling": DuelingDQNAgent,
        "noisy": NoisyDQNAgent,
        "noisy_double": NoisyDoubleDQNAgent,
        "boltzmann": BoltzmannDQNAgent,
        "boltzmann_double": BoltzmannDoubleDQNAgent,
        "per_dqn": PERDQNAgent,
        "dueling_per": DuelingPERDQNAgent,
    }
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent_type: {agent_type}. "
                         f"Choose from {list(agent_map.keys())}.")
    return agent_map[agent_type](state_dim=state_dim, action_dim=action_dim, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Agent Tests")
    print("=" * 60)

    # SumTree + PER 测试
    print("\n--- SumTree + PrioritizedReplayBuffer ---")
    per_buf = PrioritizedReplayBuffer(capacity=128, alpha=0.6)
    dummy = np.random.randn(8).astype(np.float32)
    for i in range(200):
        per_buf.push(dummy, random.randrange(2), -random.random(), dummy, False)
    print(f"  PER buffer: {len(per_buf)} entries, total_priority={per_buf.tree.total_priority:.3f}")
    s, a, r, ns, d, w, idxs = per_buf.sample(32)
    print(f"  PER sample: s={s.shape}, w={w.shape}, indices={len(idxs)}")

    # 均匀回放池
    buf = ReplayBuffer(capacity=100)
    for i in range(200):
        buf.push(dummy, random.randrange(2), random.random(), dummy, random.random() > 0.9)
    print(f"\n  ReplayBuffer: {len(buf)} transitions (cap=100)")
    s, a, r, ns, d = buf.sample(32)
    print(f"  Sample: s={s.shape}, a={a.shape}")

    # Agent 冒烟测试
    dummy_state = np.random.randn(8).astype(np.float32)
    for agent_type in ["dqn", "double", "dueling", "noisy", "noisy_double",
                       "boltzmann", "boltzmann_double", "per_dqn", "dueling_per"]:
        agent = create_agent(agent_type=agent_type)
        n_params = sum(p.numel() for p in agent.online_net.parameters())
        action = agent.select_action(dummy_state)
        for _ in range(100):
            agent.store_transition(dummy_state, random.randrange(2), -1.0, dummy_state, False)
        loss = agent.update()
        loss_str = f"{loss:.4f}" if loss is not None else "None"
        explore = agent.epsilon if agent.exploration_type == "epsilon" else \
                  agent.temperature if agent.exploration_type == "boltzmann" else "noisy"
        print(f"  [{agent_type:18s}] params={n_params:>6,d} | action={action} | "
              f"loss={loss_str:>8s} | explore={explore}")

    print("\n✅ All agent tests passed!")
