"""Q 网络架构：标准 DQN、Dueling、NoisyNet 及消融变体。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


# ──────────────────────────────────────────────────────────────────────────────
# Factorised NoisyLinear — 用于 NoisyNet 探索
# ──────────────────────────────────────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """
    Factorised Gaussian Noise 线性层 (Fortunato et al., 2018).

    参数化:
      y = (μ_w + σ_w ⊙ ε_w) @ x + (μ_b + σ_b ⊙ ε_b)

    其中 ε_w, ε_b 是独立因式化高斯噪声:
      ε_w = f(ε_i) ⊗ f(ε_j)   (outer product of factorised noise)
      ε_b = f(ε_j)

    f(x) = sgn(x) * sqrt(|x|)
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习参数: 均值 μ 和标准差 σ
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # 注册噪声缓冲区 (不参与反向传播)
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self) -> None:
        """初始化参数。"""
        # μ 使用均匀分布
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # σ 初始化为 sigma_init / sqrt(in_features)
        sigma_init_val = self.sigma_init / math.sqrt(self.in_features)
        nn.init.constant_(self.weight_sigma, sigma_init_val)
        nn.init.constant_(self.bias_sigma, sigma_init_val)

    def sample_noise(self) -> None:
        """采样因式化高斯噪声 — 每次前向传播前调用。"""
        in_noise = self._factorised_noise(self.in_features)
        out_noise = self._factorised_noise(self.out_features)

        # weight_epsilon: outer product of f(e_out) ⊗ f(e_in)
        self.weight_epsilon.copy_(
            torch.ger(out_noise, in_noise)
        )
        # bias_epsilon: f(e_out)
        self.bias_epsilon.copy_(out_noise)

    @staticmethod
    def _factorised_noise(size: int) -> torch.Tensor:
        """f(x) = sgn(x) * sqrt(|x|), x ~ N(0, 1)"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: y = (μ_w + σ_w ⊙ ε_w)x + (μ_b + σ_b ⊙ ε_b)"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ──────────────────────────────────────────────────────────────────────────────
# 标准 DQN 网络
# ──────────────────────────────────────────────────────────────────────────────

class StandardDQN(nn.Module):
    """
    标准全连接 Q 网络。

    架构:
      Input(state_dim) → FC(hidden) → ReLU → FC(hidden) → ReLU → FC(2)

    输出: Q(s, a) for a ∈ {0, 1}
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 2,
        hidden_dim: int = 128,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _init_weights(self) -> None:
        """Xavier 初始化 + 最后一层偏置置零。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# Dueling DQN 网络
# ──────────────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Dueling DQN — 分离状态价值 V(s) 与动作优势 A(s, a)。

    架构:
      共享层:
        Input(state_dim) → FC(128) → ReLU → FC(128) → ReLU
      价值流 (Value Stream):
        FC(64) → ReLU → FC(1) → V(s)
      优势流 (Advantage Stream):
        FC(64) → ReLU → FC(2) → A(s, a)
      合并:
        Q(s, a) = V(s) + [A(s, a) - mean(A(s, :))]

    这种设计允许网络区分"状态本身的好坏"与"某个动作的相对优劣",
    在交通场景中尤为有效 (路口拥堵程度 vs. 是否切换相位的边际收益)。
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 2,
        shared_hidden: int = 128,
        stream_hidden: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # ── 共享特征提取层 ──
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, shared_hidden),
            nn.ReLU(),
        )

        # ── 价值流 V(s) ──
        self.value_stream = nn.Sequential(
            nn.Linear(shared_hidden, stream_hidden),
            nn.ReLU(),
            nn.Linear(stream_hidden, 1),
        )

        # ── 优势流 A(s, a) ──
        self.advantage_stream = nn.Sequential(
            nn.Linear(shared_hidden, stream_hidden),
            nn.ReLU(),
            nn.Linear(stream_hidden, output_dim),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)

        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, output_dim)

        # Q(s, a) = V(s) + [A(s, a) - mean(A)]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# Noisy DQN 网络 (NoisyNet)
# ──────────────────────────────────────────────────────────────────────────────

class NoisyDQN(nn.Module):
    """
    NoisyNet — 使用可学习噪声替代 ε-greedy 探索。

    架构 (与 StandardDQN 相同, 但使用 NoisyLinear):
      Input(state_dim) → NoisyLinear(hidden) → ReLU → NoisyLinear(hidden) → ReLU → NoisyLinear(2)

    噪声层自带探索: 每次前向传播自动采样因式化高斯噪声,
    训练中无须手动设置 ε, 噪声强度由网络自适应学习。
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 2,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for i in range(num_hidden):
            layers.append(NoisyLinear(in_dim, hidden_dim, sigma_init=sigma_init))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(NoisyLinear(in_dim, output_dim, sigma_init=sigma_init))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def sample_noise(self) -> None:
        """采样所有 NoisyLinear 层的噪声。"""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

    def get_mean_q(self, x: torch.Tensor) -> torch.Tensor:
        """获取仅基于均值权重的 Q 值 (评估时使用)。"""
        with torch.no_grad():
            # 临时切换到 eval 模式以使用均值
            was_training = self.training
            self.eval()
            q = self.forward(x)
            if was_training:
                self.train()
            return q


# ──────────────────────────────────────────────────────────────────────────────
# 浅层 DQN (消融用)
# ──────────────────────────────────────────────────────────────────────────────

class ShallowDQN(nn.Module):
    """
    浅层网络 — 仅 1 隐藏层 FC(32), ~2.6K 参数。

    用于消融实验: 验证网络深度对性能的影响。
    架构:
      Input(state_dim) → FC(32) → ReLU → FC(2)
    """

    def __init__(self, input_dim: int = 8, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# 深层 DQN (消融用)
# ──────────────────────────────────────────────────────────────────────────────

class DeepDQN(nn.Module):
    """
    深层网络 — 4 隐藏层 FC(256), ~67K 参数。

    用于消融实验: 验证更深网络是否能学到更好的表示。
    架构:
      Input(state_dim) → 4× [FC(256) → ReLU] → FC(2)
    """

    def __init__(self, input_dim: int = 8, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for _ in range(4):
            layers.append(nn.Linear(in_dim, 256))
            layers.append(nn.ReLU())
            in_dim = 256
        layers.append(nn.Linear(256, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# Policy Gradient 网络 — Actor (Policy) & Critic (Value)
# ──────────────────────────────────────────────────────────────────────────────

class ActorNetwork(nn.Module):
    """
    Policy 网络 — 输出离散动作的 logits。

    架构:
      Input(state_dim) → FC(128) → ReLU → FC(128) → ReLU → FC(action_dim)

    输出 logits, 经 softmax 后形成 Categorical 分布。
    """

    def __init__(self, input_dim: int = 8, output_dim: int = 2,
                 hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 logits"""
        return self.network(x)

    def get_action_distribution(self, x: torch.Tensor) -> torch.distributions.Categorical:
        """返回 Categorical 分布对象"""
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CriticNetwork(nn.Module):
    """
    Value 网络 — 输出标量 V(s)。

    架构:
      Input(state_dim) → FC(128) → ReLU → FC(128) → ReLU → FC(1)

    用于 PPO/A2C 的 baseline / advantage 估计。
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 V(s), shape [batch, 1]"""
        return self.network(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# 网络工厂函数
# ──────────────────────────────────────────────────────────────────────────────

def create_network(
    network_type: str = "dueling",
    input_dim: int = 8,
    output_dim: int = 2,
    **kwargs,
) -> nn.Module:
    """
    网络工厂: 根据类型字符串创建对应的网络。

    Args:
        network_type: "standard" | "dueling" | "double" | "noisy" | "shallow" | "deep"
                      | "actor" | "critic"
                      (double 使用 StandardDQN, 算法差异在 Agent 层)
        input_dim: 输入维度
        output_dim: 输出维度 (actor=action_dim, critic 忽略)
        **kwargs: 传递给具体网络类

    Returns:
        nn.Module 实例
    """
    if network_type == "dueling":
        return DuelingDQN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif network_type in ("standard", "double"):
        return StandardDQN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif network_type == "noisy":
        return NoisyDQN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif network_type == "shallow":
        return ShallowDQN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif network_type == "deep":
        return DeepDQN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif network_type == "actor":
        return ActorNetwork(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif network_type == "critic":
        return CriticNetwork(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown network_type: {network_type}. "
                         f"Choose from 'standard', 'double', 'dueling', 'noisy', "
                         f"'shallow', 'deep', 'actor', 'critic'.")


# ──────────────────────────────────────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Network Architecture Tests")
    print("=" * 60)

    batch_size = 4
    dummy_input = torch.randn(batch_size, 8)

    for name in ["standard", "dueling"]:
        net = create_network(network_type=name)
        output = net(dummy_input)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"\n  [{name.upper()}]")
        print(f"    Input:  {dummy_input.shape}")
        print(f"    Output: {output.shape}")
        print(f"    Params: {n_params:,}")
        print(f"    Q-values:\n{output.detach().numpy()}")

    # 验证 Dueling 的 identifiability
    print("\n  [Dueling Identifiability Check]")
    dueling = DuelingDQN()
    x1 = torch.randn(1, 8)
    x2 = x1.clone()
    # 相同输入应产生相同输出
    q1 = dueling(x1)
    q2 = dueling(x2)
    assert torch.allclose(q1, q2), "Dueling output not deterministic!"
    print("    ✅ Output is deterministic for same input.")

    print("\n✅ All network tests passed!")
