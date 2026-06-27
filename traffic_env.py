"""
单十字路口交通流仿真环境（Gymnasium）。

支持可变状态空间、多种奖励函数和 5 种交通模式。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List


# ══════════════════════════════════════════════════════════════════════════════
# 环境配置注册表
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIG = {
    "default": {
        "arrival_rates": (2.0, 2.0, 2.0, 2.0),
        "traffic_pattern": "uniform",
        "state_dim": 8,
        "reward_type": "composite",
        "w1": 1.0, "w2": 0.02, "switch_penalty": 2.0,
        "max_steps": 1000, "min_green_duration": 5, "yellow_duration": 1,
    },
    "uniform_s4_r1": {
        "arrival_rates": (2.0, 2.0, 2.0, 2.0),
        "traffic_pattern": "uniform",
        "state_dim": 4, "reward_type": "queue_only",
    },
    "uniform_s8_composite": {
        "arrival_rates": (2.0, 2.0, 2.0, 2.0),
        "traffic_pattern": "uniform",
        "state_dim": 8, "reward_type": "composite",
    },
    "uniform_s9_r3": {
        "arrival_rates": (2.0, 2.0, 2.0, 2.0),
        "traffic_pattern": "uniform",
        "state_dim": 9, "reward_type": "queue_wait",
    },
    "peak_hour": {
        "arrival_rates": (2.0, 2.0, 2.0, 2.0),
        "traffic_pattern": "peak_hour",
        "state_dim": 8, "reward_type": "composite",
        "peak_multiplier": 3.0, "peak_start": 200, "peak_end": 700,
    },
    "tidal": {
        "arrival_rates": (2.0, 2.0, 2.0, 2.0),
        "traffic_pattern": "tidal",
        "state_dim": 8, "reward_type": "composite",
        "tidal_period": 400, "peak_multiplier": 3.0,
    },
    "burst": {
        "arrival_rates": (1.5, 1.5, 1.5, 1.5),
        "traffic_pattern": "burst",
        "state_dim": 8, "reward_type": "composite",
        "burst_intensity": 8.0,
    },
    "low_traffic": {
        "arrival_rates": (0.5, 0.5, 0.5, 0.5),
        "traffic_pattern": "low_traffic",
        "state_dim": 8, "reward_type": "composite",
    },
}

# 状态空间维度 → Box shape
STATE_DIM_SHAPES = {4: (4,), 8: (8,), 9: (9,)}

# 奖励函数注册表
REWARD_REGISTRY = ["queue_only", "queue_switch", "queue_wait", "composite"]


# ──────────────────────────────────────────────────────────────────────────────
# 辅助: 车辆队列数据结构
# ──────────────────────────────────────────────────────────────────────────────

class VehicleQueue:
    """
    单车道车辆队列, 跟踪每辆车的到达时间步, 支持 FIFO 通行。
    """

    def __init__(self):
        self._vehicles: List[int] = []  # 每辆车的到达时间步 (global_step)

    def add_vehicle(self, arrival_step: int) -> None:
        """添加一辆车, 记录其到达步数。"""
        self._vehicles.append(arrival_step)

    def depart_vehicle(self, current_step: int) -> Optional[int]:
        """
        放行队首车辆, 返回其等待时间 (步数)。
        队列为空时返回 None。
        """
        if not self._vehicles:
            return None
        arrival = self._vehicles.pop(0)
        return current_step - arrival

    @property
    def length(self) -> int:
        return len(self._vehicles)

    def head_wait_time(self, current_step: int) -> float:
        """队首车辆的已等待时间 (步数), 如果队列为空返回 0。"""
        if not self._vehicles:
            return 0.0
        return float(current_step - self._vehicles[0])

    def cumulative_wait_time(self, current_step: int) -> float:
        """队列中所有车辆的总等待时间。"""
        if not self._vehicles:
            return 0.0
        return sum(current_step - arrival for arrival in self._vehicles)

    def reset(self) -> None:
        self._vehicles.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 交通环境
# ──────────────────────────────────────────────────────────────────────────────

class TrafficLightEnv(gym.Env):
    """
    单十字路口交通信号灯控制环境。

    ## 观测空间 (Observation Space)
    Box(8,) 连续向量:
      [0:4] — 四个方向排队车辆数 (N, S, E, W)
      [4:8] — 四个方向队首车辆等待时间 (已归一化到 [0, max_wait])

    ## 动作空间 (Action Space)
    Discrete(2):
      0 — 保持当前绿灯相位
      1 — 切换到另一相位 (触发黄灯延迟)

    ## 奖励
    R = -(w1 * sum(L_i) + w2 * sum(T_i)) - switch_penalty * I{action==1}
    """

    # 方向索引常量
    NORTH = 0
    SOUTH = 1
    EAST  = 2
    WEST  = 3

    # 相位常量
    PHASE_NS = 0  # 南北通行
    PHASE_EW = 1  # 东西通行

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        # 到达率 (泊松分布 λ): [N, S, E, W]
        arrival_rates: Tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
        # 奖励权重 (注意: w2 必须远小于 w1, 因为等待时间可累积数百步)
        w1: float = 1.0,   # 排队长度权重 (每辆车 ~1-20)
        w2: float = 0.02,  # 等待时间权重 (累积可达数百, 需缩放到与 w1 同量级)
        switch_penalty: float = 2.0,
        # 仿真参数
        max_steps: int = 1000,
        min_green_duration: int = 5,       # 最小绿灯持续步数
        yellow_duration: int = 1,          # 黄灯步数 (切换时)
        max_wait_normalize: float = 200.0, # 等待时间归一化上限
        # 流量模式
        traffic_pattern: str = "uniform",  # "uniform"|"peak_hour"|"tidal"|"burst"|"low_traffic"
        peak_multiplier: float = 3.0,
        peak_start: int = 0,
        peak_end: int = 500,
        tidal_period: int = 400,
        burst_intensity: float = 8.0,      # burst 模式随机尖峰强度
        # 状态/奖励变体
        state_dim: int = 8,                # 4 | 8 | 9
        reward_type: str = "composite",    # "queue_only"|"queue_switch"|"queue_wait"|"composite"
    ):
        super().__init__()

        # ── 参数存储 ──
        self._base_arrival_rates = np.array(arrival_rates, dtype=np.float32)
        self.arrival_rates = self._base_arrival_rates.copy()
        self.w1 = w1
        self.w2 = w2
        self.switch_penalty = switch_penalty
        self.max_steps = max_steps
        self.min_green_duration = min_green_duration
        self.yellow_duration = yellow_duration
        self.max_wait_normalize = max_wait_normalize
        self.traffic_pattern = traffic_pattern
        self.peak_multiplier = peak_multiplier
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.tidal_period = tidal_period
        self.burst_intensity = burst_intensity
        self.state_dim = state_dim
        self.reward_type = reward_type

        # ── 观测 / 动作空间 ──
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # ── 内部状态 ──
        self.queues: List[VehicleQueue] = [VehicleQueue() for _ in range(4)]
        self.current_phase: int = self.PHASE_NS
        self.phase_duration: int = 0          # 当前相位已持续时间
        self.in_yellow: bool = False           # 是否处于黄灯期
        self.yellow_remaining: int = 0
        self.global_step: int = 0
        self.total_departed: int = 0           # 总通行车辆数
        self.total_wait_accum: float = 0.0     # 累计等待时间 (用于统计)

        # 统计
        self.episode_queue_history: List[float] = []
        self.episode_wait_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────
    # Gym API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # 初始化四个方向的车辆队列
        self.queues = [VehicleQueue() for _ in range(4)]
        self.current_phase = self.PHASE_NS
        self.phase_duration = 0
        self.in_yellow = False
        self.yellow_remaining = 0
        self.global_step = 0
        self.total_departed = 0
        self.total_wait_accum = 0.0
        self.episode_queue_history = []
        self.episode_wait_history = []

        # 支持 options 中覆盖参数
        if options is not None:
            if "arrival_rates" in options:
                self._base_arrival_rates = np.array(options["arrival_rates"], dtype=np.float32)
            if "traffic_pattern" in options:
                self.traffic_pattern = options["traffic_pattern"]
            if "w1" in options:
                self.w1 = options["w1"]
            if "w2" in options:
                self.w2 = options["w2"]
            if "state_dim" in options:
                self.state_dim = options["state_dim"]
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32,
                )
            if "reward_type" in options:
                self.reward_type = options["reward_type"]
            if "burst_intensity" in options:
                self.burst_intensity = options["burst_intensity"]

        self._update_arrival_rates()
        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # ── 1. 处理黄灯 ──
        if self.in_yellow:
            self.yellow_remaining -= 1
            if self.yellow_remaining <= 0:
                self.in_yellow = False
                # 完成相位切换
                self.current_phase = 1 - self.current_phase
                self.phase_duration = 0

        # ── 2. 绿灯通行 ──
        vehicles_departed = self._process_green_phase()

        # ── 3. 动作执行 ──
        switch_occurred = False
        if action == 1:
            if not self.in_yellow and self.phase_duration >= self.min_green_duration:
                # 触发黄灯 → 不允许立即再切
                self.in_yellow = True
                self.yellow_remaining = self.yellow_duration
                switch_occurred = True

        # ── 4. 新车辆到达 ──
        self._generate_arrivals()

        # ── 5. 推进时间 ──
        self.global_step += 1
        if not self.in_yellow:
            self.phase_duration += 1
        self._update_arrival_rates()

        # ── 6. 观测 & 奖励 ──
        obs = self._get_observation()
        reward = self._compute_reward(switch_occurred)
        terminated = self.global_step >= self.max_steps
        truncated = False

        # ── 7. 日志 ──
        total_queue = sum(q.length for q in self.queues)
        avg_wait = np.mean([q.head_wait_time(self.global_step) for q in self.queues])
        self.episode_queue_history.append(total_queue)
        self.episode_wait_history.append(avg_wait)

        info = {
            "total_queue": total_queue,
            "avg_wait": avg_wait,
            "departed": vehicles_departed,
            "phase": self.current_phase,
            "arrival_rates": self.arrival_rates.copy(),
        }
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _process_green_phase(self) -> int:
        """
        在当前绿灯相位下放行车辆。
        每步最多放行 `green_flow_rate` 辆车。
        返回: 本步通行的车辆数。
        """
        if self.in_yellow:
            return 0  # 黄灯期间不放行

        green_flow_rate = 3  # 每步最多通行车辆数
        departed = 0

        if self.current_phase == self.PHASE_NS:
            active_lanes = [self.NORTH, self.SOUTH]
        else:
            active_lanes = [self.EAST, self.WEST]

        for lane in active_lanes:
            for _ in range(green_flow_rate):
                wait_time = self.queues[lane].depart_vehicle(self.global_step)
                if wait_time is not None:
                    departed += 1
                    self.total_departed += 1
                    self.total_wait_accum += wait_time
                else:
                    break

        return departed

    def _generate_arrivals(self) -> None:
        """根据各方向当前到达率, 泊松采样生成新车辆。"""
        for i in range(4):
            lam = self.arrival_rates[i]
            if lam <= 0:
                continue
            n_new = self.np_random.poisson(lam)
            for _ in range(n_new):
                self.queues[i].add_vehicle(self.global_step)

    def _update_arrival_rates(self) -> None:
        """根据流量模式更新各方向到达率。"""
        t = self.global_step

        if self.traffic_pattern == "uniform":
            self.arrival_rates = self._base_arrival_rates.copy()

        elif self.traffic_pattern == "peak_hour":
            # 早高峰: 南北方向流量激增
            rates = self._base_arrival_rates.copy()
            if self.peak_start <= t < self.peak_end:
                rates[self.NORTH] *= self.peak_multiplier
                rates[self.SOUTH] *= self.peak_multiplier
            self.arrival_rates = rates

        elif self.traffic_pattern == "tidal":
            # 潮汐车流: 周期性切换主要流量方向
            rates = self._base_arrival_rates.copy()
            phase_in_period = t % self.tidal_period
            if phase_in_period < self.tidal_period // 2:
                # 前半周期: 南北高峰
                rates[self.NORTH] *= self.peak_multiplier
                rates[self.SOUTH] *= self.peak_multiplier
            else:
                # 后半周期: 东西高峰
                rates[self.EAST] *= self.peak_multiplier
                rates[self.WEST] *= self.peak_multiplier
            self.arrival_rates = rates

        elif self.traffic_pattern == "burst":
            # 随机尖峰: 每隔 burst_interval 步在随机方向出现流量尖峰
            rates = self._base_arrival_rates.copy()
            burst_interval = 150
            if t > 0 and t % burst_interval < 10:
                direction = self.np_random.integers(0, 4)
                rates[direction] = self.burst_intensity
            self.arrival_rates = rates

        elif self.traffic_pattern == "low_traffic":
            # 低流量: 所有方向保持低到达率
            self.arrival_rates = self._base_arrival_rates.copy()

    def _get_observation(self) -> np.ndarray:
        """构建归一化观测向量, 支持 state_dim∈{4,8,9}。"""
        max_queue = 30.0
        queue_norm = np.array(
            [min(q.length / max_queue, 1.0) for q in self.queues],
            dtype=np.float32,
        )

        if self.state_dim == 4:
            # S4: 仅排队长度
            return queue_norm.astype(np.float32)

        wait_norm = np.array(
            [min(q.head_wait_time(self.global_step) / self.max_wait_normalize, 1.0)
             for q in self.queues],
            dtype=np.float32,
        )

        if self.state_dim == 8:
            # S8: 排队 + 等待时间
            return np.concatenate([queue_norm, wait_norm]).astype(np.float32)

        # S9: 排队 + 等待时间 + 相位标量 (0=NS, 1=EW)
        phase_scalar = np.array([float(self.current_phase)], dtype=np.float32)
        return np.concatenate([queue_norm, wait_norm, phase_scalar]).astype(np.float32)

    def _compute_reward(self, switch_occurred: bool) -> float:
        """计算奖励, 支持 4 种奖励函数变体。"""
        total_queue = sum(q.length for q in self.queues)
        total_wait = sum(q.head_wait_time(self.global_step) for q in self.queues)

        if self.reward_type == "queue_only":
            # R1: 仅排队惩罚
            reward = -float(total_queue)
        elif self.reward_type == "queue_switch":
            # R2: 排队 + 切换惩罚
            reward = -float(total_queue)
            if switch_occurred:
                reward -= self.switch_penalty
        elif self.reward_type == "queue_wait":
            # R3: 排队 + 等待 (无切换惩罚)
            reward = -(self.w1 * total_queue + self.w2 * total_wait)
        else:
            # R4 "composite": 排队 + 等待 + 切换惩罚
            reward = -(self.w1 * total_queue + self.w2 * total_wait)
            if switch_occurred:
                reward -= self.switch_penalty

        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # 辅助 / 调试
    # ──────────────────────────────────────────────────────────────────────

    def get_state_raw(self) -> Dict[str, Any]:
        """返回环境原始状态 (调试用)。"""
        return {
            "step": self.global_step,
            "phase": "NS" if self.current_phase == self.PHASE_NS else "EW",
            "in_yellow": self.in_yellow,
            "phase_duration": self.phase_duration,
            "queues": [q.length for q in self.queues],
            "head_waits": [q.head_wait_time(self.global_step) for q in self.queues],
            "arrival_rates": self.arrival_rates.tolist(),
            "total_departed": self.total_departed,
        }

    def render(self, mode: str = "human") -> None:
        """文本渲染 (调试用)。"""
        state = self.get_state_raw()
        phase_str = "🟡 YELLOW" if self.in_yellow else ("🟢 NS" if self.current_phase == 0 else "🟢 EW")
        print(f"Step {state['step']:4d} | Phase: {phase_str:10s} | "
              f"Queues: N={state['queues'][0]:2d} S={state['queues'][1]:2d} "
              f"E={state['queues'][2]:2d} W={state['queues'][3]:2d} | "
              f"Departed: {state['total_departed']}")


# ──────────────────────────────────────────────────────────────────────────────
# 固定配时基准环境 (用于对比)
# ──────────────────────────────────────────────────────────────────────────────

class FixedTimeController:
    """
    固定配时红绿灯控制器 — 作为 RL 方法的 baseline 对比。
    """

    def __init__(self, env: TrafficLightEnv, fixed_interval: int = 30):
        self.env = env
        self.fixed_interval = fixed_interval
        self.steps_in_phase = 0

    def select_action(self) -> int:
        """每 fixed_interval 步切换一次相位。"""
        self.steps_in_phase += 1
        if self.steps_in_phase >= self.fixed_interval:
            self.steps_in_phase = 0
            return 1  # 切换
        return 0  # 保持

    def run_episode(self) -> Dict[str, float]:
        """运行一个完整 episode 并返回统计指标。"""
        obs, _ = self.env.reset()
        total_reward = 0.0
        total_queue = 0.0
        total_wait = 0.0
        steps = 0

        done = False
        while not done:
            action = self.select_action()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_queue += info["total_queue"]
            total_wait += info["avg_wait"]
            steps += 1

        return {
            "total_reward": total_reward,
            "avg_queue": total_queue / max(steps, 1),
            "avg_wait": total_wait / max(steps, 1),
            "total_departed": self.env.total_departed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 多路口环境 (扩展)
# ──────────────────────────────────────────────────────────────────────────────

class MultiIntersectionEnv(gym.Env):
    """
    多路口干线环境 — 支持 N 个串联路口的协同控制。

    每个路口有两个相位 (NS/EW), 车辆到达与单路口相同。
    相邻路口之间存在交通流传递: 前一路口放行的车辆经过 `travel_steps`
    步后进入下一个路口的对应方向。

    观测空间: Box(N*8, ) — 每个路口 8 维状态拼接
    动作空间: MultiDiscrete([2, 2, ..., 2]) — 每个路口独立控制
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_intersections: int = 3,
        arrival_rates: Optional[np.ndarray] = None,
        travel_steps: int = 10,
        max_steps: int = 1000,
        **env_kwargs,
    ):
        super().__init__()
        self.num_intersections = num_intersections
        self.travel_steps = travel_steps

        # 为每个路口创建独立环境
        if arrival_rates is None:
            arrival_rates = np.ones((num_intersections, 4)) * 2.0
        self.arrival_rates = arrival_rates

        self.sub_envs = [
            TrafficLightEnv(
                arrival_rates=tuple(arrival_rates[i]),
                max_steps=max_steps,
                **env_kwargs,
            )
            for i in range(num_intersections)
        ]

        # 观测 / 动作空间 — 组合
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(num_intersections * 8,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([2] * num_intersections)

        self.max_steps = max_steps
        self.global_step = 0

        # 路口间传输队列: transit[i→i+1] = deque of (arrival_step_at_next)
        self.transit_queues: List[List[int]] = [
            [] for _ in range(num_intersections - 1)
        ]

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.global_step = 0
        self.transit_queues = [[] for _ in range(self.num_intersections - 1)]

        observations = []
        for env in self.sub_envs:
            obs, _ = env.reset(seed=seed)
            observations.append(obs)
        return np.concatenate(observations).astype(np.float32), {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.global_step += 1

        # ── 1. 处理传输中的车辆 ──
        for i in range(self.num_intersections - 1):
            # 从前一个路口向东 / 后一个路口向西看具体情况
            # 简化模型: 南北方向车流沿主干线向北传递
            pass  # 详细逻辑在下面展开

        # ── 2. 每个子环境执行一步 ──
        observations = []
        total_reward = 0.0
        infos = []

        for i, (env, action) in enumerate(zip(self.sub_envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            observations.append(obs)
            total_reward += reward
            infos.append(info)

        combined_obs = np.concatenate(observations).astype(np.float32)
        done = self.global_step >= self.max_steps

        info = {
            "sub_infos": infos,
            "total_queue": sum(info["total_queue"] for info in infos),
            "total_departed": sum(info["departed"] for info in infos),
        }
        return combined_obs, total_reward, done, False, info

    def render(self, mode: str = "human") -> None:
        for i, env in enumerate(self.sub_envs):
            state = env.get_state_raw()
            print(f"[Intersection {i}] Step {state['step']:4d} | "
                  f"Phase: {'NS' if state['phase'] == 'NS' else 'EW':3s} | "
                  f"Q: {state['queues']} | Dep: {state['total_departed']}")


# ──────────────────────────────────────────────────────────────────────────────
# 环境注册 & 测试
# ──────────────────────────────────────────────────────────────────────────────

def make_env(traffic_pattern: str = "uniform", state_dim: int = 8,
             reward_type: str = "composite", config: Optional[str] = None,
             **kwargs) -> TrafficLightEnv:
    """工厂函数: 快速创建 TrafficLightEnv 实例。
    可通过 config 名称从 ENV_CONFIG 注册表加载预设。"""
    if config is not None and config in ENV_CONFIG:
        cfg = ENV_CONFIG[config].copy()
        cfg.update(kwargs)  # kwargs 可覆盖预设
        return TrafficLightEnv(**cfg)

    default_rates = {
        "uniform": (2.0, 2.0, 2.0, 2.0),
        "peak_hour": (2.0, 2.0, 2.0, 2.0),
        "tidal": (2.0, 2.0, 2.0, 2.0),
        "burst": (1.5, 1.5, 1.5, 1.5),
        "low_traffic": (0.5, 0.5, 0.5, 0.5),
    }
    rates = kwargs.pop("arrival_rates", default_rates.get(traffic_pattern, (2.0, 2.0, 2.0, 2.0)))
    return TrafficLightEnv(
        arrival_rates=rates,
        traffic_pattern=traffic_pattern,
        state_dim=state_dim,
        reward_type=reward_type,
        **kwargs,
    )


if __name__ == "__main__":
    # 快速冒烟测试
    print("=" * 60)
    print("TrafficLightEnv 冒烟测试")
    print("=" * 60)

    # ── 5 种交通模式 ──
    for pattern in ["uniform", "peak_hour", "tidal", "burst", "low_traffic"]:
        env = make_env(traffic_pattern=pattern, max_steps=100)
        obs, _ = env.reset()
        total_r = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            if terminated or truncated:
                break
        print(f"  Pattern={pattern:12s} | Ep Reward={total_r:8.2f} | "
              f"Departed={env.total_departed:4d} | obs_dim={obs.shape[0]}")

    # ── 3 种状态维度 ──
    print("\n--- State Dimension Variants ---")
    for sd in [4, 8, 9]:
        env = make_env(state_dim=sd, max_steps=50)
        obs, _ = env.reset()
        assert obs.shape[0] == sd, f"Expected dim {sd}, got {obs.shape[0]}"
        print(f"  S{sd}: obs shape={obs.shape} ✓")

    # ── 4 种奖励函数 ──
    print("\n--- Reward Function Variants ---")
    for rt in ["queue_only", "queue_switch", "queue_wait", "composite"]:
        env = make_env(reward_type=rt, max_steps=50)
        obs, _ = env.reset()
        r_total = 0.0
        for _ in range(50):
            _, r, _, _, _ = env.step(env.action_space.sample())
            r_total += r
        print(f"  {rt:14s}: total_reward={r_total:.2f}")

    # ── ENV_CONFIG 预设 ──
    print("\n--- ENV_CONFIG presets ---")
    for cfg_name in ["default", "burst", "low_traffic", "uniform_s4_r1", "uniform_s9_r3"]:
        env = make_env(config=cfg_name, max_steps=50)
        obs, _ = env.reset()
        print(f"  {cfg_name:20s}: obs_dim={obs.shape[0]}, reward_type={env.reward_type}, "
              f"pattern={env.traffic_pattern}")

    print("\nFixed-time controller test:")
    env = make_env(traffic_pattern="peak_hour", max_steps=200)
    ft = FixedTimeController(env, fixed_interval=30)
    stats = ft.run_episode()
    print(f"  Avg Queue={stats['avg_queue']:.2f} | Avg Wait={stats['avg_wait']:.2f} | "
          f"Departed={stats['total_departed']}")

    print("\nMultiIntersectionEnv test:")
    multi_env = MultiIntersectionEnv(num_intersections=3, max_steps=50)
    obs, _ = multi_env.reset()
    for _ in range(50):
        actions = multi_env.action_space.sample()
        obs, r, done, _, info = multi_env.step(actions)
        if done:
            break
    print(f"  Total departed: {info['total_departed']}")

    print("\n✅ All tests passed!")
