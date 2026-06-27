"""
多路口交通信号灯环境 — 2×2 网格
=================================
四个路口呈 2×2 网格排列，车辆在相邻路口间流转。

拓扑:
      I0 (NW)  ────  I1 (NE)
        |               |
      I2 (SW)  ────  I3 (SE)

外部到达: 8 个边缘方向
  I0: N0(0), W0(1)
  I1: N1(2), E1(3)
  I2: S2(4), W2(5)
  I3: S3(6), E3(7)

内部流转 (8 条走廊):
  I0→E → I1→W    I1→W → I0→E     (东西走廊 ×2)
  I0→S → I2→N    I2→N → I0→S     (南北走廊 ×2)
  I1→S → I3→N    I3→N → I1→S
  I2→E → I3→W    I3→W → I2→E

每个路口独立观测 8 维 (4排队 + 4等待), 独立决策 2 动作 (HOLD/SWITCH)。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from traffic_env import VehicleQueue

# 方向/相位常量
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
PHASE_NS, PHASE_EW = 0, 1

# 8 条内部走廊定义: (src_i, src_lane, dst_i, dst_lane)
INTERNAL_LINKS = [
    (0, EAST,  1, WEST),   # I0→E → I1→W
    (1, WEST,  0, EAST),   # I1→W → I0→E
    (0, SOUTH, 2, NORTH),  # I0→S → I2→N
    (2, NORTH, 0, SOUTH),  # I2→N → I0→S
    (1, SOUTH, 3, NORTH),  # I1→S → I3→N
    (3, NORTH, 1, SOUTH),  # I3→N → I1→S
    (2, EAST,  3, WEST),   # I2→E → I3→W
    (3, WEST,  2, EAST),   # I3→W → I2→E
]


class Grid2x2Env(gym.Env):
    """
    2×2 网格多路口交通环境。

    观测空间: Box(32,) — 4路口拼接 ([I0 8维, I1 8维, I2 8维, I3 8维])
    动作空间: MultiDiscrete([2, 2, 2, 2]) — 4路口独立 HOLD/SWITCH
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        arrival_rates: Tuple[float, ...] = (2.0,) * 8,
        # 8 个外部方向: N0, W0, N1, E1, S2, W2, S3, E3
        w1: float = 1.0,
        w2: float = 0.02,
        switch_penalty: float = 2.0,
        max_steps: int = 1000,
        min_green_duration: int = 5,
        yellow_duration: int = 1,
        green_flow_rate: int = 3,
        travel_delay: int = 3,
        max_wait_normalize: float = 200.0,
        traffic_pattern: str = "uniform",
        state_dim: int = 8,
        reward_type: str = "composite",
        peak_multiplier: float = 3.0,
        peak_start: int = 0,
        peak_end: int = 500,
        tidal_period: int = 400,
        burst_intensity: float = 8.0,
    ):
        super().__init__()
        self.n_intersections = 4
        self.state_dim_per = state_dim
        self.travel_delay = travel_delay
        self.green_flow_rate = green_flow_rate

        # 外部到达率: [N0, W0, N1, E1, S2, W2, S3, E3]
        self._base_arrival_rates = np.array(arrival_rates, dtype=np.float32)
        self.arrival_rates = self._base_arrival_rates.copy()

        # 奖励参数
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
        self.reward_type = reward_type

        # 观测/动作空间
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_intersections * state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([2] * self.n_intersections)

        # 内部状态
        self.all_queues: List[List[VehicleQueue]] = []
        self.all_phases: List[int] = []
        self.all_phase_durations: List[int] = []
        self.all_in_yellow: List[bool] = []
        self.all_yellow_remaining: List[int] = []
        self.all_total_departed: List[int] = []
        self.all_total_wait_accum: List[float] = []
        self.global_step: int = 0

        # 传输管道: (remaining_steps, dst_i, dst_lane)
        self.pipelines: List[Tuple[int, int, int]] = []

        # 统计
        self.episode_queue_history: List[float] = []
        self.episode_wait_history: List[float] = []

    # ── Gym API ──
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.all_queues = [[VehicleQueue() for _ in range(4)] for _ in range(self.n_intersections)]
        self.all_phases = [PHASE_NS] * self.n_intersections
        self.all_phase_durations = [0] * self.n_intersections
        self.all_in_yellow = [False] * self.n_intersections
        self.all_yellow_remaining = [0] * self.n_intersections
        self.all_total_departed = [0] * self.n_intersections
        self.all_total_wait_accum = [0.0] * self.n_intersections
        self.global_step = 0
        self.pipelines = []
        self.episode_queue_history = []
        self.episode_wait_history = []

        if options is not None:
            for k in ('w1', 'w2', 'switch_penalty', 'traffic_pattern',
                       'burst_intensity', 'travel_delay'):
                if k in options:
                    setattr(self, k, options[k])

        self._update_arrival_rates()
        obs = self._get_observation()
        return obs, {}

    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        n = self.n_intersections

        # 1. 绿灯通行
        departed_this_step = [self._process_green(i) for i in range(n)]

        # 2. 黄灯推进
        for i in range(n):
            if self.all_in_yellow[i]:
                self.all_yellow_remaining[i] -= 1
                if self.all_yellow_remaining[i] <= 0:
                    self.all_in_yellow[i] = False
                    self.all_phases[i] = 1 - self.all_phases[i]
                    self.all_phase_durations[i] = 0

        # 3. 路由车辆到相邻路口 (8 条内部走廊)
        for src_i, src_lane, dst_i, dst_lane in INTERNAL_LINKS:
            for _ in range(departed_this_step[src_i][src_lane]):
                self.pipelines.append((self.travel_delay, dst_i, dst_lane))

        # 4. 处理传输管道
        new_pipelines = []
        for rem, dest_i, dest_lane in self.pipelines:
            if rem <= 0:
                self.all_queues[dest_i][dest_lane].add_vehicle(self.global_step)
            else:
                new_pipelines.append((rem - 1, dest_i, dest_lane))
        self.pipelines = new_pipelines

        # 5. 动作执行 (触发黄灯)
        switches_occurred = []
        for i in range(n):
            sw = False
            if actions[i] == 1:
                if not self.all_in_yellow[i] and self.all_phase_durations[i] >= self.min_green_duration:
                    self.all_in_yellow[i] = True
                    self.all_yellow_remaining[i] = self.yellow_duration
                    sw = True
            switches_occurred.append(sw)

        # 6. 外部车辆到达
        self._generate_external_arrivals()

        # 7. 推进时间
        self.global_step += 1
        for i in range(n):
            if not self.all_in_yellow[i]:
                self.all_phase_durations[i] += 1
        self._update_arrival_rates()

        # 8. 观测 & 奖励
        obs = self._get_observation()
        reward = self._compute_reward(switches_occurred)
        terminated = self.global_step >= self.max_steps
        truncated = False

        total_q = sum(sum(q.length for q in iq) for iq in self.all_queues)
        avg_w = np.mean([
            q.head_wait_time(self.global_step)
            for iq in self.all_queues for q in iq
        ])
        self.episode_queue_history.append(total_q)
        self.episode_wait_history.append(avg_w)

        info = {
            "total_queue": total_q,
            "avg_wait": avg_w,
            "total_departed": sum(self.all_total_departed),
            "departed_per": self.all_total_departed,
            "phases": self.all_phases,
            "pipeline_count": len(self.pipelines),
        }
        return obs, reward, terminated, truncated, info

    # ── 内部 ──
    def _process_green(self, i: int) -> List[int]:
        departed = [0, 0, 0, 0]
        if self.all_in_yellow[i]:
            return departed

        phase = self.all_phases[i]
        active = [NORTH, SOUTH] if phase == PHASE_NS else [EAST, WEST]

        for lane in active:
            for _ in range(self.green_flow_rate):
                wt = self.all_queues[i][lane].depart_vehicle(self.global_step)
                if wt is not None:
                    departed[lane] += 1
                    self.all_total_departed[i] += 1
                    self.all_total_wait_accum[i] += wt
                else:
                    break
        return departed

    def _generate_external_arrivals(self) -> None:
        """生成外部车辆 (含边缘贯通车流)。"""
        # 8 个外部方向: N0(0), W0(1), N1(2), E1(3), S2(4), W2(5), S3(6), E3(7)
        external_map = [
            (0, NORTH, 0),   # I0 北
            (0, WEST,  1),   # I0 西
            (1, NORTH, 2),   # I1 北
            (1, EAST,  3),   # I1 东
            (2, SOUTH, 4),   # I2 南
            (2, WEST,  5),   # I2 西
            (3, SOUTH, 6),   # I3 南
            (3, EAST,  7),   # I3 东
        ]
        for inter_i, lane, rate_idx in external_map:
            lam = self.arrival_rates[rate_idx]
            if lam > 0:
                n = self.np_random.poisson(lam)
                for _ in range(n):
                    self.all_queues[inter_i][lane].add_vehicle(self.global_step)

        # 贯通车流: 边缘方向的 40% 直行通过
        # I0 WEST → 直行到 EAST (从 I0 穿越到 I1)
        lam = self.arrival_rates[1] * 0.4
        if lam > 0:
            n = self.np_random.poisson(lam)
            for _ in range(n):
                self.all_queues[0][EAST].add_vehicle(self.global_step)

        # I1 EAST → 直行到 WEST (从 I1 穿越到 I0)
        lam = self.arrival_rates[3] * 0.4
        if lam > 0:
            n = self.np_random.poisson(lam)
            for _ in range(n):
                self.all_queues[1][WEST].add_vehicle(self.global_step)

        # I2 WEST → 直行到 EAST (从 I2 穿越到 I3)
        lam = self.arrival_rates[5] * 0.4
        if lam > 0:
            n = self.np_random.poisson(lam)
            for _ in range(n):
                self.all_queues[2][EAST].add_vehicle(self.global_step)

        # I3 EAST → 直行到 WEST (从 I3 穿越到 I2)
        lam = self.arrival_rates[7] * 0.4
        if lam > 0:
            n = self.np_random.poisson(lam)
            for _ in range(n):
                self.all_queues[3][WEST].add_vehicle(self.global_step)

    def _update_arrival_rates(self) -> None:
        t = self.global_step
        base = self._base_arrival_rates.copy()

        if self.traffic_pattern == "uniform":
            pass
        elif self.traffic_pattern == "peak_hour":
            if self.peak_start <= t < self.peak_end:
                for idx in range(8):
                    base[idx] *= self.peak_multiplier
        elif self.traffic_pattern == "tidal":
            phase_in = t % self.tidal_period
            if phase_in < self.tidal_period // 2:
                # 南北高峰
                for idx in [0, 2, 4, 6]:  # N0,N1,S2,S3
                    base[idx] *= self.peak_multiplier
            else:
                # 东西高峰
                for idx in [1, 3, 5, 7]:  # W0,E1,W2,E3
                    base[idx] *= self.peak_multiplier
        elif self.traffic_pattern == "burst":
            if self.np_random.random() < 0.05:
                idx = self.np_random.integers(0, 8)
                base[idx] += self.burst_intensity * self.np_random.random()
        elif self.traffic_pattern == "low_traffic":
            base = base * 0.25

        self.arrival_rates = np.maximum(0, base)

    def _get_observation(self) -> np.ndarray:
        obs_parts = []
        for i in range(self.n_intersections):
            qs = [self.all_queues[i][j].length for j in range(4)]
            waits = [
                min(self.all_queues[i][j].head_wait_time(self.global_step),
                    self.max_wait_normalize) / self.max_wait_normalize
                for j in range(4)
            ]
            obs_parts.extend(qs + waits)
        return np.array(obs_parts, dtype=np.float32)

    def get_local_obs(self, i: int) -> np.ndarray:
        """获取路口 i 的本地 8 维观测。"""
        qs = [self.all_queues[i][j].length for j in range(4)]
        waits = [
            min(self.all_queues[i][j].head_wait_time(self.global_step),
                self.max_wait_normalize) / self.max_wait_normalize
            for j in range(4)
        ]
        return np.array(qs + waits, dtype=np.float32)

    def _compute_reward(self, switches: List[bool]) -> float:
        total = 0.0
        for i in range(self.n_intersections):
            queues = sum(self.all_queues[i][j].length for j in range(4))
            waits = sum(
                self.all_queues[i][j].cumulative_wait_time(self.global_step)
                for j in range(4)
            )
            r = -(self.w1 * queues + self.w2 * waits)
            if switches[i]:
                r -= self.switch_penalty
            total += r
        return total


# ── 工厂函数 (兼容 make_env 风格) ──
def make_grid2x2_env(
    traffic_pattern: str = "uniform",
    state_dim: int = 8,
    max_steps: int = 1000,
    **kwargs,
) -> Grid2x2Env:
    """创建 2×2 网格环境。"""
    return Grid2x2Env(
        arrival_rates=(2.0,) * 8,
        traffic_pattern=traffic_pattern,
        state_dim=state_dim,
        max_steps=max_steps,
        **kwargs,
    )
