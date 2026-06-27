"""
多路口交通信号灯环境 — 1×2 走廊
=================================
两路口相邻排列 (West-East), 车辆在路口间流转。

架构:
  Intersection 0 (West)          Intersection 1 (East)

      N0                             N1
      |                              |
  W0--+--E0  ====走廊(travel_delay)====  W1--+--E1
      |                              |
      S0                             S1

外部到达: I0[N,S,W] + I1[N,S,E]  (6个边缘方向)
内部流转: I0→E 出发 → delay后 → I1→W 到达
          I1→W 出发 → delay后 → I0→E 到达

每个路口独立观测 8 维 (4排队 + 4等待), 独立决策 2 动作 (HOLD/SWITCH)。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from traffic_env import VehicleQueue

# 方向/相位常量 (与单路口一致)
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
PHASE_NS, PHASE_EW = 0, 1


class MultiIntersectionEnv(gym.Env):
    """
    1×2 走廊多路口交通环境。

    观测空间: Box(16,) — 两路口拼接 ([I0 8维, I1 8维])
    动作空间: MultiDiscrete([2, 2]) — 两路口独立 HOLD/SWITCH
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        arrival_rates: Tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        # 6个方向: N0, S0, W0,  N1, S1, E1
        w1: float = 1.0,
        w2: float = 0.02,
        switch_penalty: float = 2.0,
        max_steps: int = 1000,
        min_green_duration: int = 5,
        yellow_duration: int = 1,
        green_flow_rate: int = 3,
        travel_delay: int = 3,          # 路口间行驶步数
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
        self.n_intersections = 2
        self.state_dim_per = state_dim
        self.travel_delay = travel_delay
        self.green_flow_rate = green_flow_rate

        # 外部到达率: [N0, S0, W0, N1, S1, E1]
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

        # ── 观测/动作空间 ──
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_intersections * state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([2, 2])

        # ── 内部状态 ──
        # 每个路口 4 条队列
        self.all_queues: List[List[VehicleQueue]] = []
        self.all_phases: List[int] = []
        self.all_phase_durations: List[int] = []
        self.all_in_yellow: List[bool] = []
        self.all_yellow_remaining: List[int] = []
        self.all_total_departed: List[int] = []
        self.all_total_wait_accum: List[float] = []

        self.global_step: int = 0

        # 传输管道: (remaining_steps, dest_intersection, dest_lane)
        self.pipelines: List[Tuple[int, int, int]] = []

        # 统计
        self.episode_queue_history: List[float] = []
        self.episode_wait_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────
    # Gym API
    # ──────────────────────────────────────────────────────────────────

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

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        actions: list[int] 长度 2, 每路口独立动作。
        """
        # ── 1. 绿灯通行 (所有路口) ──
        departed_this_step = [[0, 0, 0, 0], [0, 0, 0, 0]]  # [路口][方向]
        for i in range(self.n_intersections):
            departed_this_step[i] = self._process_green(i)

        # ── 2. 黄灯推进 ──
        for i in range(self.n_intersections):
            if self.all_in_yellow[i]:
                self.all_yellow_remaining[i] -= 1
                if self.all_yellow_remaining[i] <= 0:
                    self.all_in_yellow[i] = False
                    self.all_phases[i] = 1 - self.all_phases[i]
                    self.all_phase_durations[i] = 0

        # ── 3. 路由车辆到相邻路口 ──
        # I0 东向出发 → I1 西向到达
        for _ in range(departed_this_step[0][EAST]):
            self.pipelines.append((self.travel_delay, 1, WEST))
        # I1 西向出发 → I0 东向到达
        for _ in range(departed_this_step[1][WEST]):
            self.pipelines.append((self.travel_delay, 0, EAST))

        # ── 4. 处理传输管道 ──
        new_pipelines = []
        for rem, dest_i, dest_lane in self.pipelines:
            if rem <= 0:
                self.all_queues[dest_i][dest_lane].add_vehicle(self.global_step)
            else:
                new_pipelines.append((rem - 1, dest_i, dest_lane))
        self.pipelines = new_pipelines

        # ── 5. 动作执行 (触发黄灯) ──
        switches_occurred = []
        for i in range(self.n_intersections):
            sw = False
            if actions[i] == 1:
                if not self.all_in_yellow[i] and self.all_phase_durations[i] >= self.min_green_duration:
                    self.all_in_yellow[i] = True
                    self.all_yellow_remaining[i] = self.yellow_duration
                    sw = True
            switches_occurred.append(sw)

        # ── 6. 外部车辆到达 ──
        self._generate_external_arrivals()

        # ── 7. 推进时间 ──
        self.global_step += 1
        for i in range(self.n_intersections):
            if not self.all_in_yellow[i]:
                self.all_phase_durations[i] += 1
        self._update_arrival_rates()

        # ── 8. 观测 & 奖励 ──
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

    # ──────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────

    def _process_green(self, i: int) -> List[int]:
        """路口 i 的绿灯通行, 返回 [N, S, E, W] 出发数。"""
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
        """生成外部车辆 (含走廊贯通车流)。"""
        # I0 外部: N0(0), S0(1), W0(2) — E0 是走廊, 由 I1→W 管道注入
        for lane, rate_idx in [(NORTH, 0), (SOUTH, 1), (WEST, 2)]:
            lam = self.arrival_rates[rate_idx]
            if lam > 0:
                n = self.np_random.poisson(lam)
                for _ in range(n):
                    self.all_queues[0][lane].add_vehicle(self.global_step)

        # I0 EAST 走廊贯通车流: W0 方向的 40% 改为直行通过 (W→E 直行)
        lam_through = self.arrival_rates[2] * 0.4
        if lam_through > 0:
            n_through = self.np_random.poisson(lam_through)
            for _ in range(n_through):
                self.all_queues[0][EAST].add_vehicle(self.global_step)

        # I1 外部: N1(3), S1(4), E1(5) — W1 是走廊, 由 I0→E 管道注入
        for lane, rate_idx in [(NORTH, 3), (SOUTH, 4), (EAST, 5)]:
            lam = self.arrival_rates[rate_idx]
            if lam > 0:
                n = self.np_random.poisson(lam)
                for _ in range(n):
                    self.all_queues[1][lane].add_vehicle(self.global_step)

        # I1 WEST 走廊贯通车流: E1 方向的 40% 改为直行通过 (E→W 直行)
        lam_through = self.arrival_rates[5] * 0.4
        if lam_through > 0:
            n_through = self.np_random.poisson(lam_through)
            for _ in range(n_through):
                self.all_queues[1][WEST].add_vehicle(self.global_step)

    def _update_arrival_rates(self) -> None:
        t = self.global_step
        base = self._base_arrival_rates.copy()

        if self.traffic_pattern == "uniform":
            pass
        elif self.traffic_pattern == "peak_hour":
            if self.peak_start <= t < self.peak_end:
                base[0] *= self.peak_multiplier  # N0
                base[1] *= self.peak_multiplier  # S0
                base[3] *= self.peak_multiplier  # N1
                base[4] *= self.peak_multiplier  # S1
        elif self.traffic_pattern == "tidal":
            phase_in = t % self.tidal_period
            if phase_in < self.tidal_period // 2:
                base[0] *= self.peak_multiplier  # N0
                base[1] *= self.peak_multiplier  # S0
                base[3] *= self.peak_multiplier  # N1
                base[4] *= self.peak_multiplier  # S1
            else:
                base[2] *= self.peak_multiplier  # W0
                base[5] *= self.peak_multiplier  # E1
        elif self.traffic_pattern == "burst":
            if self.np_random.random() < 0.05:
                idx = self.np_random.integers(0, 6)
                base[idx] += self.burst_intensity * self.np_random.random()
        elif self.traffic_pattern == "low_traffic":
            base = base * 0.25

        self.arrival_rates = np.maximum(0, base)

    def _get_observation(self) -> np.ndarray:
        """拼接两路口观测 → (16,) 或 (2,8)。"""
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

    # ── 方便访问的属性 ──
    @property
    def queues_i0(self):
        return self.all_queues[0]

    @property
    def queues_i1(self):
        return self.all_queues[1]

    @property
    def phase_i0(self):
        return self.all_phases[0]

    @property
    def phase_i1(self):
        return self.all_phases[1]

    @property
    def total_departed_i0(self):
        return self.all_total_departed[0]

    @property
    def total_departed_i1(self):
        return self.all_total_departed[1]
