"""
交通信号灯 RL — 实时仿真运行器
==============================
负责:
  - 初始化 TrafficLightEnv + RL Agent
  - 逐步执行仿真, 生成每步状态快照
  - 输出车辆坐标用于 Canvas 渲染
  - 支持 play/pause/step/speed 控制
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable

from traffic_env import make_env
from agent import create_agent
from pg_agent import create_pg_agent


# ══════════════════════════════════════════════════════════════════════════════
# 车辆数据模型
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleSnapshot:
    """单个车辆的快照, 供前端 Canvas 渲染。"""
    vid: int
    direction: str          # 'N','S','E','W'
    x: float                # Canvas x 坐标 (0-720)
    y: float                # Canvas y 坐标 (0-720)
    speed: float            # 当前速度
    waiting: bool           # 是否在排队
    distance_to_stop: float # 到停车线距离


@dataclass
class StepSnapshot:
    """单步仿真的完整状态快照。"""
    episode: int
    step: int
    phase: int              # 0=NS_Green, 1=EW_Green
    phase_label: str        # 'NS' | 'EW'
    action: int             # 0=Hold, 1=Switch
    vehicles: List[VehicleSnapshot]
    queues: List[int]       # [N, S, E, W]
    avg_wait: List[float]   # [N, S, E, W]
    q_values: List[float]   # [Q(s,0), Q(s,1)]
    policy: List[float]     # [π(0), π(1)] — PG 系列
    reward: float
    cumulative_reward: float
    throughput: int
    switch_penalty: float
    done: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Canvas 坐标映射
# ══════════════════════════════════════════════════════════════════════════════

# 画布 720×720, 路口中心 (360,360)
CANVAS_SIZE = 720
CENTER = 360
ROAD_WIDTH = 80       # 每条道路宽度
LANE_WIDTH = 32       # 单车道宽
STOP_LINE_OFFSET = 50 # 停车线距中心距离
VEHICLE_LENGTH = 18
VEHICLE_WIDTH = 10

# 方向 → 坐标系参数
DIRECTION_PARAMS = {
    'N': {'start_y': 730, 'end_y': -10, 'x_offset': -12, 'stop_y': CENTER + STOP_LINE_OFFSET, 'dx': 0, 'dy': -1},
    'S': {'start_y': -10, 'end_y': 730, 'x_offset': 12,  'stop_y': CENTER - STOP_LINE_OFFSET, 'dx': 0, 'dy': 1},
    'E': {'start_x': -10, 'end_x': 730, 'y_offset': 12,  'stop_x': CENTER - STOP_LINE_OFFSET, 'dx': 1, 'dy': 0},
    'W': {'start_x': 730, 'end_x': -10, 'y_offset': -12, 'stop_x': CENTER + STOP_LINE_OFFSET, 'dx': -1, 'dy': 0},
}


def _compute_vehicle_positions(env, canvas_size: int = CANVAS_SIZE) -> List[VehicleSnapshot]:
    """
    根据环境的排队长度和已通过车辆, 计算每辆车在 Canvas 上的坐标。
    车辆按照方向分车道排列, 排队车辆停在停车线后。
    """
    vehicles = []
    vid = 0
    center = canvas_size // 2
    stop_offset = STOP_LINE_OFFSET
    veh_gap = VEHICLE_LENGTH + 6  # 车长+间距

    # 连续计数: 每种方向通过模拟一个虚拟车辆队列
    # 方法: 取每个方向排队长度 + 一些已出发车的尾迹
    for d in ['N', 'S', 'E', 'W']:
        params = DIRECTION_PARAMS[d]
        queue_idx = {'N': 0, 'S': 1, 'E': 2, 'W': 3}[d]

        queue_len = max(0, int(env.queues[queue_idx].length))
        arrivals = env.arrival_rates[queue_idx]

        # 排队车辆: 从停车线向后排列
        for i in range(min(queue_len, 20)):  # 最多渲染 20 辆
            if d in ('N', 'S'):
                y_stop = params['stop_y']
                y = y_stop + (i + 0.5) * veh_gap * (-params['dy'])
                x = center + params['x_offset']
            else:
                x_stop = params['stop_x']
                x = x_stop + (i + 0.5) * veh_gap * (-params['dx'])
                y = center + params['y_offset']

            waiting = True
            speed = 0.0

            vehicles.append(VehicleSnapshot(
                vid=vid, direction=d,
                x=float(x), y=float(y),
                speed=speed, waiting=waiting,
                distance_to_stop=float(i * veh_gap),
            ))
            vid += 1

        # 已通过/正在通过路口的车辆: 从停车线向前
        # 简单估算: 根据 throughput 和历史, 渲染 2-5 辆行驶中的车
        moving_count = min(3, int(arrivals * 3)) if queue_len < 10 else 0
        for j in range(moving_count):
            progress = (j + 1) * 40  # 行进距离
            if d in ('N', 'S'):
                y = params['stop_y'] - progress * params['dy']
                x = center + params['x_offset']
            else:
                x = params['stop_x'] - progress * params['dx']
                y = center + params['y_offset']

            vehicles.append(VehicleSnapshot(
                vid=vid, direction=d,
                x=float(x), y=float(y),
                speed=2.0, waiting=False,
                distance_to_stop=float(-progress),
            ))
            vid += 1

    return vehicles


# ══════════════════════════════════════════════════════════════════════════════
# 仿真运行器
# ══════════════════════════════════════════════════════════════════════════════

class SimRunner:
    """
    仿真运行器 — 管理一个仿真会话。

    用法:
        runner = SimRunner()
        snapshot = runner.init(agent_type='dueling', pattern='uniform')
        while not snapshot.done:
            snapshot = runner.step()
        # 控制播放速度用 play_async()
    """

    def __init__(self):
        self.env = None
        self.agent = None
        self.agent_type = None
        self.is_pg = False
        self.episode = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.obs = None
        self.last_action = 0
        self.last_q = [0.0, 0.0]
        self.last_policy = [0.5, 0.5]
        self.last_reward = 0.0

        # 异步播放
        self._play_thread: Optional[threading.Thread] = None
        self._play_speed: float = 1.0
        self._play_active = threading.Event()  # True=播放中, False=暂停
        self._play_active.clear()  # 初始暂停
        self._play_running = False
        self._on_step_callback: Optional[Callable] = None

    # ── 初始化 ──
    def init(self, agent_type: str = 'dueling', pattern: str = 'uniform',
             state_dim: int = 8, reward_type: str = 'composite',
             max_steps: int = 2000) -> StepSnapshot:
        """初始化新仿真会话。"""
        self.agent_type = agent_type
        self.is_pg = agent_type in ('a2c', 'ppo')

        self.env = make_env(
            traffic_pattern=pattern,
            state_dim=state_dim,
            reward_type=reward_type,
            max_steps=max_steps,
        )
        if self.is_pg:
            self.agent = create_pg_agent(agent_type=agent_type, state_dim=state_dim)
        else:
            self.agent = create_agent(agent_type=agent_type, state_dim=state_dim)

        self.episode = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False

        self.obs, _ = self.env.reset()
        self.last_action = 0
        self.last_q = [0.0, 0.0]
        self.last_policy = [0.5, 0.5]

        return self._build_snapshot()

    def reset(self) -> StepSnapshot:
        """重置当前 episode。"""
        if self.env is None:
            return self.init()
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.episode += 1
        return self._build_snapshot()

    # ── 单步执行 ──
    def step(self) -> StepSnapshot:
        """执行一步仿真并返回快照。"""
        if self.env is None or self.done:
            return self.reset()

        # 获取 Q 值 / 策略分布
        self._compute_q_or_policy()

        # 推理模式: 纯 argmax (模型训练时的最优决策)
        action = self.agent.select_action(self.obs, evaluate=True)
        self.last_action = action

        # 执行动作
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # 存储 transition
        self.agent.store_transition(self.obs, action, reward, next_obs, done)
        if not self.is_pg:
            self.agent.update()

        self.obs = next_obs
        self.cumulative_reward += reward
        self.last_reward = float(reward)
        self.step_count += 1
        self.done = done

        if done and self.is_pg:
            self.agent.update()

        return self._build_snapshot()

    def _compute_q_or_policy(self) -> None:
        """获取当前状态的 Q 值或策略分布。"""
        import torch
        st = torch.FloatTensor(self.obs).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            if self.is_pg:
                logits = self.agent.actor(st)
                probs = torch.softmax(logits, dim=-1)
                self.last_policy = probs.squeeze().cpu().tolist()
                self.last_q = self.last_policy  # PG 用 policy 代替 Q
            else:
                q = self.agent.online_net(st)
                self.last_q = q.squeeze().cpu().tolist()
                # 温差软最大化: tau = max(1, Q值范围) 避免 Q 值差距过大时退化为 0/100%
                q_tensor = q.squeeze()
                q_range = (q_tensor.max() - q_tensor.min()).item()
                tau = max(1.0, q_range)
                probs = torch.softmax(q_tensor / tau, dim=-1)
                self.last_policy = probs.cpu().tolist()

    def _build_snapshot(self) -> StepSnapshot:
        """构建当前状态的 Snapshot。"""
        if self.env is None:
            return StepSnapshot(episode=0, step=0, phase=0, phase_label='NS',
                                action=0, vehicles=[], queues=[0,0,0,0],
                                avg_wait=[0,0,0,0], q_values=[0,0], policy=[0,0],
                                reward=0, cumulative_reward=0, throughput=0,
                                switch_penalty=0)

        vehicles = _compute_vehicle_positions(self.env)
        phase = self.env.current_phase

        # 平均等待
        avg_wait = [
            self.env.queues[i].head_wait_time(self.env.global_step)
            for i in range(4)
        ]

        return StepSnapshot(
            episode=self.episode,
            step=self.step_count,
            phase=phase,
            phase_label='NS' if phase == 0 else 'EW',
            action=self.last_action,
            vehicles=vehicles,
            queues=[int(self.env.queues[i].length) for i in range(4)],
            avg_wait=avg_wait,
            q_values=self.last_q,
            policy=self.last_policy,
            reward=self.last_reward,
            cumulative_reward=float(self.cumulative_reward),
            throughput=int(self.env.total_departed),
            switch_penalty=float(self.env.switch_penalty) if self.last_action == 1 else 0.0,
            done=self.done,
        )

    def to_json(self, snapshot: StepSnapshot) -> dict:
        """将快照序列化为前端 JSON 格式。"""
        return {
            "type": "state",
            "episode": snapshot.episode,
            "step": snapshot.step,
            "phase": snapshot.phase,
            "phase_label": snapshot.phase_label,
            "action": snapshot.action,
            "vehicles": [
                {
                    "vid": v.vid, "dir": v.direction,
                    "x": round(v.x, 1), "y": round(v.y, 1),
                    "speed": round(v.speed, 1), "waiting": v.waiting,
                }
                for v in snapshot.vehicles[:50]  # 最多 50 辆车
            ],
            "queues": snapshot.queues,
            "avg_wait": [round(w, 1) for w in snapshot.avg_wait],
            "q_values": [round(v, 4) for v in snapshot.q_values],
            "policy": [round(v, 4) for v in snapshot.policy],
            "reward": round(snapshot.reward, 1),
            "cumulative_reward": round(snapshot.cumulative_reward, 1),
            "throughput": snapshot.throughput,
            "switch_penalty": snapshot.switch_penalty,
            "done": snapshot.done,
            "arrival_rates": [round(r, 2) for r in (self.env.arrival_rates.tolist() if self.env else [0,0,0,0])],
            "mean_wait": round(float(np.mean(snapshot.avg_wait)) if snapshot.avg_wait else 0.0, 1),
            "total_wait_accum": round(float(self.env.total_wait_accum) if self.env else 0.0, 1),
        }

    # ── 异步播放控制 ──
    def play(self, speed: float = 1.0, on_step: Optional[Callable] = None) -> None:
        """开始连续播放。"""
        self._play_speed = speed
        self._on_step_callback = on_step
        self._play_active.set()  # 激活播放
        if not self._play_running:
            self._play_running = True
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()

    def pause(self) -> None:
        """暂停播放。"""
        self._play_active.clear()

    def stop(self) -> None:
        """停止播放。"""
        self._play_running = False
        self._play_active.set()  # 让 wait() 通过，退出循环

    def set_speed(self, speed: float) -> None:
        self._play_speed = speed

    @property
    def is_playing(self) -> bool:
        return self._play_running and self._play_active.is_set()

    def _play_loop(self) -> None:
        """播放线程主循环。"""
        base_delay = 0.05  # 1× 速度 = 50ms/步
        while self._play_running:
            self._play_active.wait()  # 等待激活信号 (True=播放)
            if not self._play_running:
                break
            if self.done:
                self._play_active.clear()
                break
            try:
                snap = self.step()
                if self._on_step_callback:
                    self._on_step_callback(self.to_json(snap))
            except Exception as e:
                print(f"  Play step error: {e}")
                self._play_active.clear()
                break
            time.sleep(base_delay / max(self._play_speed, 0.1))

    # ── 模型加载 ──
    def load_model(self, path: str) -> bool:
        """加载预训练模型权重。"""
        if self.agent is None:
            return False
        try:
            self.agent.load(path)
            return True
        except Exception as e:
            print(f"  Model load failed: {e}")
            return False
