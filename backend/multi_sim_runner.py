"""
交通信号灯 RL — 多路口实时仿真运行器
=====================================
负责:
  - 初始化 MultiIntersectionEnv + 多个 RL Agent
  - 逐步执行仿真, 生成每步状态快照 (含两路口)
  - 输出车辆坐标用于 Canvas 渲染
  - 支持 play/pause/step/speed 控制
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, threading, numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable

from multi_intersection_env import MultiIntersectionEnv
from agent import create_agent
from pg_agent import create_pg_agent

# ══════════════════════════════════════════════════════════════════════════════
# 车辆数据模型
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleSnapshot:
    vid: int
    direction: str
    x: float
    y: float
    speed: float
    waiting: bool
    distance_to_stop: float
    intersection: int = 0  # 所属路口


@dataclass
class MultiStepSnapshot:
    episode: int
    step: int
    phases: List[int]
    phase_labels: List[str]
    actions: List[int]
    vehicles: List[VehicleSnapshot]
    queues: List[List[int]]       # [[N0,S0,E0,W0], [N1,S1,E1,W1]]
    avg_wait: List[List[float]]   # 同上
    q_values: List[List[float]]   # [[q0_hold,q0_switch], [q1_hold,q1_switch]]
    policy: List[List[float]]
    reward: float
    cumulative_reward: float
    throughputs: List[int]
    switch_penalties: List[float]
    done: bool = False
    pipeline_count: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Canvas 坐标映射 (每个路口 400×400, 路口中心在 (200,200))
# ══════════════════════════════════════════════════════════════════════════════

CANVAS_W = 800   # 总画布宽
CANVAS_H = 420   # 总画布高
INTERSECTION_SIZE = 360
ROAD_WIDTH = 60
LANE_WIDTH = 26
STOP_LINE_OFFSET = 40
VEHICLE_LENGTH = 16
VEHICLE_WIDTH = 9

# 路口 0 中心 (左)
I0_CX, I0_CY = 200, 210
# 路口 1 中心 (右)
I1_CX, I1_CY = 600, 210


DIR_PARAMS = {
    'N': {'dy': -1, 'dx': 0,  'x_offset': -10},
    'S': {'dy': 1,  'dx': 0,  'x_offset': 10},
    'E': {'dx': 1,  'dy': 0,  'y_offset': 10},
    'W': {'dx': -1, 'dy': 0,  'y_offset': -10},
}


def _compute_multi_vehicle_positions(env: MultiIntersectionEnv) -> List[VehicleSnapshot]:
    """计算两路口所有车辆在 Canvas 上的坐标。"""
    vehicles = []
    vid = 0
    veh_gap = VEHICLE_LENGTH + 5

    for i_idx, (cx, cy) in enumerate([(I0_CX, I0_CY), (I1_CX, I1_CY)]):
        queues = env.all_queues[i_idx]
        for d_idx, d in enumerate(['N', 'S', 'E', 'W']):
            q_len = max(0, int(queues[d_idx].length))
            dp = DIR_PARAMS[d]

            for j in range(min(q_len, 20)):
                if d in ('N', 'S'):
                    stop_y = cy + (STOP_LINE_OFFSET * (-dp['dy']))
                    y = stop_y + (j + 0.5) * veh_gap * (-dp['dy'])
                    x = cx + dp['x_offset']
                else:
                    stop_x = cx + (STOP_LINE_OFFSET * (-dp['dx']))
                    x = stop_x + (j + 0.5) * veh_gap * (-dp['dx'])
                    y = cy + dp.get('y_offset', 0)

                vehicles.append(VehicleSnapshot(
                    vid=vid, direction=d, x=float(x), y=float(y),
                    speed=0.0, waiting=True,
                    distance_to_stop=float(j * veh_gap),
                    intersection=i_idx,
                ))
                vid += 1

    return vehicles


# ══════════════════════════════════════════════════════════════════════════════
# 仿真运行器
# ══════════════════════════════════════════════════════════════════════════════

class MultiSimRunner:
    def __init__(self):
        self.env: Optional[MultiIntersectionEnv] = None
        self.agents: List[Any] = []
        self.agent_types: List[str] = []
        self.episode = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.obs = None
        self.last_actions = [0, 0]
        self.last_qs = [[0.0, 0.0], [0.0, 0.0]]
        self.last_policies = [[0.5, 0.5], [0.5, 0.5]]
        self.last_reward = 0.0

        self._play_thread: Optional[threading.Thread] = None
        self._play_speed = 1.0
        self._play_active = threading.Event()
        self._play_active.clear()
        self._play_running = False
        self._on_step_callback: Optional[Callable] = None

    # ── 初始化 ──
    def init(self, agent_type: str = 'dueling', pattern: str = 'uniform',
             state_dim: int = 8, max_steps: int = 2000,
             travel_delay: int = 3) -> MultiStepSnapshot:
        self.agent_types = [agent_type, agent_type]

        self.env = MultiIntersectionEnv(
            arrival_rates=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            traffic_pattern=pattern,
            state_dim=state_dim,
            max_steps=max_steps,
            travel_delay=travel_delay,
        )

        self.agents = []
        for _ in range(2):
            if agent_type in ('a2c', 'ppo'):
                ag = create_pg_agent(agent_type=agent_type, state_dim=state_dim)
            else:
                ag = create_agent(agent_type=agent_type, state_dim=state_dim)
            self.agents.append(ag)

        self.episode = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False

        self.obs, _ = self.env.reset()
        self.last_actions = [0, 0]
        self.last_qs = [[0.0, 0.0], [0.0, 0.0]]
        self.last_policies = [[0.5, 0.5], [0.5, 0.5]]

        return self._build_snapshot()

    def load_models(self, model_paths: List[str]) -> List[bool]:
        """为两个路口加载模型 (可不同)。"""
        import torch
        ok = []
        for i, path in enumerate(model_paths):
            if path and os.path.isfile(path):
                try:
                    ckpt = torch.load(path, map_location=self.agents[i].device, weights_only=False)
                    # 兼容两种格式: 完整checkpoint dict 或纯state_dict
                    if isinstance(ckpt, dict) and 'online_net' in ckpt:
                        self.agents[i].online_net.load_state_dict(ckpt['online_net'])
                        self.agents[i].target_net.load_state_dict(ckpt['target_net'])
                    else:
                        self.agents[i].online_net.load_state_dict(ckpt)
                        self.agents[i].target_net.load_state_dict(ckpt)
                    ok.append(True)
                except Exception as e:
                    print(f"[MultiSimRunner] load_models[{i}] failed: {e}")
                    ok.append(False)
            else:
                ok.append(False)
        return ok

    # ── 单步 ──
    def step(self) -> MultiStepSnapshot:
        if self.env is None or self.done:
            return self.reset()

        self._compute_qs_or_policies()

        actions = []
        for i in range(2):
            local_obs = self.env.get_local_obs(i)
            a = self.agents[i].select_action(local_obs, evaluate=True)
            actions.append(a)
        self.last_actions = actions

        next_obs, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated or truncated

        for i in range(2):
            local_obs = self.env.get_local_obs(i)
            next_local = self.env.get_local_obs(i)
            self.agents[i].store_transition(local_obs, actions[i], reward, next_local, done)
            if self.agent_types[i] not in ('a2c', 'ppo'):
                self.agents[i].update()

        self.obs = next_obs
        self.cumulative_reward += reward
        self.last_reward = float(reward)
        self.step_count += 1
        self.done = done

        return self._build_snapshot()

    def _compute_qs_or_policies(self) -> None:
        import torch
        self.last_qs = []
        self.last_policies = []
        for i in range(2):
            local_obs = self.env.get_local_obs(i)
            st = torch.FloatTensor(local_obs).unsqueeze(0).to(self.agents[i].device)
            with torch.no_grad():
                if self.agent_types[i] in ('a2c', 'ppo'):
                    logits = self.agents[i].actor(st)
                    probs = torch.softmax(logits, dim=-1)
                    pol = probs.squeeze().cpu().tolist()
                    self.last_policies.append(pol)
                    self.last_qs.append(pol)
                else:
                    q = self.agents[i].online_net(st)
                    q_list = q.squeeze().cpu().tolist()
                    self.last_qs.append(q_list)
                    # 温差软最大化: tau = max(1, Q值范围) 避免 Q 值差距过大时退化为 0/100%
                    q_tensor = q.squeeze()
                    q_range = (q_tensor.max() - q_tensor.min()).item()
                    tau = max(1.0, q_range)
                    probs = torch.softmax(q_tensor / tau, dim=-1)
                    self.last_policies.append(probs.cpu().tolist())

    def _build_snapshot(self) -> MultiStepSnapshot:
        if self.env is None:
            return MultiStepSnapshot(
                episode=0, step=0, phases=[0,0], phase_labels=['NS','NS'],
                actions=[0,0], vehicles=[], queues=[[0]*4,[0]*4],
                avg_wait=[[0]*4,[0]*4], q_values=[[0,0],[0,0]],
                policy=[[0,0],[0,0]], reward=0, cumulative_reward=0,
                throughputs=[0,0], switch_penalties=[0,0],
            )

        vehicles = _compute_multi_vehicle_positions(self.env)
        queues = [
            [int(self.env.all_queues[i][j].length) for j in range(4)]
            for i in range(2)
        ]
        avg_wait = [
            [self.env.all_queues[i][j].head_wait_time(self.env.global_step) for j in range(4)]
            for i in range(2)
        ]
        phases = self.env.all_phases

        return MultiStepSnapshot(
            episode=self.episode, step=self.step_count,
            phases=phases,
            phase_labels=['NS' if p == 0 else 'EW' for p in phases],
            actions=self.last_actions,
            vehicles=vehicles,
            queues=queues,
            avg_wait=avg_wait,
            q_values=self.last_qs,
            policy=self.last_policies,
            reward=self.last_reward,
            cumulative_reward=float(self.cumulative_reward),
            throughputs=self.env.all_total_departed,
            switch_penalties=[
                self.env.switch_penalty if a == 1 else 0.0
                for a in self.last_actions
            ],
            done=self.done,
            pipeline_count=len(self.env.pipelines),
        )

    def to_json(self, snap: MultiStepSnapshot) -> dict:
        return {
            "type": "state",
            "episode": snap.episode,
            "step": snap.step,
            "phases": snap.phases,
            "phase_labels": snap.phase_labels,
            "actions": snap.actions,
            "vehicles": [
                {"vid": v.vid, "dir": v.direction,
                 "x": round(v.x, 1), "y": round(v.y, 1),
                 "speed": round(v.speed, 1), "waiting": v.waiting,
                 "intersection": v.intersection}
                for v in snap.vehicles[:80]
            ],
            "queues": snap.queues,
            "avg_wait": [[round(w,1) for w in row] for row in snap.avg_wait],
            "q_values": [[round(v,4) for v in row] for row in snap.q_values],
            "policy": [[round(v,4) for v in row] for row in snap.policy],
            "reward": round(snap.reward, 1),
            "cumulative_reward": round(snap.cumulative_reward, 1),
            "throughputs": snap.throughputs,
            "switch_penalties": [round(p,1) for p in snap.switch_penalties],
            "done": snap.done,
            "pipeline_count": snap.pipeline_count,
        }

    # ── 播放控制 ──
    def play(self, speed: float = 1.0, on_step: Optional[Callable] = None) -> None:
        self._on_step_callback = on_step
        self._play_speed = speed
        self._play_active.set()
        if self._play_thread and self._play_thread.is_alive():
            return
        self._play_running = True
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _play_loop(self) -> None:
        while self._play_running:
            self._play_active.wait()
            if not self._play_running:
                break
            snap = self.step()
            if self._on_step_callback:
                self._on_step_callback(self.to_json(snap))
            if snap.done:
                self._play_active.clear()
            base_delay = 0.12
            time.sleep(base_delay / max(0.1, self._play_speed))

    def pause(self) -> None:
        self._play_active.clear()

    def stop(self) -> None:
        self._play_running = False
        self._play_active.set()

    def set_speed(self, speed: float) -> None:
        self._play_speed = speed

    def reset(self) -> MultiStepSnapshot:
        if self.env is None:
            return self.init()
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.episode += 1
        return self._build_snapshot()
