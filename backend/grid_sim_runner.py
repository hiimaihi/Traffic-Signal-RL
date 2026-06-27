"""
交通信号灯 RL — 2×2 网格实时仿真运行器
=====================================
负责:
  - 初始化 Grid2x2Env + 4 个 RL Agent
  - 逐步执行仿真, 生成每步状态快照 (含四路口)
  - 输出车辆坐标用于 Canvas 渲染
  - 支持 play/pause/step/speed 控制
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, threading, numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple

from grid_2x2_env import Grid2x2Env
from agent import create_agent
from pg_agent import create_pg_agent

# ══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridVehicleSnapshot:
    vid: int
    direction: str
    x: float
    y: float
    speed: float
    waiting: bool
    distance_to_stop: float
    intersection: int = 0


@dataclass
class GridStepSnapshot:
    episode: int
    step: int
    phases: List[int]
    phase_labels: List[str]
    actions: List[int]
    vehicles: List[GridVehicleSnapshot]
    queues: List[List[int]]       # [[N,S,E,W]×4]
    avg_wait: List[List[float]]
    q_values: List[List[float]]
    policy: List[List[float]]
    reward: float
    cumulative_reward: float
    throughputs: List[int]
    switch_penalties: List[float]
    done: bool = False
    pipeline_count: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Canvas 坐标映射 (800×640, 4 路口排列)
# ══════════════════════════════════════════════════════════════════════════════

CANVAS_W = 800
CANVAS_H = 640
# 四个路口中心: I0(NW), I1(NE), I2(SW), I3(SE)
CX = [200, 600, 200, 600]
CY = [210, 210, 430, 430]
ROAD_W = 60
STOP = 40
VEH_LEN = 14
VEH_WID = 8
VEH_GAP = VEH_LEN + 5


def _compute_grid_vehicle_positions(env: Grid2x2Env) -> List[GridVehicleSnapshot]:
    """计算所有车辆在 2×2 Canvas 上的坐标。"""
    vehicles = []
    vid = 0

    for i in range(4):
        cx = CX[i]
        cy = CY[i]
        for d_idx, d in enumerate(['N', 'S', 'E', 'W']):
            queue_len = max(0, int(env.all_queues[i][d_idx].length))
            for j in range(min(queue_len, 15)):
                # 车辆从停车线向外排列
                if d == 'N':
                    x = cx + 10
                    y = cy + STOP + j * VEH_GAP
                elif d == 'S':
                    x = cx - 10
                    y = cy - STOP - j * VEH_GAP
                elif d == 'E':
                    x = cx - STOP - j * VEH_GAP
                    y = cy + 10
                else:  # W
                    x = cx + STOP + j * VEH_GAP
                    y = cy - 10

                vehicles.append(GridVehicleSnapshot(
                    vid=vid, direction=d,
                    x=float(x), y=float(y),
                    speed=0.0, waiting=True,
                    distance_to_stop=float(j * VEH_GAP),
                    intersection=i,
                ))
                vid += 1

            # 行驶中的车 (简单估算)
            moving = min(2, int(env.arrival_rates[min(i*2+d_idx, 7)] * 2)) if queue_len < 8 else 0
            for j in range(moving):
                if d == 'N':
                    x = cx + 10
                    y = cy + STOP - (j+1)*35
                elif d == 'S':
                    x = cx - 10
                    y = cy - STOP + (j+1)*35
                elif d == 'E':
                    x = cx - STOP + (j+1)*35
                    y = cy + 10
                else:
                    x = cx + STOP - (j+1)*35
                    y = cy - 10

                vehicles.append(GridVehicleSnapshot(
                    vid=vid, direction=d,
                    x=float(x), y=float(y),
                    speed=2.0, waiting=False,
                    distance_to_stop=-float((j+1)*35),
                    intersection=i,
                ))
                vid += 1

    return vehicles


# ══════════════════════════════════════════════════════════════════════════════
# 网格仿真运行器
# ══════════════════════════════════════════════════════════════════════════════

class GridSimRunner:
    def __init__(self):
        self.env: Optional[Grid2x2Env] = None
        self.agents: List[Any] = []
        self.agent_types: List[str] = []
        self.episode = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.obs = None
        self.last_actions = [0, 0, 0, 0]
        self.last_qs = [[0.0, 0.0]] * 4
        self.last_policies = [[0.5, 0.5]] * 4
        self.last_reward = 0.0

        self._play_thread: Optional[threading.Thread] = None
        self._play_speed: float = 1.0
        self._play_active = threading.Event()
        self._play_active.clear()
        self._play_running = False
        self._on_step_callback: Optional[Callable] = None

    def init(self, agent_type: str = 'dueling', pattern: str = 'uniform',
             state_dim: int = 8, max_steps: int = 2000,
             travel_delay: int = 3) -> GridStepSnapshot:
        self.agent_types = [agent_type] * 4

        self.env = Grid2x2Env(
            arrival_rates=(2.0,) * 8,
            traffic_pattern=pattern,
            state_dim=state_dim,
            max_steps=max_steps,
            travel_delay=travel_delay,
        )

        self.agents = []
        for _ in range(4):
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
        self.last_actions = [0, 0, 0, 0]
        self.last_qs = [[0.0, 0.0]] * 4
        self.last_policies = [[0.5, 0.5]] * 4

        return self._build_snapshot()

    def load_models(self, model_paths: List[str]) -> List[bool]:
        import torch
        ok = []
        for i, path in enumerate(model_paths):
            if path and os.path.isfile(path):
                try:
                    ckpt = torch.load(path, map_location=self.agents[i].device, weights_only=False)
                    if isinstance(ckpt, dict) and 'online_net' in ckpt:
                        self.agents[i].online_net.load_state_dict(ckpt['online_net'])
                        self.agents[i].target_net.load_state_dict(ckpt['target_net'])
                    else:
                        self.agents[i].online_net.load_state_dict(ckpt)
                        self.agents[i].target_net.load_state_dict(ckpt)
                    ok.append(True)
                except Exception as e:
                    print(f"[GridSimRunner] load_models[{i}] failed: {e}")
                    ok.append(False)
            else:
                ok.append(False)
        return ok

    def step(self) -> GridStepSnapshot:
        if self.env is None or self.done:
            return self.reset()

        self._compute_qs_or_policies()

        actions = []
        for i in range(4):
            local_obs = self.env.get_local_obs(i)
            a = self.agents[i].select_action(local_obs, evaluate=True)
            actions.append(a)
        self.last_actions = actions

        next_obs, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated or truncated

        for i in range(4):
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
        for i in range(4):
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
                    q_tensor = q.squeeze()
                    q_range = (q_tensor.max() - q_tensor.min()).item()
                    tau = max(1.0, q_range)
                    probs = torch.softmax(q_tensor / tau, dim=-1)
                    self.last_policies.append(probs.cpu().tolist())

    def _build_snapshot(self) -> GridStepSnapshot:
        if self.env is None:
            return GridStepSnapshot(
                episode=0, step=0, phases=[0]*4, phase_labels=['NS']*4,
                actions=[0]*4, vehicles=[], queues=[[0]*4]*4,
                avg_wait=[[0]*4]*4, q_values=[[0,0]]*4,
                policy=[[0,0]]*4, reward=0, cumulative_reward=0,
                throughputs=[0]*4, switch_penalties=[0]*4,
            )

        vehicles = _compute_grid_vehicle_positions(self.env)
        queues = [
            [int(self.env.all_queues[i][j].length) for j in range(4)]
            for i in range(4)
        ]
        avg_wait = [
            [self.env.all_queues[i][j].head_wait_time(self.env.global_step) for j in range(4)]
            for i in range(4)
        ]
        phases = self.env.all_phases

        return GridStepSnapshot(
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

    def to_json(self, snap: GridStepSnapshot) -> dict:
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
                for v in snap.vehicles[:120]
            ],
            "queues": snap.queues,
            "avg_wait": [[round(w, 1) for w in row] for row in snap.avg_wait],
            "q_values": [[round(v, 4) for v in row] for row in snap.q_values],
            "policy": [[round(v, 4) for v in row] for row in snap.policy],
            "reward": round(snap.reward, 1),
            "cumulative_reward": round(snap.cumulative_reward, 1),
            "throughputs": snap.throughputs,
            "switch_penalties": [round(p, 1) for p in snap.switch_penalties],
            "done": snap.done,
            "pipeline_count": snap.pipeline_count,
        }

    def reset(self) -> GridStepSnapshot:
        if self.env:
            self.obs, _ = self.env.reset()
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.last_actions = [0, 0, 0, 0]
        self.last_qs = [[0.0, 0.0]] * 4
        self.last_policies = [[0.5, 0.5]] * 4
        self.last_reward = 0.0
        return self._build_snapshot()

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
            if self.done:
                self._play_active.clear()
                continue
            snap = self.step()
            if self._on_step_callback:
                self._on_step_callback(self.to_json(snap))
            delay = max(0.02, 0.5 / max(0.1, self._play_speed))
            time.sleep(delay)

    def pause(self) -> None:
        self._play_active.clear()

    def stop(self) -> None:
        self._play_running = False
        self._play_active.set()
        if self._play_thread:
            self._play_thread.join(timeout=1.0)
            self._play_thread = None

    def set_speed(self, speed: float) -> None:
        self._play_speed = speed
