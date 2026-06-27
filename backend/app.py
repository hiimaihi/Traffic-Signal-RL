"""
交通信号灯 RL — Flask 后端 API 服务器
=====================================
提供:
  - /         → 前端模拟驾驶舱
  - /ws       → WebSocket 双向通信 (仿真控制 + 状态推送)
  - /api/*    → REST 端点
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
from sim_runner import SimRunner
from multi_sim_runner import MultiSimRunner
from grid_sim_runner import GridSimRunner

app = Flask(__name__)
sock = Sock(app)

# 全局仿真实例
runner = SimRunner()
multi_runner = MultiSimRunner()
grid_runner = GridSimRunner()
ws_clients = set()


# ══════════════════════════════════════════════════════════════════════════════
# 页面路由
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """模拟驾驶舱主页面。"""
    return render_template('index.html')


@app.route('/multi')
def multi_index():
    """多路口模拟驾驶舱。"""
    return render_template('multi_index.html')


@app.route('/grid')
def grid_index():
    """2×2 网格模拟驾驶舱。"""
    return render_template('grid_index.html')


# ══════════════════════════════════════════════════════════════════════════════
# REST API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/init', methods=['POST'])
def api_init():
    """初始化仿真。"""
    data = request.get_json(force=True)
    snap = runner.init(
        agent_type=data.get('agent', 'dueling'),
        pattern=data.get('pattern', 'uniform'),
        state_dim=data.get('state_dim', 8),
        reward_type=data.get('reward', 'composite'),
        max_steps=data.get('max_steps', 2000),
    )
    model_path = data.get('model_path') or _resolve_model(data.get('agent', 'dueling'), data.get('pattern', 'uniform'))
    if model_path and os.path.isfile(model_path):
        runner.load_model(model_path)
    return app.response_class(
        response=json.dumps(runner.to_json(snap)),
        status=200,
        mimetype='application/json',
    )


from typing import Optional

def _resolve_model(agent_type: str, pattern: str) -> Optional[str]:
    """在 results/models/ 中查找匹配的模型文件。"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "models")
    candidates = [
        f"{agent_type}_{pattern}.pt",
        f"{agent_type}_{pattern}.pth",
    ]
    for c in candidates:
        full = os.path.join(models_dir, c)
        if os.path.isfile(full):
            return full
    return None


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置当前 episode。"""
    snap = runner.reset()
    return jsonify(runner.to_json(snap))


@app.route('/api/agents', methods=['GET'])
def api_agents():
    """可用 Agent 列表。"""
    return jsonify({
        "dqn": ["dqn", "double", "dueling", "noisy", "noisy_double",
                "boltzmann", "boltzmann_double", "per", "dueling_per"],
        "pg": ["a2c", "ppo"],
    })


@app.route('/api/patterns', methods=['GET'])
def api_patterns():
    """可用交通模式。"""
    return jsonify({
        "patterns": ["uniform", "peak_hour", "tidal", "burst", "low_traffic"],
    })


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """算法对比: 在相同流量模式下快速评估多个 Agent。"""
    import numpy as np
    data = request.get_json(force=True)
    pattern = data.get('pattern', 'low_traffic')
    agents = data.get('agents', ['dueling', 'dqn', 'double'])
    steps = min(data.get('steps', 200), 500)

    results = []
    for agent_type in agents:
        try:
            r = SimRunner()
            r.init(agent_type=agent_type, pattern=pattern, max_steps=steps + 50)
            model_path = _resolve_model(agent_type, pattern)
            if model_path and os.path.isfile(model_path):
                r.load_model(model_path)
            total_q = 0
            total_wait = 0.0
            switches = 0
            for _ in range(steps):
                snap = r.step()
                total_q += sum(snap.queues)
                total_wait += float(np.mean(snap.avg_wait)) if snap.avg_wait else 0.0
                if snap.action == 1:
                    switches += 1
            results.append({
                "agent": agent_type,
                "avg_queue": round(total_q / steps, 1),
                "avg_wait": round(total_wait / steps, 1),
                "throughput": snap.throughput,
                "reward": round(snap.cumulative_reward, 0),
                "switches": switches,
                "switch_rate": round(switches / steps * 100, 1),
            })
        except Exception as e:
            results.append({"agent": agent_type, "error": str(e)})

    # 找最佳
    valid = [r for r in results if 'error' not in r]
    if valid:
        best = min(valid, key=lambda x: x['avg_queue'])
        best['_best'] = True
    return jsonify({"pattern": pattern, "steps": steps, "results": results})


@app.route('/api/models', methods=['GET'])
def api_models():
    """列出已训练的可加载模型。"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "models")
    models = []
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith('.pt') or f.endswith('.pth'):
                full = os.path.join(models_dir, f)
                size_kb = os.path.getsize(full) / 1024
                # 解析 agent_type 和 pattern
                stem = f.rsplit('.', 1)[0]
                parts = stem.split('_', 1)
                agent = parts[0] if parts else 'unknown'
                pattern = parts[1] if len(parts) > 1 else 'uniform'
                models.append({
                    "filename": f,
                    "path": full,
                    "agent": agent,
                    "pattern": pattern,
                    "size_kb": round(size_kb, 1),
                })
    return jsonify({"models": models})


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ══════════════════════════════════════════════════════════════════════════════

@sock.route('/ws')
def ws_handler(ws):
    """WebSocket 双向通信。"""
    ws_clients.add(ws)
    player_id = None
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            data = json.loads(msg)
            cmd = data.get('cmd', '')

            if cmd == 'init':
                # 初始化, 自动加载预训练模型
                agent_type = data.get('agent', 'dueling')
                pattern = data.get('pattern', 'uniform')
                snap = runner.init(
                    agent_type=agent_type,
                    pattern=pattern,
                    state_dim=data.get('state_dim', 8),
                    reward_type=data.get('reward', 'composite'),
                )
                model_path = data.get('model_path') or _resolve_model(agent_type, pattern)
                if model_path and os.path.isfile(model_path):
                    ok = runner.load_model(model_path)
                    ws.send(json.dumps({"type": "control", "model_loaded": ok, "path": model_path}) + "\n")
                ws.send(json.dumps(runner.to_json(snap)) + "\n")

            elif cmd == 'step':
                # 单步
                snap = runner.step()
                ws.send(json.dumps(runner.to_json(snap)) + "\n")

            elif cmd == 'reset':
                snap = runner.reset()
                ws.send(json.dumps(runner.to_json(snap)) + "\n")

            elif cmd == 'play':
                speed = data.get('speed', 1.0)
                def on_step(state_json):
                    try:
                        ws.send(json.dumps(state_json) + "\n")
                    except Exception:
                        runner.stop()
                runner.play(speed=speed, on_step=on_step)
                ws.send(json.dumps({"type": "control", "status": "playing"}) + "\n")

            elif cmd == 'pause':
                runner.pause()
                ws.send(json.dumps({"type": "control", "status": "paused"}) + "\n")

            elif cmd == 'stop':
                runner.stop()
                ws.send(json.dumps({"type": "control", "status": "stopped"}) + "\n")

            elif cmd == 'speed':
                runner.set_speed(data.get('speed', 1.0))
                ws.send(json.dumps({"type": "control", "speed": data.get('speed', 1.0)}) + "\n")

            elif cmd == 'load_model':
                path = data.get('path', '')
                ok = runner.load_model(path)
                ws.send(json.dumps({"type": "control", "model_loaded": ok, "path": path}) + "\n")

            else:
                ws.send(json.dumps({"type": "error", "msg": f"Unknown cmd: {cmd}"}) + "\n")

    except Exception as e:
        print(f"WS error: {e}")
    finally:
        ws_clients.discard(ws)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — 多路口
# ══════════════════════════════════════════════════════════════════════════════

@sock.route('/ws_multi')
def ws_multi_handler(ws):
    """多路口 WebSocket 双向通信。"""
    ws_clients.add(ws)
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            data = json.loads(msg)
            cmd = data.get('cmd', '')

            if cmd == 'init':
                agent_type = data.get('agent', 'dueling')
                pattern = data.get('pattern', 'uniform')
                snap = multi_runner.init(
                    agent_type=agent_type,
                    pattern=pattern,
                    state_dim=data.get('state_dim', 8),
                    max_steps=data.get('max_steps', 2000),
                )
                # 加载两路口模型 (支持单路口模型复用到两路口)
                model_path = data.get('model_path') or _resolve_model(agent_type, pattern)
                if model_path and os.path.isfile(model_path):
                    ok = multi_runner.load_models([model_path, model_path])
                else:
                    ok = [False, False]
                ws.send(json.dumps({"type": "control", "models_loaded": ok}) + "\n")
                ws.send(json.dumps(multi_runner.to_json(snap)) + "\n")

            elif cmd == 'step':
                snap = multi_runner.step()
                ws.send(json.dumps(multi_runner.to_json(snap)) + "\n")

            elif cmd == 'reset':
                snap = multi_runner.reset()
                ws.send(json.dumps(multi_runner.to_json(snap)) + "\n")

            elif cmd == 'play':
                speed = data.get('speed', 1.0)
                def on_step(state_json):
                    try:
                        ws.send(json.dumps(state_json) + "\n")
                    except Exception:
                        multi_runner.stop()
                multi_runner.play(speed=speed, on_step=on_step)
                ws.send(json.dumps({"type": "control", "status": "playing"}) + "\n")

            elif cmd == 'pause':
                multi_runner.pause()
                ws.send(json.dumps({"type": "control", "status": "paused"}) + "\n")

            elif cmd == 'stop':
                multi_runner.stop()
                ws.send(json.dumps({"type": "control", "status": "stopped"}) + "\n")

            elif cmd == 'speed':
                multi_runner.set_speed(data.get('speed', 1.0))
                ws.send(json.dumps({"type": "control", "speed": data.get('speed', 1.0)}) + "\n")

            else:
                ws.send(json.dumps({"type": "error", "msg": f"Unknown cmd: {cmd}"}) + "\n")

    except Exception as e:
        print(f"WS Multi error: {e}")
    finally:
        ws_clients.discard(ws)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — 2×2 网格
# ══════════════════════════════════════════════════════════════════════════════

@sock.route('/ws_grid')
def ws_grid_handler(ws):
    """2×2 网格 WebSocket 双向通信。"""
    ws_clients.add(ws)
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            data = json.loads(msg)
            cmd = data.get('cmd', '')

            if cmd == 'init':
                agent_type = data.get('agent', 'dueling')
                pattern = data.get('pattern', 'uniform')
                snap = grid_runner.init(
                    agent_type=agent_type,
                    pattern=pattern,
                    state_dim=data.get('state_dim', 8),
                    max_steps=data.get('max_steps', 2000),
                )
                model_path = data.get('model_path') or _resolve_model(agent_type, pattern)
                if model_path and os.path.isfile(model_path):
                    ok = grid_runner.load_models([model_path] * 4)
                else:
                    # 尝试加载 grid_ 前缀模型
                    grid_model_path = data.get('model_path') or _resolve_grid_model(agent_type, pattern)
                    if grid_model_path and os.path.isfile(grid_model_path):
                        ok = grid_runner.load_models([grid_model_path] * 4)
                    else:
                        ok = [False] * 4
                ws.send(json.dumps({"type": "control", "models_loaded": ok}) + "\n")
                ws.send(json.dumps(grid_runner.to_json(snap)) + "\n")

            elif cmd == 'step':
                snap = grid_runner.step()
                ws.send(json.dumps(grid_runner.to_json(snap)) + "\n")

            elif cmd == 'reset':
                snap = grid_runner.reset()
                ws.send(json.dumps(grid_runner.to_json(snap)) + "\n")

            elif cmd == 'play':
                speed = data.get('speed', 1.0)
                def on_step(state_json):
                    try:
                        ws.send(json.dumps(state_json) + "\n")
                    except Exception:
                        grid_runner.stop()
                grid_runner.play(speed=speed, on_step=on_step)
                ws.send(json.dumps({"type": "control", "status": "playing"}) + "\n")

            elif cmd == 'pause':
                grid_runner.pause()
                ws.send(json.dumps({"type": "control", "status": "paused"}) + "\n")

            elif cmd == 'stop':
                grid_runner.stop()
                ws.send(json.dumps({"type": "control", "status": "stopped"}) + "\n")

            elif cmd == 'speed':
                grid_runner.set_speed(data.get('speed', 1.0))
                ws.send(json.dumps({"type": "control", "speed": data.get('speed', 1.0)}) + "\n")

            else:
                ws.send(json.dumps({"type": "error", "msg": f"Unknown cmd: {cmd}"}) + "\n")

    except Exception as e:
        print(f"WS Grid error: {e}")
    finally:
        ws_clients.discard(ws)


def _resolve_grid_model(agent_type: str, pattern: str) -> Optional[str]:
    """在 results/models/ 中查找 grid_ 前缀模型。"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "models")
    candidates = [
        f"grid_{agent_type}_{pattern}.pt",
        f"{agent_type}_{pattern}.pt",  # fallback to single model
    ]
    for c in candidates:
        full = os.path.join(models_dir, c)
        if os.path.isfile(full):
            return full
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Traffic Signal RL Simulator Backend")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"🎮 Traffic RL Cockpit → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
