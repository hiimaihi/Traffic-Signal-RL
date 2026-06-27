"""2×2 网格 Dueling DQN 评估脚本。"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from grid_2x2_env import Grid2x2Env
from agent import create_agent

PATTERNS = ["uniform", "low_traffic", "peak_hour", "tidal", "burst"]
EPISODES = 100
MAX_STEPS = 300
MODELS_DIR = os.path.join("results", "models")
OUTPUT_FILE = os.path.join("results", "grid_eval_results.json")

def evaluate_pattern(agent_type, pattern):
    """加载模型，在指定模式下推理评估。"""
    model_path = os.path.join(MODELS_DIR, f"grid_{agent_type}_{pattern}.pt")
    if not os.path.isfile(model_path):
        print(f"  [SKIP] 模型不存在: {model_path}")
        return None

    env = Grid2x2Env(
        arrival_rates=(2.0,) * 8,
        traffic_pattern=pattern,
        max_steps=MAX_STEPS,
    )

    agents = [create_agent(agent_type=agent_type, state_dim=8) for _ in range(4)]

    # 加载模型权重
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    for ag in agents:
        ag.online_net.load_state_dict(ckpt["online_net"])
        ag.online_net.eval()

    rewards, queues, waits, throughputs = [], [], [], []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        total_r, total_q, total_w = 0.0, 0.0, 0.0
        steps = 0
        for _ in range(MAX_STEPS):
            prev_locals = [env.get_local_obs(i) for i in range(4)]
            actions = [agents[i].select_action(prev_locals[i], evaluate=True) for i in range(4)]
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            obs = next_obs
            total_r += reward
            total_q += info["total_queue"]
            total_w += info["avg_wait"]
            steps += 1
            if done:
                break

        rewards.append(total_r)
        queues.append(total_q / max(steps, 1))
        waits.append(total_w / max(steps, 1))
        throughputs.append(info["total_departed"])

    result = {
        "reward": {"mean": float(np.mean(rewards)), "std": float(np.std(rewards))},
        "queue":  {"mean": float(np.mean(queues)),  "std": float(np.std(queues))},
        "wait":   {"mean": float(np.mean(waits)),   "std": float(np.std(waits))},
        "throughput": {"mean": float(np.mean(throughputs)), "std": float(np.std(throughputs))},
    }
    return result


if __name__ == "__main__":
    AGENT = "dueling"
    results = {}
    t0 = time.time()

    for pat in PATTERNS:
        print(f"\n[{pat}] 评估中...", flush=True)
        res = evaluate_pattern(AGENT, pat)
        if res is not None:
            results[pat] = res
            print(f"  Reward={res['reward']['mean']:.1f} +/- {res['reward']['std']:.1f}  "
                  f"Queue={res['queue']['mean']:.1f}  Wait={res['wait']['mean']:.1f}  "
                  f"Thru={res['throughput']['mean']:.0f}", flush=True)
        else:
            results[pat] = None

    results["elapsed_sec"] = round(time.time() - t0, 1)
    results["params"] = {"agent": AGENT, "episodes": EPISODES, "max_steps": MAX_STEPS}

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {results['elapsed_sec']}s  ->  {OUTPUT_FILE}")
