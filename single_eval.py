"""单路口多模式评估脚本。"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from traffic_env import make_env
from agent import create_agent
from pg_agent import create_pg_agent

AGENTS = ["dqn", "double", "dueling", "noisy", "boltzmann", "per_dqn", "a2c", "ppo"]
PATTERNS = ["low_traffic", "peak_hour", "tidal", "burst"]
EPISODES = 100
MAX_STEPS = 300
MODELS_DIR = os.path.join("results", "models")
OUTPUT_FILE = os.path.join("results", "single_multi_pattern_eval.json")

def create_eval_agent(agent_type: str, state_dim=8):
    if agent_type in ("a2c", "ppo"):
        return create_pg_agent(agent_type=agent_type, state_dim=state_dim)
    return create_agent(agent_type=agent_type, state_dim=state_dim)


def evaluate_one(agent_type: str, pattern: str):
    """加载 {agent_type}_{pattern}.pt 模型，在指定模式下推理评估。"""
    model_path = os.path.join(MODELS_DIR, f"{agent_type}_{pattern}.pt")
    if not os.path.isfile(model_path):
        print(f"  [SKIP] 模型不存在: {model_path}")
        return None

    env = make_env(traffic_pattern=pattern, max_steps=MAX_STEPS)
    agent = create_eval_agent(agent_type)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "online_net" in ckpt:
        agent.online_net.load_state_dict(ckpt["online_net"])
        agent.online_net.eval()
    else:
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.actor.eval()
        agent.critic.eval()

    rewards, queues, waits, throughputs = [], [], [], []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        total_r, total_q, total_w = 0.0, 0.0, 0.0
        steps = 0
        for _ in range(MAX_STEPS):
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
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
        throughputs.append(env.total_departed)

    return {
        "reward":     {"mean": float(np.mean(rewards)),     "std": float(np.std(rewards))},
        "queue":      {"mean": float(np.mean(queues)),      "std": float(np.std(queues))},
        "wait":       {"mean": float(np.mean(waits)),       "std": float(np.std(waits))},
        "throughput": {"mean": float(np.mean(throughputs)), "std": float(np.std(throughputs))},
    }


if __name__ == "__main__":
    t0 = time.time()
    results = {}

    for agent in AGENTS:
        results[agent] = {}
        for pat in PATTERNS:
            label = f"{agent}/{pat}"
            print(f"\n[{label}] 评估中...", flush=True)
            res = evaluate_one(agent, pat)
            if res is not None:
                results[agent][pat] = res
                print(f"  R={res['reward']['mean']:.1f}  Q={res['queue']['mean']:.1f}  "
                      f"W={res['wait']['mean']:.1f}  T={res['throughput']['mean']:.0f}", flush=True)
            else:
                results[agent][pat] = None
                print(f"  [SKIP]", flush=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    results["params"] = {"episodes": EPISODES, "max_steps": MAX_STEPS}

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done in {results['elapsed_sec']}s  ->  {OUTPUT_FILE}")
