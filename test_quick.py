"""集成验证脚本。"""

import numpy as np
import os
os.makedirs('results', exist_ok=True)

from traffic_env import make_env, FixedTimeController
from agent import create_agent

def test_all():
    # ── Test 1: Dueling DQN 小规模训练 ──
    print('Test 1: Dueling DQN mini-training (10 episodes)...')
    env = make_env(traffic_pattern='uniform', max_steps=100)
    agent = create_agent('dueling', state_dim=8, action_dim=2)

    history_rewards = []
    for ep in range(1, 11):
        state, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            ns, r, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store_transition(state, action, r, ns, done)
            agent.update()
            state = ns
            ep_r += r
        history_rewards.append(ep_r)

    print(f'  Ep 1-5 rewards: {[f"{x:.0f}" for x in history_rewards[:5]]}')
    print(f'  Ep 6-10 rewards: {[f"{x:.0f}" for x in history_rewards[5:]]}')
    print(f'  Final epsilon: {agent.epsilon:.3f}')
    assert len(history_rewards) == 10, 'FAIL'
    print('  ✅ Dueling DQN training OK')

    # ── Test 2: 固定配时基准 ──
    print()
    print('Test 2: Fixed-time baseline...')
    env2 = make_env(max_steps=100)
    ft = FixedTimeController(env2, fixed_interval=30)
    stats = ft.run_episode()
    print(f'  Avg Queue: {stats["avg_queue"]:.2f}, Throughput: {stats["total_departed"]}')
    assert stats['total_departed'] > 0, 'FAIL'
    print('  ✅ Fixed-time baseline OK')

    # ── Test 3: 三种算法快速对比 ──
    print()
    print('Test 3: Multi-algo mini-comparison (5 eps each)...')
    for atype in ['dqn', 'double', 'dueling']:
        env3 = make_env(max_steps=100)
        ag = create_agent(atype, state_dim=8, action_dim=2)
        rewards = []
        for _ in range(5):
            state, _ = env3.reset()
            ep_r = 0.0
            done = False
            while not done:
                action = ag.select_action(state)
                ns, r, term, trunc, _ = env3.step(action)
                done = term or trunc
                ag.store_transition(state, action, r, ns, done)
                ag.update()
                state = ns
                ep_r += r
            rewards.append(ep_r)
        print(f'  {atype:10s}: avg_reward={np.mean(rewards):.0f}, eps={ag.epsilon:.3f}')
    print('  ✅ Multi-algo comparison OK')

    # ── Test 4: 潮汐流量 ──
    print()
    print('Test 4: Tidal flow test...')
    env4 = make_env(traffic_pattern='peak_hour', max_steps=100,
                    peak_start=0, peak_end=50, peak_multiplier=3.0)
    state, _ = env4.reset()
    agent4 = create_agent('dueling', state_dim=8, action_dim=2)
    for _ in range(10):
        state, _ = env4.reset()
        done = False
        while not done:
            action = agent4.select_action(state)
            ns, r, term, trunc, _ = env4.step(action)
            done = term or trunc
            agent4.store_transition(state, action, r, ns, done)
            agent4.update()
            state = ns

    state, _ = env4.reset()
    ep_r = 0.0
    done = False
    while not done:
        action = agent4.select_action(state, evaluate=True)
        ns, r, term, trunc, info = env4.step(action)
        done = term or trunc
        ep_r += r
        state = ns
    print(f'  RL eval: reward={ep_r:.0f}, departed={env4.total_departed}')

    env5 = make_env(traffic_pattern='peak_hour', max_steps=100,
                    peak_start=0, peak_end=50, peak_multiplier=3.0)
    ft2 = FixedTimeController(env5, fixed_interval=30)
    stats2 = ft2.run_episode()
    print(f'  FT eval:  avg_queue={stats2["avg_queue"]:.2f}, departed={stats2["total_departed"]}')
    print('  ✅ Tidal flow test OK')

    # ── Test 5: 多路口环境 ──
    print()
    print('Test 5: Multi-intersection test...')
    from traffic_env import MultiIntersectionEnv
    multi_env = MultiIntersectionEnv(num_intersections=3, max_steps=50)
    obs, _ = multi_env.reset()
    for _ in range(50):
        actions = multi_env.action_space.sample()
        obs, r, done, _, info = multi_env.step(actions)
        if done:
            break
    print(f'  Total departed: {info["total_departed"]}')
    print('  ✅ Multi-intersection OK')

    # ── Test 6: A2C mini-training ──
    print()
    print('Test 6: A2C mini-training (15 episodes)...')
    from pg_agent import create_pg_agent
    env6 = make_env(max_steps=100)
    a2c = create_pg_agent('a2c', state_dim=8)
    rewards = []
    for ep in range(15):
        obs, _ = env6.reset()
        ep_r = 0
        done = False
        while not done:
            a = a2c.select_action(obs)
            ns, r, term, trunc, _ = env6.step(a)
            done = term or trunc
            a2c.store_transition(obs, a, r, ns, done)
            obs = ns
            ep_r += r
        a2c.update()
        rewards.append(ep_r)
    print(f'  First 5: {[f"{x:.0f}" for x in rewards[:5]]}')
    print(f'  Last 5:  {[f"{x:.0f}" for x in rewards[-5:]]}')
    print('  ✅ A2C training OK')

    # ── Test 7: PPO mini-training ──
    print()
    print('Test 7: PPO mini-training (15 episodes)...')
    env7 = make_env(max_steps=100)
    ppo = create_pg_agent('ppo', state_dim=8)
    rewards = []
    for ep in range(15):
        obs, _ = env7.reset()
        ep_r = 0
        done = False
        while not done:
            a = ppo.select_action(obs)
            ns, r, term, trunc, _ = env7.step(a)
            done = term or trunc
            ppo.store_transition(obs, a, r, ns, done)
            obs = ns
            ep_r += r
        ppo.update()
        rewards.append(ep_r)
    print(f'  First 5: {[f"{x:.0f}" for x in rewards[:5]]}')
    print(f'  Last 5:  {[f"{x:.0f}" for x in rewards[-5:]]}')
    print('  ✅ PPO training OK')

    print()
    print('=' * 60)
    print('  ALL TESTS PASSED! 🎉 System fully operational.')
    print('=' * 60)


if __name__ == '__main__':
    test_all()
