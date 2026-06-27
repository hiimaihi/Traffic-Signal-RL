# 交通信号灯强化学习自适应控制系统 — 扩展需求文档

> **硕士学位论文级学术实验框架 v2.0**
> 包含 6 大类消融实验、5 种可视化仪表盘、统计显著性检验、多路口绿波协同

---

## 1. 项目概述

### 1.1 背景与动机
城市交通拥堵是现代城市化进程中的核心痛点。传统固定配时红绿灯无法响应实时流量变化，导致：低流量方向空等绿灯、高流量方向排队溢出、系统吞吐量低下。强化学习通过与环境持续交互学习最优策略，天然适合解决此类序贯决策问题。

### 1.2 核心贡献 (学术论文亮点)
1. **完整的 RL 算法族实现与对比**：DQN / Double DQN / Dueling DQN / NoisyNet / Boltzmann DQN / PER DQN + **A2C / PPO** (Policy Gradient 族)
2. **经验回放优化**：Uniform Replay vs Prioritized Experience Replay (PER)
3. **探索策略对比**：ε-greedy vs Boltzmann Softmax vs Noisy Networks
4. **多维度消融实验**：网络架构、状态表示、奖励函数、超参数敏感性、交通模式鲁棒性
5. **MDP 决策过程可视化**：Q 值热力图、相位决策时间线、排队动态动画、多算法雷达图
6. **多路口绿波协同**：三路口干线的独立 MARL 控制
7. **统计显著性检验**：多种子 t-test、95% 置信区间、Cohen's d 效应量
8. **🆕 交互式前端模拟驾驶舱**：Canvas 实时交通动画 + 决策过程可视化 + Apple/Claude 极简美学风格

| 用户原始需求 | 已实现 / 扩展 |
|---|---|
| Dueling DQN 基础训练 | ✔ + 6 DQN + 2 PG Agent 变体 |
| 潮汐车流测试 | ✔ + 5 种流量模式 |
| 三算法对比 | ✔ + 8 种算法横向对比 |
| 基础训练曲线 | ✔ + 6 类学术级可视化 |
| 多路口扩展 | ✔ + 绿波协同增强 |
| — 消融实验 | ✔ 新增 6 大类 × 多组 |
| — 统计检验 | ✔ 新增 t-test + CI + Cohen's d |
| — 决策仪表盘 | ✔ 新增 Q 热力图 + 相位时间线 + 动画 |

---

## 2. 系统架构 (模块职责)

```
RL/
├── REQUIREMENTS.md       # 本文档
├── traffic_env.py        # 5种流量模式, 3种状态空间, 4种奖励函数, 多路口
├── network.py            # StandardDQN, DuelingDQN, NoisyDQN, ShallowDQN, DeepDQN, Actor, Critic
├── agent.py              # 9种DQN Agent, Uniform/PER Replay, 3种探索策略
├── pg_agent.py           # A2C + PPO (Policy Gradient 族, RolloutBuffer)
├── visualization.py      # 7种学术级图表: Q热力图, 相位线, 雷达图, 带状图, 策略熵
├── experiments.py        # 6类消融实验 + 统计检验(t-test/CI/Cohen's d) + LaTeX报告
├── main.py               # 主入口: 完整实验编排, 命令行接口
├── test_quick.py         # 7项快速集成测试
├── results/              # 输出: PNG图表 + JSON + LaTeX表格
└── backend/              # 🆕 前端模拟驾驶舱
    ├── app.py            # Flask + WebSocket 服务器
    ├── sim_runner.py     # 仿真运行器
    └── templates/static/ # 前端资源 (HTML/CSS/JS)
```

---

## 3. 环境扩展 (traffic_env.py)

### 3.1 状态空间变体 (状态表示消融实验用)

| 变体 | 维度 | 内容 |
|------|------|------|
| **S4** (Queue-only) | 4 | 仅四个方向排队长度 L_i |
| **S8** (Queue+Wait) | 8 | L_i + 队首等待时间 T_i (当前默认) |
| **S9** (Full) | 9 | S8 + 当前相位 one-hot |

### 3.2 流量模式 (流量鲁棒性消融)

| 模式 | 描述 |
|------|------|
| `uniform` | 四方向等概率 λ=2.0 |
| `peak_hour` | 前半段 NS×3, 后半段均匀 |
| `tidal` | 周期交替 NS/EW 高峰 |
| `burst` | 随机突发事件 (每 200 步投放一波 20 辆车) |
| `low_traffic` | 全天低流量 (λ=0.5), 测试是否过度切换 |

### 3.3 奖励函数变体 (奖励设计消融)

| 变体 | 公式 |
|------|------|
| **R1** Queue-only | $R = -\sum L_i$ |
| **R2** Queue+Switch | $R = -\sum L_i - p \cdot \mathbb{1}_{\text{switch}}$ |
| **R3** Queue+Wait | $R = -(w_1\sum L_i + w_2\sum T_i)$ |
| **R4** Full (默认) | $R = -(w_1\sum L_i + w_2\sum T_i) - p \cdot \mathbb{1}_{\text{switch}}$ |

### 3.4 动作空间
- `0`：保持当前相位
- `1`：切换相位（触发黄灯 → 新相位生效）

---

## 4. 神经网络扩展 (network.py)

### 4.1 架构变体 (网络结构消融)

| 网络 | 参数量 | 特点 |
|------|--------|------|
| `ShallowDQN` | ~2.6K | 1 层 FC(32) — 测试欠拟合 |
| `StandardDQN` | ~17.9K | 2 层 FC(128) |
| `DeepDQN` | ~67K | 4 层 FC(256) — 测试过拟合 |
| `DuelingDQN` | ~34.4K | V/A 双流, 2 层共享 + 各 2 层分支 |
| `NoisyDQN` | ~34.4K | Dueling 架构 + NoisyLinear 替换标准 Linear |

### 4.2 Noisy Networks (Fortunato et al., 2018)
使用 Factorised Gaussian Noise 实现状态依赖的**自适应探索**，无需手动设置 ε 衰减:
$$y = (\mu^w + \sigma^w \odot \epsilon^w) x + (\mu^b + \sigma^b \odot \epsilon^b)$$

---

## 5. 算法扩展 (agent.py)

### 5.1 完整 Agent 类型矩阵

| Agent 类 | 网络 | 探索 | 回放 | TD Target |
|----------|------|------|------|-----------|
| `DQNAgent` | Standard | ε-greedy | Uniform | Vanilla DQN |
| `DoubleDQNAgent` | Standard | ε-greedy | Uniform | Double DQN |
| `DuelingDQNAgent` | Dueling | ε-greedy | Uniform | Double DQN |
| `DuelingDDQNAgent` | Dueling | ε-greedy | Uniform | Double DQN |
| `PERDQNAgent` | Dueling | ε-greedy | **PER** | Double DQN |
| `NoisyDQNAgent` | NoisyDueling | **NoisyNet** | Uniform | Double DQN |

### 5.2 Prioritized Experience Replay (Schaul et al., 2016)
- TD-error 绝对值作为优先级: $p_i = |\delta_i| + \epsilon$
- 采样概率: $P(i) = p_i^\alpha / \sum_k p_k^\alpha$
- 重要性采样: $w_i = (N \cdot P(i))^{-\beta}$
- α=0.6, β 从 0.4 线性增长到 1.0
- 使用 SumTree 数据结构实现 O(log N) 采样和更新

### 5.3 探索策略对比 (agent.py 内可选择)

| 策略 | 公式 | 参数 |
|------|------|------|
| ε-greedy | $a = \arg\max Q(s)$ w.p. $1-\epsilon$ | ε 指数衰减 |
| Boltzmann | $P(a) = e^{Q(s,a)/T} / \sum e^{Q(s,a')/T}$ | T 从 10 衰减到 0.1 |
| NoisyNet | 权重自带可学习噪声 | σ 可学习 |

---

## 6. 消融实验矩阵 (experiments.py)

### 实验 1: 算法横向对比 (6 组)
DQN / Double DQN / Dueling DQN / Dueling DDQN / PER DQN / Fixed-Time
- 5 种子 × 平均 ± 标准差
- 输出: 收敛对比图 + 柱状图 + LaTeX 表格

### 实验 2: 网络架构消融 (4 组)
Shallow / Standard / Deep / Dueling
- 分析参数量—性能关系、过拟合 vs 欠拟合

### 实验 3: 状态表示消融 (3 组)
S4(Queue) / S8(Queue+Wait) / S9(Full)
- 对比收敛速度和饥饿发生率

### 实验 4: 奖励函数消融 (4 组)
R1 / R2 / R3 / R4
- 评估各项奖励分量对性能的贡献

### 实验 5: 超参数敏感性 (3 子实验 × 4 组)
- 学习率 {1e-4, 5e-4, 1e-3, 5e-3}
- 折扣因子 γ {0.9, 0.95, 0.99, 0.999}
- 目标网络 τ {0.001, 0.005, 0.01, 0.1}
- 输出: 折线对比图

### 实验 6: 流量鲁棒性 (4 场景)
Uniform / PeakHour / Tidal / Burst
- RL vs Fixed-Time 在各流量下的性能差距
- 分析 RL 的自适应能力和 Fixed-Time 的局限性

---

## 7. 可视化仪表盘 (visualization.py — 新建)

### 7.1 训练监控面板
四合一面板: Loss + Reward + Queue + ε/噪声水平衰减

### 7.2 Q 值热力图 (决策边界可视化)
二维投影 (avg_queue_NS vs avg_queue_EW) 上的 Q(s,0)-Q(s,1) 差值热力图，展示智能体在何种状态下倾向于切换/保持

### 7.3 相位决策时间线
单 episode 逐步分析: 相位状态带、四方向排队堆叠面积图、Q 值差值曲线

### 7.4 雷达图 (多维度对比)
6 维度: 平均排队 ↓ / 最大等待 ↓ / 吞吐量 ↑ / 切换频率 ↓ / 收敛速度 ↑ / 稳定性 ↑

### 7.5 多种子带状图
5 种子训练的 mean ± std 阴影带 + 95% CI

### 7.6 实验报告汇总大图
所有消融实验的结果汇总到一张综合性信息图中

---

## 8. 编程接口 (CLI)

```bash
# 完整实验流程
python main.py

# 快速测试 (少量 episode)
python main.py --quick

# 仅训练最佳算法
python main.py --train-only --agent dueling_ddqn --episodes 800

# 仅运行消融实验
python main.py --ablation --experiment algorithm

# 生成可视化仪表盘
python main.py --dashboard

# 多路口协同
python main.py --multi --intersections 3

# 完整实验 + 仪表盘 + 多路口
python main.py --full
```

---

## 9. 实现路线图

| 步骤 | 文件 | 核心内容 |
|------|------|---------|
| Step 1 | agent.py | PER(SumTree), NoisyDQNAgent, Boltzmann探索 |
| Step 2 | network.py | NoisyLinear, ShallowDQN, DeepDQN |
| Step 3 | traffic_env.py | 5种流量, 3种状态空间, 4种奖励 |
| Step 4 | visualization.py | 新建: Q热力图, 相位线, 雷达图, 动画, 带状图 |
| Step 5 | experiments.py | 新建: 6类消融实验 + 统计检验 |
| Step 6 | main.py | 完整实验编排 + CLI |
| Step 7 | 全量验证 | 运行所有实验, 生成全部图表 |
| **Step 8** | **frontend/** | **🆕 前端模拟驾驶舱 (交互式交通动画 + 决策可视化)** |

---

## 10. 前端交互式模拟驾驶舱 (frontend/) 🆕

> **设计哲学**: 参考 Claude 官网 (anthropic.com) 与 Apple 官网 (apple.com) 的极简美学 — 克制的配色、大量留白、微妙的毛玻璃与阴影、清晰的字体层级、流畅的过渡动画。"Less is more, every pixel has a purpose."

### 10.1 系统架构

```
┌────────────────────────────────────────────────────────┐
│                    浏览器 (Frontend)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Traffic      │  │  Metrics     │  │  Decision     │  │
│  │ Animation    │  │  Dashboard   │  │  Inspector    │  │
│  │ (Canvas)     │  │  (面板)       │  │  (Q/policy)   │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         └─────────────────┼─────────────────┘           │
│                      WebSocket / HTTP                    │
└────────────────────────────────────────────────────────┘
                         │
┌────────────────────────────────────────────────────────┐
│                  Python 后端 (Flask)                      │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ RL Engine │  │  Sim Runner  │  │  API / WS       │  │
│  │ (agent.py)│  │ (traffic_env)│  │  (flask_sock)   │  │
│  └───────────┘  └──────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────┘
```

- **前端**: 纯 HTML5 + CSS3 + Vanilla JS (零依赖框架), Canvas 渲染交通动画
- **后端**: Flask + flask-sock (WebSocket), 复用现有 `traffic_env.py` + `agent.py`
- **通信**: WebSocket 实时推送每步仿真状态; HTTP REST 获取模型列表、配置等

### 10.2 页面布局 (1200px 基准宽度, 响应式)

```
┌──────────────────────────────────────────────────────────┐
│  🚦 TrafficRL · Adaptive Signal Control                  │  ← Navbar (64px高, 磨砂玻璃)
├────────────────────┬─────────────────────────────────────┤
│                    │                                     │
│   交通场景          │   指标面板                           │
│   Canvas 动画       │   ┌─────────────┐                  │
│   (720×720)        │   │ Episode   42 │  ← 大数字字体     │
│                    │   │ Reward  -3.2K│                  │
│   ┌──────────┐     │   ├─────────────┤                  │
│   │  ↑  N    │     │   │ 队列长度     │  ← 四个方向柱状图   │
│   │  ██      │     │   │ N: 12  S: 8 │                  │
│   │←W   E→  │     │   │ E: 5   W: 3 │                  │
│   │     ██   │     │   ├─────────────┤                  │
│   │  ↓  S    │     │   │ 等待时间     │  ← 折线趋势       │
│   └──────────┘     │   │ 排队曲线     │  ← mini sparkline │
│                    │   └─────────────┘                  │
│                    │                                     │
├────────────────────┴─────────────────────────────────────┤
│  底部面板: Q值分布  | 策略熵 | 决策历史时间线               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Action 0 (Hold)  ████████████░░░░░  0.72       │   │
│  │  Action 1 (Switch)██████░░░░░░░░░░░  0.28       │   │
│  │  ── Phase Timeline ──────────────────────────   │   │
│  │  NS ████████████░░░████████████░░░███░░░█████   │   │
│  │  EW ░░░░░░░░░░████░░░░░░░░░░░░████░░░███░░░░░   │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 10.3 配色系统 (Apple / Claude 风格)

| 令牌 | Hex | 用途 |
|------|-----|------|
| `--bg-primary` | `#f5f5f7` | 页面主背景 (Apple 经典浅灰) |
| `--bg-card` | `#ffffff` | 卡片/面板背景 |
| `--bg-glass` | `rgba(255,255,255,0.72)` | 毛玻璃导航栏 |
| `--text-primary` | `#1d1d1f` | 主文字 (Apple 深灰, 非纯黑) |
| `--text-secondary` | `#86868b` | 次级文字 |
| `--accent` | `#0071e3` | 主题色 (Apple 蓝) |
| `--accent-green` | `#34c759` | 成功/绿灯 |
| `--accent-red` | `#ff3b30` | 警告/红灯 |
| `--accent-orange` | `#ff9500` | 黄灯/中等 |
| `--border` | `rgba(0,0,0,0.08)` | 分隔线 |
| `--shadow-sm` | `0 1px 3px rgba(0,0,0,0.06)` | 卡片投影 |
| `--shadow-md` | `0 4px 16px rgba(0,0,0,0.08)` | 悬浮投影 |
| `--radius` | `16px` | 统一圆角 |

### 10.4 交通场景 Canvas 动画模块

#### 10.4.1 路口渲染规范
- **画布尺寸**: 720×720px (可缩放)
- **道路**: 双向四车道 (N/S 各 2 条, E/W 各 2 条), 浅灰 `#e8e8ed`, 车道线虚线
- **车辆**: 圆角矩形 (6×10px 比例), 四色区分方向: N→蓝, S→绿, E→橙, W→紫
  - 平滑移动: `requestAnimationFrame` 驱动, 60fps
  - 车辆间距: 最小 8px, 停车线前 4px 停
- **信号灯**: 路口四角各一, 直径 12px, Active=实心彩色, Inactive=浅灰 `#d1d1d6`
  - 绿灯 `#34c759`, 红灯 `#ff3b30`, 黄灯 `#ff9500` (切换过渡 3 帧)
- **排队区域**: 停车线后方半透明色块, 深度 = 排队长度
- **动画速度**: 可调 0.5× / 1× / 2× / 5×

#### 10.4.2 车辆动画生命周期
1. **生成**: 从道路边界外平滑进入 (0.3s ease-out)
2. **行驶**: 匀速向路口移动, 速度 2px/帧 (可调)
3. **停车**: 排队时停在队尾位置, 后车自动调整间距
4. **通过**: 绿灯时加速通过路口 (1.5× 速度), 0.3s 后消失在对面
5. **移除**: 离开画布后从车辆数组移除, 计数 `throughput++`

#### 10.4.3 HUD 覆盖层
画布右上角半透明覆盖:
```
┌────────────────┐
│  ⏱ Step 247   │
│  🚗 Active 34  │
│  ✅ Departed 582│
│  🚦 Phase: NS  │
└────────────────┘
```

### 10.5 实时指标面板

#### 10.5.1 数值指标 (大字体, SF Pro / Inter 字体)
- **Episode Reward**: 主指标, 48px 粗体, 带趋势箭头 (↑/↓) 和变化百分比
- **Current Queue**: 四方向总和, 32px, 带迷你柱状图
- **Avg Wait Time**: 32px, 带迷你 sparkline (最近 50 步)
- **Throughput**: 累计通过车辆数, 32px

#### 10.5.2 动态柱状图 (四方向排队)
垂直柱状, 高度 = 排队长度 / max_queue, 颜色: N=蓝 S=绿 E=橙 W=紫
实时动画过渡 (CSS `transition: height 0.3s ease`)

#### 10.5.3 Mini Sparklines
使用 Canvas 绘制 50 步滑动窗口的排队/等待趋势线

### 10.6 决策过程可视化 (底部面板)

#### 10.6.1 Q 值对比条 (DQN 系列)
```
Action 0 (Hold)   ████████████████░░░░░░  0.72  max ←
Action 1 (Switch) ██████░░░░░░░░░░░░░░░░  0.28
```
- 实时更新, 最大值高亮 `--accent` 色, 次值灰色
- 宽度比例 = Q 值大小, 0.3s 过渡动画

#### 10.6.2 策略概率分布 (PG 系列: A2C/PPO)
```
π(a₀=Hold)   ████████████████░░░░░░  81%
π(a₁=Switch) ████░░░░░░░░░░░░░░░░░░  19%
```
- 从 Categorical 分布实时读取

#### 10.6.3 相位决策时间线
水平滚动条, 显示最近 200 步的相位状态:
- NS-Green: 蓝色块 `#0071e3`
- EW-Green: 橙色块 `#ff9500`
- Yellow: 黄色块 `#ffd60a`
- 点击某步可回看该步状态

### 10.7 控制栏 (播放器风格)

```
┌─────────────────────────────────────────────────────────┐
│  [⏮] [▶] [⏸] [⏭]  速度: [0.5x] [1x] [2x] [5x]        │
│  模式: [○ 单步] [● 连续] [○ Episode]                   │
│  Agent: [Dueling DQN ▾]  流量: [Uniform ▾]             │
│  Episode: [42/300]  ████████░░░░  73%                   │
└─────────────────────────────────────────────────────────┘
```
- 玻璃拟态圆角条, 固定在页面底部
- 按钮: 纯图标, 悬停放大 1.1×, 过渡 0.2s ease

### 10.8 后端 API 设计

| 方法 | 路由 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查, 返回可用模型列表 |
| `POST` | `/api/sim/init` | 初始化仿真: `{agent, pattern, state_dim, reward_type}` |
| `WS` | `/ws/sim` | WebSocket 双向通信, 逐步仿真 |
| `POST` | `/api/sim/reset` | 重置环境到 episode 开头 |
| `POST` | `/api/sim/step` | 单步仿真 (REST fallback) |
| `POST` | `/api/sim/load_model` | 加载预训练 Agent 权重 |
| `GET` | `/api/models` | 列出 results/ 下的模型文件 |

#### WebSocket 协议 (JSON)

**Client → Server (控制指令)**:
```json
{"type": "init", "agent": "dueling", "pattern": "peak_hour", "state_dim": 8}
{"type": "step"}
{"type": "play", "speed": 2}
{"type": "pause"}
{"type": "reset"}
{"type": "load_model", "path": "results/model_ep300.pth"}
```

**Server → Client (每步状态推送)**:
```json
{
  "type": "state",
  "episode": 42, "step": 247,
  "phase": 0, "action": 1,
  "queues": [12, 8, 5, 3],
  "vehicles": [
    {"id": 1, "dir": "N", "x": 340, "y": 520, "speed": 2.0, "waiting": true},
    {"id": 2, "dir": "S", "x": 380, "y": 180, "speed": 0.0, "waiting": false}
  ],
  "q_values": [0.72, 0.28],
  "policy": [0.81, 0.19],
  "reward": -15.2, "cumulative_reward": -3245.0,
  "throughput": 582,
  "avg_wait": [5.2, 3.1, 2.8, 1.9],
  "info": {"total_departed": 582, "switch_penalty": 0}
}
```

### 10.9 交互动效规范

| 元素 | 动效 | 时长 | 缓动函数 |
|------|------|------|---------|
| 卡片悬浮 | scale(1.01) + shadow ↑ | 0.25s | ease-out |
| 数值更新 | 数字滚动效果 | 0.4s | ease-out |
| Canvas 车辆 | translate 平滑移动 | 每帧 | linear |
| 柱状图高度 | height transition | 0.3s | ease |
| Q 值条宽度 | width transition | 0.3s | ease |
| 信号灯切换 | opacity + scale | 0.2s | ease-in-out |
| 页面加载 | 卡片从下淡入 (staggered) | 0.5s | ease-out |
| 模态框 | backdrop-filter blur + scale(0.95→1) | 0.3s | ease-out |

### 10.10 字体规范
- 主字体: `'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif`
- 等宽数字: `'SF Mono', 'JetBrains Mono', 'Fira Code', monospace` (用于数值显示, tabular-nums)
- 字号层级: 48px (主指标) / 32px (次指标) / 16px (正文) / 13px (标签) / 11px (注释)

### 10.11 响应式断点
| 断点 | 布局 | 画布缩放 |
|------|------|---------|
| ≥1200px | 双栏 (画布 + 面板并排) | 1.0× |
| 768–1199px | 上下堆叠 | 0.75× |
| <768px | 单栏滚动, 面板折叠 | 0.5× |

### 10.12 文件结构

```
RL/
├── backend/
│   ├── app.py              # Flask + WebSocket 服务器
│   ├── sim_runner.py       # 仿真运行器 (控制步进/连续/速度)
│   ├── templates/
│   │   └── index.html      # 主页面 (~600行)
│   └── static/
│       ├── css/
│       │   └── style.css   # 全局样式 (~500行)
│       └── js/
│           ├── app.js      # 主逻辑: WS连接, 状态管理 (~300行)
│           ├── canvas.js   # Canvas动画渲染引擎 (~400行)
│           ├── panels.js   # 指标面板更新 (~200行)
│           └── controls.js # 控制栏交互 (~150行)
```

### 10.13 技术选型

| 层 | 技术 | 理由 |
|----|------|------|
| 后端框架 | **Flask** | 轻量, 与 Python RL 代码天然集成 |
| WebSocket | **flask-sock** | 极简 WebSocket 支持, 零配置 |
| 前端框架 | **无框架, Vanilla JS** | 零依赖, 加载快, 完全可控 |
| Canvas 渲染 | **原生 Canvas 2D API** | 高性能 60fps, 不依赖第三方库 |
| CSS 方案 | **CSS Variables + Flexbox/Grid** | 现代布局, 动态主题切换 |
| 字体加载 | **Google Fonts (Inter)** | 仅一个外部依赖 |

### 10.14 实现路线

| 子步骤 | 文件 | 核心内容 |
|--------|------|---------|
| 8.1 | `backend/app.py` | Flask 应用骨架, 静态文件路由, WebSocket 端点 |
| 8.2 | `backend/sim_runner.py` | 仿真控制器: init/step/reset/play/pause/speed |
| 8.3 | `templates/index.html` | HTML 结构: 导航栏 + Canvas + 面板 + 控制栏 |
| 8.4 | `static/css/style.css` | Apple/Claude 风格 CSS: 玻璃拟态, 圆角, 阴影, 字体 |
| 8.5 | `static/js/canvas.js` | Canvas 动画引擎: 路口/道路/车辆/信号灯绘制与更新 |
| 8.6 | `static/js/panels.js` | 指标面板: 数值动画, 柱状图, sparkline, Q 值条 |
| 8.7 | `static/js/controls.js` | 播放控制: 播放/暂停/速度/步进/重置 |
| 8.8 | `static/js/app.js` | 集成: WebSocket 连接管理 + 状态分发 + 事件总线 |
| 8.9 | 联调测试 | 加载预训练模型, 验证动画流畅度与数据准确性 |

