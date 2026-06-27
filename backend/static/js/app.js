/**
 * TrafficRL Cockpit — 主应用
 * ==========================
 * WebSocket 通信 + 模块整合 + 状态管理。
 */

const App = (() => {
    // ── WebSocket ──
    let ws = null;
    let reconnectTimer = null;
    const RECONNECT_DELAY = 2000;

    // ── 状态 ──
    let currentAgent = 'dueling';
    let isConnected = false;

    // ── 连接 WebSocket ──
    function connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            ws = new WebSocket(wsUrl);
        } catch (e) {
            console.error('WebSocket create error:', e);
            scheduleReconnect();
            return;
        }

        ws.onopen = () => {
            console.log(' WebSocket connected');
            isConnected = true;
            Panels.setStatus('🟢 已连接', 'playing');
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        };

        ws.onmessage = (event) => {
            const lines = event.data.split('\n').filter(line => line.trim());
            for (const line of lines) {
                try {
                    const data = JSON.parse(line);
                    handleMessage(data);
                } catch (e) {
                    console.error('WS parse error:', e, 'line:', line.substring(0, 80));
                }
            }
        };

        ws.onclose = () => {
            console.log(' WebSocket closed');
            isConnected = false;
            Panels.setStatus('🔴 断开连接', 'paused');
            scheduleReconnect();
        };

        ws.onerror = (err) => {
            console.error(' WebSocket error:', err);
        };
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        Panels.setStatus('🟡 正在重连...', 'paused');
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connect();
        }, RECONNECT_DELAY);
    }

    function send(cmd, data = {}) {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not connected, cannot send:', cmd);
            // 尝试用 REST API 降级
            sendViaRest(cmd, data);
            return;
        }
        ws.send(JSON.stringify({ cmd, ...data }));
    }

    async function sendViaRest(cmd, data) {
        try {
            if (cmd === 'init') {
                const resp = await fetch('/api/init', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data),
                });
                if (resp.ok) {
                    const state = await resp.json();
                    handleMessage(state);
                }
            } else if (cmd === 'reset') {
                const resp = await fetch('/api/reset', { method: 'POST' });
                if (resp.ok) {
                    const state = await resp.json();
                    handleMessage(state);
                }
            }
        } catch (e) {
            console.error('REST fallback error:', e);
        }
    }

    // ── 消息处理 ──
    function handleMessage(data) {
        if (!data) return;

        if (data.type === 'state') {
            // 仿真状态快照
            Canvas.update(data);
            Panels.update(data);

            if (data.done) {
                Controls.setStopped();
                Panels.setStatus('✅ Episode 完成', 'paused');
            }
        } else if (data.type === 'control') {
            // 控制响应
            if (data.status === 'playing') {
                Controls.setPlaying(true);
                Panels.setStatus('▶ 运行中', 'playing');
            } else if (data.status === 'paused') {
                Controls.setPaused();
                Panels.setStatus('⏸ 已暂停', 'paused');
            } else if (data.status === 'stopped') {
                Controls.setStopped();
                Panels.setStatus('⏹ 已停止', 'paused');
            }
            if (data.model_loaded) {
                Panels.setStatus('🧠 模型已加载', 'playing');
                console.log(' Model loaded:', data.path);
            }
        }
    }

    // ── 命令分发 ──
    function onCommand(cmd, data = {}) {
        switch (cmd) {
            case 'init':
                currentAgent = data.agent || currentAgent;
                Panels.setAgent(getAgentDisplayName(currentAgent));
                Panels.setStatus('⏳ 初始化...', 'paused');
                send('init', data);
                break;
            case 'step':
                send('step');
                break;
            case 'play':
                send('play', { speed: data.speed || 1.0 });
                break;
            case 'pause':
                send('pause');
                Controls.setPaused();
                Panels.setStatus('⏸ 已暂停', 'paused');
                break;
            case 'reset':
                Panels.setStatus('⏳ 重置...', 'paused');
                send('reset');
                break;
            case 'speed':
                send('speed', { speed: data.speed || 1.0 });
                break;
            case 'compare':
                Panels.setCompareLoading();
                runCompare(data);
                break;
        }
    }

    async function runCompare(data) {
        try {
            const resp = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (resp.ok) {
                const result = await resp.json();
                Panels.renderCompare(result);
            } else {
                Panels.renderCompare(null);
            }
        } catch (e) {
            console.error('Compare error:', e);
            Panels.renderCompare(null);
        }
        Controls.setCompareReady();
    }

    function getAgentDisplayName(key) {
        const names = {
            dqn: 'DQN',
            double: 'Double DQN',
            dueling: 'Dueling DQN',
            noisy: 'Noisy DQN',
            noisy_double: 'Noisy Double',
            boltzmann: 'Boltzmann DQN',
            boltzmann_double: 'Boltzmann Double',
            per: 'PER DQN',
            dueling_per: 'Dueling PER',
            a2c: 'A2C',
            ppo: 'PPO',
        };
        return names[key] || key;
    }

    // ── 初始化 ──
    function init() {
        Canvas.init();
        Panels.init();
        Controls.init(onCommand);

        // 自动连接
        connect();

        // 页面卸载清理
        window.addEventListener('beforeunload', () => {
            if (ws) {
                send('stop');
            }
        });

        console.log(' TrafficRL Cockpit initialized');
    }

    // ── 启动 ──
    document.addEventListener('DOMContentLoaded', init);

    return { send, onCommand };
})();
