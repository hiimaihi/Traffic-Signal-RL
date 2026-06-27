/**
 * TrafficRL Cockpit — 多路口 WebSocket 主控制器
 * ==============================================
 * 连接 ws_multi 端点, 路由消息到 Canvas/Panels/Controls。
 */
(() => {
    let ws = null;
    let reconnectTimer = null;

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = proto + '//' + location.host + '/ws_multi';
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('[Multi] WebSocket connected');
            MultiControls.setConnected(true);
            MultiControls.init(ws);
            if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        };

        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);

                if (data.type === 'state') {
                    MultiCanvas.update(data);
                    MultiPanels.update(data);
                    if (data.done) {
                        MultiControls.setStatus('✅ Episode 完成');
                        MultiControls.setButtons(false);
                    }
                } else if (data.type === 'control') {
                    if (data.status === 'playing') {
                        MultiControls.setButtons(true);
                    } else if (data.status === 'paused' || data.status === 'stopped') {
                        MultiControls.setButtons(false);
                    }
                    if (data.models_loaded !== undefined) {
                        MultiControls.setInitialized(data.models_loaded.every(v => v));
                    }
                    if (data.status === 'playing') {
                        MultiControls.setStatus('▶ 播放中');
                    }
                }
            } catch (err) {
                console.error('[Multi] Parse error:', err);
            }
        };

        ws.onclose = () => {
            console.log('[Multi] WebSocket closed');
            MultiControls.setConnected(false);
            reconnectTimer = setTimeout(connect, 2000);
        };

        ws.onerror = (err) => {
            console.error('[Multi] WebSocket error:', err);
        };
    }

    // 初始化
    MultiCanvas.init();
    MultiPanels.init();
    connect();
})();
