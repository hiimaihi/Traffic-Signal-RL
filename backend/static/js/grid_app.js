/**
 * TrafficRL Cockpit — 2×2 网格 WebSocket 主控制器
 */
(() => {
    let ws = null;
    let reconnectTimer = null;

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = proto + '//' + location.host + '/ws_grid';
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('[Grid] WebSocket connected');
            GridControls.setConnected(true);
            GridControls.init(ws);
            if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        };

        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'state') {
                    GridCanvas.update(data);
                    GridPanels.update(data);
                    if (data.done) {
                        GridControls.setStatus('✅ Episode 完成');
                        GridControls.setButtons(false);
                    }
                } else if (data.type === 'control') {
                    if (data.status === 'playing') {
                        GridControls.setButtons(true);
                    } else if (data.status === 'paused' || data.status === 'stopped') {
                        GridControls.setButtons(false);
                    }
                    if (data.models_loaded !== undefined) {
                        GridControls.setInitialized(data.models_loaded.every(v => v));
                    }
                    if (data.status === 'playing') {
                        GridControls.setStatus('▶ 播放中');
                    }
                }
            } catch (err) {
                console.error('[Grid] Parse error:', err);
            }
        };

        ws.onclose = () => {
            console.log('[Grid] WebSocket closed');
            GridControls.setConnected(false);
            reconnectTimer = setTimeout(connect, 2000);
        };

        ws.onerror = (err) => console.error('[Grid] WebSocket error:', err);
    }

    GridCanvas.init();
    GridPanels.init();
    connect();
})();
