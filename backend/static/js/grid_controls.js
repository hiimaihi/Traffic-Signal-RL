/**
 * TrafficRL Cockpit — 2×2 网格控制栏
 */
const GridControls = (() => {
    let ws = null;
    let playing = false;

    const els = {};
    const SPEEDS = [0.5, 1, 2, 4, 8];

    function init(webSocket) {
        ws = webSocket;

        els.btnInit = document.getElementById('btnInit');
        els.btnStep = document.getElementById('btnStep');
        els.btnPlay = document.getElementById('btnPlay');
        els.btnPause = document.getElementById('btnPause');
        els.btnReset = document.getElementById('btnReset');
        els.selectAgent = document.getElementById('selectAgent');
        els.selectPattern = document.getElementById('selectPattern');
        els.speedSlider = document.getElementById('speedSlider');
        els.speedLabel = document.getElementById('speedLabel');
        els.navStatus = document.getElementById('navStatus');
        els.navAgent = document.getElementById('navAgent');

        els.btnInit.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    cmd: 'init',
                    agent: els.selectAgent.value,
                    pattern: els.selectPattern.value,
                }));
                els.navStatus.textContent = '⏳ 初始化...';
                els.navAgent.textContent = els.selectAgent.options[els.selectAgent.selectedIndex].text;
            }
        });

        els.btnStep.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'step' }));
            }
        });

        els.btnPlay.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const speed = SPEEDS[parseInt(els.speedSlider.value) - 1] || 1;
                ws.send(JSON.stringify({ cmd: 'play', speed }));
                playing = true;
            }
        });

        els.btnPause.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'pause' }));
                playing = false;
            }
        });

        els.btnReset.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'reset' }));
            }
        });

        els.speedSlider.addEventListener('input', () => {
            const speed = SPEEDS[parseInt(els.speedSlider.value) - 1] || 1;
            els.speedLabel.textContent = speed + '×';
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'speed', speed }));
            }
        });

        // speed ticks
        document.querySelectorAll('.speed-ticks span').forEach(span => {
            span.addEventListener('click', () => {
                const s = parseFloat(span.dataset.speed);
                const idx = SPEEDS.indexOf(s);
                if (idx >= 0) {
                    els.speedSlider.value = idx + 1;
                    els.speedLabel.textContent = s + '×';
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ cmd: 'speed', speed: s }));
                    }
                }
            });
        });
    }

    function setButtons(isPlaying) {
        playing = isPlaying;
        if (els.btnPlay) els.btnPlay.disabled = isPlaying;
        if (els.btnPause) els.btnPause.disabled = !isPlaying;
        if (els.btnStep) els.btnStep.disabled = isPlaying;
    }

    function setStatus(text) {
        if (els.navStatus) els.navStatus.textContent = text;
    }

    function setConnected(connected) {
        if (els.navStatus) {
            els.navStatus.textContent = connected ? '🟢 已连接' : '🔴 断开';
        }
    }

    function setInitialized(ok) {
        if (els.navStatus) {
            els.navStatus.textContent = ok ? '🧠 模型已加载' : '⚠ 未找到模型';
        }
        if (els.btnInit) els.btnInit.disabled = false;
    }

    return { init, setButtons, setStatus, setConnected, setInitialized };
})();
