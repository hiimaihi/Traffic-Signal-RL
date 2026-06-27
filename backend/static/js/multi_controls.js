/**
 * TrafficRL Cockpit — 多路口控制栏
 * ==================================
 * 初始化/单步/播放/暂停/重置/速度控制。
 */
const MultiControls = (() => {
    let ws = null;
    const els = {};

    function init(websocket) {
        ws = websocket;
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
            els.navStatus.textContent = '⏳ 初始化...';
            els.btnInit.disabled = true;
            ws.send(JSON.stringify({
                cmd: 'init',
                agent: els.selectAgent.value,
                pattern: els.selectPattern.value,
            }));
        });

        els.btnStep.addEventListener('click', () => {
            ws.send(JSON.stringify({ cmd: 'step' }));
        });

        els.btnPlay.addEventListener('click', () => {
            ws.send(JSON.stringify({ cmd: 'play', speed: parseFloat(els.speedSlider.value) }));
        });

        els.btnPause.addEventListener('click', () => {
            ws.send(JSON.stringify({ cmd: 'pause' }));
        });

        els.btnReset.addEventListener('click', () => {
            ws.send(JSON.stringify({ cmd: 'reset' }));
        });

        els.speedSlider.addEventListener('input', () => {
            els.speedLabel.textContent = els.speedSlider.value + '×';
            ws.send(JSON.stringify({ cmd: 'speed', speed: parseFloat(els.speedSlider.value) }));
        });

        els.selectAgent.addEventListener('change', () => {
            els.navAgent.textContent = els.selectAgent.options[els.selectAgent.selectedIndex].text;
        });

        // 预设速度点击
        document.querySelectorAll('.speed-presets span').forEach(sp => {
            sp.addEventListener('click', () => {
                els.speedSlider.value = sp.dataset.speed;
                els.speedLabel.textContent = sp.dataset.speed + '×';
                ws.send(JSON.stringify({ cmd: 'speed', speed: parseFloat(sp.dataset.speed) }));
            });
        });
    }

    function _nav() { return els.navStatus || document.getElementById('navStatus'); }

    function setConnected(connected) {
        const el = _nav();
        if (el) el.textContent = connected ? '🟢 已连接' : '🔴 断开';
    }

    function setStatus(text) {
        const el = _nav();
        if (el) el.textContent = text;
    }

    function setButtons(playing) {
        if (els.btnPlay) els.btnPlay.disabled = playing;
        if (els.btnPause) els.btnPause.disabled = !playing;
        if (els.btnInit) els.btnInit.disabled = false;
    }

    function setInitialized(modelLoaded) {
        const el = _nav();
        if (els.btnInit) els.btnInit.disabled = false;
        if (el) el.textContent = modelLoaded ? '🧠 模型已加载' : '⚠ 无模型';
    }

    return { init, setConnected, setStatus, setButtons, setInitialized };
})();
