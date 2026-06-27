/**
 * TrafficRL Cockpit — 控制栏交互
 * ===============================
 * 处理按钮点击、下拉菜单、速度滑块事件。
 * 不直接操作仿真, 通过 App 事件总线通信。
 */

const Controls = (() => {
    // ── 元素 ──
    let btnInit, btnStep, btnPlay, btnPause, btnReset, btnCompare;
    let selectAgent, selectPattern;
    let speedRange, speedLabel, speedPresets;
    let onCommand; // 回调: (cmd, data) => void

    // ── 状态 ──
    let isPlaying = false;

    // ── 初始化 ──
    function init(cmdCallback) {
        onCommand = cmdCallback;

        btnInit = document.getElementById('btnInit');
        btnStep = document.getElementById('btnStep');
        btnPlay = document.getElementById('btnPlay');
        btnPause = document.getElementById('btnPause');
        btnReset = document.getElementById('btnReset');
        btnCompare = document.getElementById('btnCompare');
        selectAgent = document.getElementById('selectAgent');
        selectPattern = document.getElementById('selectPattern');
        speedRange = document.getElementById('speedRange');
        speedLabel = document.getElementById('speedLabel');
        speedPresets = document.querySelector('.speed-presets');

        // 按钮事件
        btnInit.addEventListener('click', () => {
            onCommand('init', {
                agent: selectAgent.value,
                pattern: selectPattern.value,
            });
        });

        btnStep.addEventListener('click', () => {
            onCommand('step');
        });

        btnPlay.addEventListener('click', () => {
            if (isPlaying) {
                onCommand('pause');
            } else {
                onCommand('play', { speed: parseFloat(speedRange.value) });
            }
        });

        btnPause.addEventListener('click', () => {
            onCommand('pause');
        });

        btnReset.addEventListener('click', () => {
            onCommand('reset');
        });

        // 算法对比按钮
        if (btnCompare) {
            btnCompare.addEventListener('click', () => {
                btnCompare.disabled = true;
                btnCompare.textContent = '⏳ 评估中...';
                onCommand('compare', {
                    pattern: selectPattern.value,
                    agents: ['dueling', 'dqn', 'double', 'noisy', 'boltzmann', 'per_dqn', 'a2c', 'ppo'],
                });
            });
        }

        // 速度滑块
        speedRange.addEventListener('input', () => {
            const speed = parseFloat(speedRange.value);
            speedLabel.textContent = speed + '×';
            if (isPlaying) {
                onCommand('speed', { speed });
            }
            updateSpeedPresets(speed);
        });

        // 速度预设
        if (speedPresets) {
            speedPresets.addEventListener('click', (e) => {
                const span = e.target.closest('span[data-speed]');
                if (!span) return;
                const speed = parseFloat(span.dataset.speed);
                speedRange.value = speed;
                speedLabel.textContent = speed + '×';
                if (isPlaying) {
                    onCommand('speed', { speed });
                }
                updateSpeedPresets(speed);
            });
        }
    }

    function updateSpeedPresets(speed) {
        document.querySelectorAll('.speed-presets span').forEach(s => {
            s.classList.toggle('active', parseFloat(s.dataset.speed) === speed);
        });
    }

    // ── 播放状态切换 ──
    function setPlaying(playing) {
        isPlaying = playing;
        btnPlay.textContent = playing ? '⏸ 暂停' : '▶ 播放';
        btnPlay.classList.toggle('pause-mode', playing);
        btnPause.disabled = !playing;
        btnStep.disabled = playing;
        btnInit.disabled = playing;
    }

    function setPaused() {
        isPlaying = false;
        btnPlay.textContent = '▶ 播放';
        btnPlay.classList.remove('pause-mode');
        btnPause.disabled = true;
        btnStep.disabled = false;
        btnInit.disabled = false;
    }

    function setStopped() {
        isPlaying = false;
        btnPlay.textContent = '▶ 播放';
        btnPlay.classList.remove('pause-mode');
        btnPause.disabled = true;
        btnStep.disabled = false;
        btnInit.disabled = false;
    }

    // ── 公开 API ──
    function setCompareReady() {
        if (btnCompare) {
            btnCompare.disabled = false;
            btnCompare.textContent = '▶ 运行对比';
        }
    }

    return { init, setPlaying, setPaused, setStopped, setCompareReady };
})();
