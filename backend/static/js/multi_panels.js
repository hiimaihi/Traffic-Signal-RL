/**
 * TrafficRL Cockpit — 多路口面板更新
 * ====================================
 * 更新两路口的指标、排队、置信度。
 */
const MultiPanels = (() => {
    const els = {};

    function init() {
        els.reward = document.getElementById('metricReward');
        els.tp0 = document.getElementById('metricTp0');
        els.tp1 = document.getElementById('metricTp1');
        els.pipeline = document.getElementById('metricPipeline');
        els.action0 = document.getElementById('metricAction0');
        els.action1 = document.getElementById('metricAction1');
        els.phaseI0 = document.getElementById('phaseI0');
        els.phaseI1 = document.getElementById('phaseI1');
        els.navEpisode = document.getElementById('navEpisode');
        els.navStep = document.getElementById('navStep');

        // Q bars (prefixed with 'm' to avoid conflict with queue elements)
        els.mqBar0H = document.getElementById('mqBar0H');
        els.mqBar0Sw = document.getElementById('mqBar0Sw');
        els.mqBar1H = document.getElementById('mqBar1H');
        els.mqBar1Sw = document.getElementById('mqBar1Sw');
        els.mqNum0H = document.getElementById('mqNum0H');
        els.mqNum0Sw = document.getElementById('mqNum0Sw');
        els.mqNum1H = document.getElementById('mqNum1H');
        els.mqNum1Sw = document.getElementById('mqNum1Sw');
    }

    function update(data) {
        // 导航
        if (els.navEpisode) els.navEpisode.textContent = 'Episode ' + (data.episode||0);
        if (els.navStep) els.navStep.textContent = 'Step ' + (data.step||0);

        // 指标
        if (els.reward) els.reward.textContent = (data.cumulative_reward||0).toFixed(1);
        if (els.tp0) els.tp0.textContent = (data.throughputs||[0,0])[0];
        if (els.tp1) els.tp1.textContent = (data.throughputs||[0,0])[1];
        if (els.pipeline) els.pipeline.textContent = data.pipeline_count||0;
        if (els.action0) els.action0.textContent = (data.actions||[0,0])[0] === 1 ? 'SWITCH' : 'HOLD';
        if (els.action1) els.action1.textContent = (data.actions||[0,0])[1] === 1 ? 'SWITCH' : 'HOLD';
        if (els.phaseI0) els.phaseI0.textContent = (data.phase_labels||['NS','NS'])[0];
        if (els.phaseI1) els.phaseI1.textContent = (data.phase_labels||['NS','NS'])[1];

        // 排队
        const queues = data.queues || [[0,0,0,0],[0,0,0,0]];
        for (const i of [0,1]) {
            for (const [j, d] of ['N','S','E','W'].entries()) {
                const bar = document.getElementById('qBar'+i+d);
                const num = document.getElementById('qNum'+i+d);
                if (num) num.textContent = queues[i][j];
                if (bar) {
                    const pct = Math.min(100, queues[i][j] * 5);
                    bar.style.width = pct + '%';
                    bar.style.background = queues[i][j] > 15 ? '#ff3b30' : queues[i][j] > 8 ? '#ff9500' : '#34c759';
                }
            }
        }

        // Q值/置信度
        const pol = data.policy || [[0.5,0.5],[0.5,0.5]];
        const pairs = [
            ['mqBar0H','mqNum0H',0,0], ['mqBar0Sw','mqNum0Sw',0,1],
            ['mqBar1H','mqNum1H',1,0], ['mqBar1Sw','mqNum1Sw',1,1],
        ];
        for (const [barKey, numKey, i, j] of pairs) {
            const bar = els[barKey];
            const num = els[numKey];
            if (num) num.textContent = (pol[i][j]*100).toFixed(1) + '%';
            if (bar) bar.style.width = (pol[i][j]*100) + '%';
        }
    }

    return { init, update };
})();
