/**
 * TrafficRL Cockpit — 2×2 网格面板更新
 */
const GridPanels = (() => {
    const els = {};

    function init() {
        els.reward = document.getElementById('metricReward');
        els.tp0 = document.getElementById('metricTp0');
        els.tp1 = document.getElementById('metricTp1');
        els.tp2 = document.getElementById('metricTp2');
        els.tp3 = document.getElementById('metricTp3');
        els.pipeline = document.getElementById('metricPipeline');
        els.action0 = document.getElementById('metricAction0');
        els.action1 = document.getElementById('metricAction1');
        els.phaseI0 = document.getElementById('phaseI0');
        els.phaseI1 = document.getElementById('phaseI1');
        els.phaseI2 = document.getElementById('phaseI2');
        els.phaseI3 = document.getElementById('phaseI3');
        els.navEpisode = document.getElementById('navEpisode');
        els.navStep = document.getElementById('navStep');

        // Q bars (g = grid prefix)
        for (const i of [0,1,2,3]) {
            for (const a of ['H','Sw']) {
                els['gBar'+i+a] = document.getElementById('gBar'+i+a);
                els['gNum'+i+a] = document.getElementById('gNum'+i+a);
            }
        }
    }

    function update(data) {
        if (els.navEpisode) els.navEpisode.textContent = 'Episode ' + (data.episode||0);
        if (els.navStep) els.navStep.textContent = 'Step ' + (data.step||0);
        if (els.reward) els.reward.textContent = (data.cumulative_reward||0).toFixed(1);
        if (els.pipeline) els.pipeline.textContent = data.pipeline_count||0;

        const tps = data.throughputs || [0,0,0,0];
        if (els.tp0) els.tp0.textContent = tps[0];
        if (els.tp1) els.tp1.textContent = tps[1];
        if (els.tp2) els.tp2.textContent = tps[2];
        if (els.tp3) els.tp3.textContent = tps[3];

        const actions = data.actions || [0,0,0,0];
        if (els.action0) els.action0.textContent = actions[0] === 1 ? 'SWITCH' : 'HOLD';
        if (els.action1) els.action1.textContent = actions[1] === 1 ? 'SWITCH' : 'HOLD';

        const labels = data.phase_labels || ['NS','NS','NS','NS'];
        if (els.phaseI0) els.phaseI0.textContent = labels[0];
        if (els.phaseI1) els.phaseI1.textContent = labels[1];
        if (els.phaseI2) els.phaseI2.textContent = labels[2];
        if (els.phaseI3) els.phaseI3.textContent = labels[3];

        // 排队 — 使用 gq 前缀
        const queues = data.queues || [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        for (const i of [0,1,2,3]) {
            for (const [j, d] of ['N','S','E','W'].entries()) {
                const bar = document.getElementById('gqBar'+i+d);
                const num = document.getElementById('gqNum'+i+d);
                if (num) num.textContent = queues[i][j];
                if (bar) {
                    const pct = Math.min(100, queues[i][j] * 5);
                    bar.style.width = pct + '%';
                    bar.style.background = queues[i][j] > 15 ? '#ff3b30' : queues[i][j] > 8 ? '#ff9500' : '#34c759';
                }
            }
        }

        // Q值/置信度
        const pol = data.policy || [[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]];
        for (const i of [0,1,2,3]) {
            for (const [j, a] of [[0,'H'],[1,'Sw']]) {
                const bar = els['gBar'+i+a];
                const num = els['gNum'+i+a];
                if (num) num.textContent = (pol[i][j]*100).toFixed(1) + '%';
                if (bar) bar.style.width = (pol[i][j]*100) + '%';
            }
        }
    }

    return { init, update };
})();
