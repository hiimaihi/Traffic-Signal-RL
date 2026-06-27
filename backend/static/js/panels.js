/**
 * TrafficRL Cockpit — 数据面板更新
 * ================================
 * 负责更新所有指标卡片、排队柱状图、Q值/策略条。
 */

const Panels = (() => {
    // ── 元素引用 ──
    const els = {};

    function cacheEls() {
        els.metricReward = document.getElementById('metricReward');
        els.metricThroughput = document.getElementById('metricThroughput');
        els.metricWait = document.getElementById('metricWait');
        els.metricPenalty = document.getElementById('metricPenalty');
        els.metricAction = document.getElementById('metricAction');
        els.phaseIndicator = document.getElementById('phaseIndicator');
        els.navEpisode = document.getElementById('navEpisode');
        els.navStep = document.getElementById('navStep');
        els.navAgent = document.getElementById('navAgent');
        els.navStatus = document.getElementById('navStatus');

        // 排队条
        ['N', 'S', 'E', 'W'].forEach(d => {
            els[`queueBar${d}`] = document.getElementById(`queueBar${d}`);
            els[`queueNum${d}`] = document.getElementById(`queueNum${d}`);
        });

        // Q值
        els.qBarHold = document.getElementById('qBarHold');
        els.qBarSwitch = document.getElementById('qBarSwitch');
        els.qNumHold = document.getElementById('qNumHold');
        els.qNumSwitch = document.getElementById('qNumSwitch');
    }

    // ── 数字滚动效果 ──
    function animateNumber(el, newVal, fmt) {
        const oldText = el.textContent;
        const newText = typeof fmt === 'function' ? fmt(newVal) : String(newVal);
        if (oldText !== newText) {
            el.textContent = newText;
            el.classList.remove('number-changed');
            void el.offsetWidth; // reflow
            el.classList.add('number-changed');
        }
    }

    function fmtFloat(v) { return Number(v).toFixed(1); }
    function fmtInt(v) { return String(Math.round(v)); }
    function fmtPct(v) { return (Number(v) * 100).toFixed(1) + '%'; }

    // ── 主更新 ──
    function update(data) {
        // 导航栏
        animateNumber(els.navEpisode, data.episode, fmtInt);
        animateNumber(els.navStep, data.step, fmtInt);

        // 相位指示器
        const phaseLabel = data.phase_label || (data.phase === 0 ? 'NS' : 'EW');
        els.phaseIndicator.textContent = phaseLabel;
        els.phaseIndicator.className = 'phase-indicator' + (phaseLabel === 'EW' ? ' ew' : '');

        // 指标
        updateMetric(els.metricReward, data.cumulative_reward, fmtFloat);
        updateMetric(els.metricThroughput, data.throughput, fmtInt);
        if (els.metricWait) {
            const meanWait = data.mean_wait != null ? data.mean_wait :
                (data.avg_wait ? data.avg_wait.reduce((a,b)=>a+b,0)/Math.max(1,data.avg_wait.length) : 0);
            animateNumber(els.metricWait, meanWait, fmtFloat);
        }
        updateMetric(els.metricPenalty, data.switch_penalty, fmtFloat);

        const actionText = data.action === 0 ? 'HOLD' : 'SWITCH';
        els.metricAction.textContent = actionText;
        els.metricAction.style.color = data.action === 0 ? 'var(--green)' : 'var(--orange)';

        // 各方向排队
        const maxQueue = Math.max(1, ...(data.queues || [0,0,0,0]));
        ['N', 'S', 'E', 'W'].forEach((d, i) => {
            const q = data.queues ? data.queues[i] : 0;
            const pct = Math.min(100, (q / maxQueue) * 100);
            const bar = els[`queueBar${d}`];
            const num = els[`queueNum${d}`];
            if (bar) bar.style.width = pct + '%';
            if (num) num.textContent = q;
            if (bar) {
                bar.classList.remove('low', 'medium', 'high');
                if (q >= 12) bar.classList.add('high');
                else if (q >= 6) bar.classList.add('medium');
            }
        });

        // Q值/策略 → softmax 置信度 (支持负值, 数值稳定)
        const qs = data.policy && data.policy[0] + data.policy[1] > 0.01
            ? data.policy : (data.q_values || [0.0, 0.0]);
        const maxQ = Math.max(qs[0], qs[1]);
        const exp0 = Math.exp(qs[0] - maxQ);
        const exp1 = Math.exp(qs[1] - maxQ);
        const sumExp = exp0 + exp1;
        const qHold = sumExp > 0 ? exp0 / sumExp : 0.5;
        const qSwitch = sumExp > 0 ? exp1 / sumExp : 0.5;

        els.qBarHold.style.width = (qHold * 100).toFixed(1) + '%';
        els.qBarSwitch.style.width = (qSwitch * 100).toFixed(1) + '%';
        animateNumber(els.qNumHold, qHold, fmtPct);
        animateNumber(els.qNumSwitch, qSwitch, fmtPct);
    }

    function updateMetric(el, val, fmt) {
        const newText = fmt(val);
        if (el.textContent !== newText) {
            const oldVal = parseFloat(el.textContent);
            el.textContent = newText;
            el.classList.remove('positive', 'negative', 'number-changed');
            if (!isNaN(oldVal) && val > oldVal) el.classList.add('positive');
            else if (!isNaN(oldVal) && val < oldVal) el.classList.add('negative');
            void el.offsetWidth;
            el.classList.add('number-changed');
        }
    }

    function setAgent(agentName) {
        els.navAgent.textContent = agentName;
    }

    function setStatus(text, className) {
        els.navStatus.textContent = text;
        els.navStatus.className = 'nav-status';
        if (className) els.navStatus.classList.add(className);
    }

    // ── 算法对比渲染 ──
    function renderCompare(data) {
        const container = document.getElementById('compareResults');
        if (!container) return;
        if (!data || !data.results) {
            container.innerHTML = '<span class="compare-placeholder">对比数据无效</span>';
            return;
        }
        const valid = data.results.filter(r => !r.error);
        if (valid.length === 0) {
            container.innerHTML = '<span class="compare-error">所有算法评估失败</span>';
            return;
        }
        // 排序: avg_queue 升序
        valid.sort((a, b) => a.avg_queue - b.avg_queue);
        const best = valid[0];

        let html = `<div class="compare-meta">${data.pattern} · ${data.steps}步</div>`;
        html += '<table class="compare-table"><thead><tr>';
        html += '<th>算法</th><th>平均排队</th><th>平均等待</th><th>吞吐</th><th>奖励</th><th>切换率</th>';
        html += '</tr></thead><tbody>';

        for (const r of valid) {
            const isBest = r.agent === best.agent;
            const cls = isBest ? ' class="best-row"' : '';
            html += `<tr${cls}>`;
            html += `<td>${isBest ? '⭐ ' : ''}${getAgentName(r.agent)}</td>`;
            html += `<td class="mono">${r.avg_queue}</td>`;
            html += `<td class="mono">${r.avg_wait}</td>`;
            html += `<td class="mono">${r.throughput}</td>`;
            html += `<td class="mono">${r.reward}</td>`;
            html += `<td class="mono">${r.switch_rate}%</td>`;
            html += '</tr>';
        }

        html += '</tbody></table>';

        // 错误
        const errs = data.results.filter(r => r.error);
        if (errs.length > 0) {
            html += '<div class="compare-errors">';
            for (const e of errs) html += `<span>⚠ ${e.agent}: ${e.error}</span>`;
            html += '</div>';
        }
        container.innerHTML = html;
    }

    function getAgentName(key) {
        const m = {
            dqn:'DQN',double:'Double DQN',dueling:'Dueling DQN',
            noisy:'Noisy DQN',noisy_double:'Noisy Double',
            boltzmann:'Boltzmann',boltzmann_double:'Boltz×Double',
            per_dqn:'PER DQN',dueling_per:'Dueling PER',
            a2c:'A2C',ppo:'PPO',
        };
        return m[key] || key;
    }

    function setCompareLoading() {
        const container = document.getElementById('compareResults');
        if (container) container.innerHTML = '<span class="compare-placeholder">⏳ 正在评估所有算法...</span>';
    }

    // ── 初始化 ──
    function init() {
        cacheEls();
    }

    return { init, update, setAgent, setStatus, renderCompare, setCompareLoading };
})();
