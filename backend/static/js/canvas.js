/**
 * TrafficRL Cockpit — Canvas 动画渲染引擎
 * =========================================
 * 负责:
 *   - 道路网 & 路口绘制
 *   - 信号灯状态实时渲染
 *   - 车辆动画 (排队 / 行驶 / 通过路口)
 *   - 粒子效果 & 过渡动画
 */

const Canvas = (() => {
    // ── 私有状态 ──
    let ctx, canvas;
    let vehicles = [];
    let phase = 0;
    let prevVehicles = [];
    let frameId = null;
    let animProgress = 0; // 0→1 插值进度

    // 缓动函数
    const easeOutCubic = t => 1 - Math.pow(1 - t, 3);
    const easeInOutCubic = t => t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;

    // ── 常量 ──
    const SIZE = 720;
    const CENTER = 360;
    const ROAD_W = 80;
    const STOP_OFFSET = 50;
    const VEH_LEN = 18;
    const VEH_WID = 10;

    // ── 颜色 ──
    const COLORS = {
        bg: '#fafafa',
        road: '#e8e8ed',
        roadLine: '#d1d1d6',
        stopLine: '#ffffff',
        crosswalk: '#e0e0e0',
        centerIsland: '#f5f5f7',
        greenLight: '#34c759',
        greenGlow: 'rgba(52, 199, 89, 0.35)',
        redLight: '#ff3b30',
        redGlow: 'rgba(255, 59, 48, 0.35)',
        yellowLight: '#ffcc00',
        vehicleBody: '#0071e3',
        vehicleBodyWait: '#ff9500',
        vehicleOutline: '#ffffff',
        vehicleGlass: 'rgba(255,255,255,0.6)',
    };

    // ── 工具函数 ──
    function lerp(a, b, t) { return a + (b - a) * t; }

    function lerpV(a, b, t) {
        return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
    }

    function findPrev(v) {
        return prevVehicles.find(p => p.vid === v.vid) || v;
    }

    // ── 初始化 ──
    function init() {
        canvas = document.getElementById('simCanvas');
        ctx = canvas.getContext('2d');
        canvas.width = SIZE;
        canvas.height = SIZE;
        // 初始绑定防止缩放问题
        canvas.style.width = SIZE + 'px';
        canvas.style.height = SIZE + 'px';
        draw();
    }

    // ── 接收新数据 ──
    function update(data) {
        prevVehicles = [...vehicles];
        vehicles = data.vehicles || [];
        phase = data.phase || 0;
        animProgress = 0;

        // 开始动画插值循环
        if (frameId) cancelAnimationFrame(frameId);
        animateFrame();
    }

    function animateFrame() {
        animProgress = Math.min(1, animProgress + 0.18);
        const t = easeOutCubic(animProgress);
        draw(t);
        if (animProgress < 1) {
            frameId = requestAnimationFrame(animateFrame);
        }
    }

    // ── 主绘制 ──
    function draw(t = 1.0) {
        if (!ctx) return;
        ctx.clearRect(0, 0, SIZE, SIZE);
        drawBackground();
        drawRoads();
        drawTrafficLights();
        drawVehicles(t);
    }

    // ── 背景 ──
    function drawBackground() {
        // 外圈绿地/街区
        ctx.fillStyle = '#ecedf0';
        ctx.fillRect(0, 0, SIZE, SIZE);
        // 中心画布底色
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(0, 0, SIZE, SIZE);
    }

    // ── 道路 ──
    function drawRoads() {
        ctx.fillStyle = COLORS.road;

        // 东西向主路
        ctx.fillRect(0, CENTER - ROAD_W/2, SIZE, ROAD_W);
        // 南北向主路
        ctx.fillRect(CENTER - ROAD_W/2, 0, ROAD_W, SIZE);

        // 中央分界线 (虚线)
        ctx.strokeStyle = COLORS.roadLine;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([12, 8]);

        ctx.beginPath();
        ctx.moveTo(0, CENTER);
        ctx.lineTo(SIZE, CENTER);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(CENTER, 0);
        ctx.lineTo(CENTER, SIZE);
        ctx.stroke();
        ctx.setLineDash([]);

        // 停车线 (白实线)
        drawStopLine(CENTER - STOP_OFFSET, 0, CENTER - STOP_OFFSET, CENTER - ROAD_W/2); // 东入口
        drawStopLine(CENTER + STOP_OFFSET, CENTER + ROAD_W/2, CENTER + STOP_OFFSET, SIZE); // 西入口
        drawStopLine(0, CENTER + STOP_OFFSET, CENTER - ROAD_W/2, CENTER + STOP_OFFSET); // 南入口
        drawStopLine(CENTER + ROAD_W/2, CENTER - STOP_OFFSET, SIZE, CENTER - STOP_OFFSET); // 北入口

        // 中央方块 (路口区域)
        ctx.fillStyle = COLORS.centerIsland;
        ctx.fillRect(CENTER - ROAD_W/2, CENTER - ROAD_W/2, ROAD_W, ROAD_W);

        // 斑马线
        drawCrosswalk(CENTER - STOP_OFFSET - 8, CENTER - ROAD_W/2, true);
        drawCrosswalk(CENTER + STOP_OFFSET + 8, CENTER + ROAD_W/2, true);
        drawCrosswalk(CENTER + STOP_OFFSET + 8, CENTER - ROAD_W/2, false);
        drawCrosswalk(CENTER - STOP_OFFSET - 8, CENTER + ROAD_W/2, false);
    }

    function drawStopLine(x1, y1, x2, y2) {
        ctx.strokeStyle = COLORS.stopLine;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    function drawCrosswalk(cx, cy, vertical) {
        ctx.fillStyle = COLORS.crosswalk;
        const w = vertical ? 6 : 30;
        const h = vertical ? 30 : 6;
        for (let i = -2; i <= 2; i++) {
            const dx = vertical ? 0 : i * 8;
            const dy = vertical ? i * 8 : 0;
            ctx.fillRect(cx + dx - w/2, cy + dy - h/2, w, h);
        }
    }

    // ── 信号灯 ──
    function drawTrafficLights() {
        const positions = [
            { x: CENTER - STOP_OFFSET + 18, y: CENTER - ROAD_W/2 - 8, dir: 'NS' }, // 东侧NS
            { x: CENTER + STOP_OFFSET - 18, y: CENTER + ROAD_W/2 + 8, dir: 'NS' }, // 西侧NS
            { x: CENTER + ROAD_W/2 + 8, y: CENTER - STOP_OFFSET + 18, dir: 'EW' }, // 北侧EW
            { x: CENTER - ROAD_W/2 - 8, y: CENTER + STOP_OFFSET - 18, dir: 'EW' }, // 南侧EW
        ];

        positions.forEach(pos => {
            const isGreen = (pos.dir === 'NS' && phase === 0) || (pos.dir === 'EW' && phase === 1);
            drawSingleLight(pos.x, pos.y, isGreen);
        });
    }

    function drawSingleLight(x, y, isGreen) {
        const r = 7;
        // 发光效果
        const glow = ctx.createRadialGradient(x, y, r*0.3, x, y, r*2.5);
        if (isGreen) {
            glow.addColorStop(0, COLORS.greenGlow);
            glow.addColorStop(1, 'rgba(52,199,89,0)');
        } else {
            glow.addColorStop(0, COLORS.redGlow);
            glow.addColorStop(1, 'rgba(255,59,48,0)');
        }
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
        ctx.fill();

        // 灯体
        ctx.fillStyle = isGreen ? COLORS.greenLight : COLORS.redLight;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();

        // 高光
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.beginPath();
        ctx.arc(x - 1.5, y - 1.5, r * 0.35, 0, Math.PI * 2);
        ctx.fill();
    }

    // ── 车辆 ──
    function drawVehicles(t) {
        vehicles.forEach(v => {
            const prev = findPrev(v);
            const pos = lerpV(prev, v, t);

            // 限制在画布内
            const x = Math.max(-20, Math.min(SIZE + 20, pos.x));
            const y = Math.max(-20, Math.min(SIZE + 20, pos.y));

            drawSingleVehicle(x, y, v.dir, v.waiting, v.speed);
        });
    }

    function drawSingleVehicle(x, y, dir, waiting, speed) {
        ctx.save();
        ctx.translate(x, y);

        // 根据方向旋转
        const angles = { N: 0, S: Math.PI, E: Math.PI/2, W: -Math.PI/2 };
        ctx.rotate(angles[dir] || 0);

        const len = VEH_LEN;
        const wid = VEH_WID;

        // 阴影
        ctx.fillStyle = 'rgba(0,0,0,0.08)';
        ctx.fillRect(-len/2 + 2, -wid/2 + 2, len, wid);

        // 车身
        const bodyColor = waiting ? COLORS.vehicleBodyWait : COLORS.vehicleBody;
        ctx.fillStyle = bodyColor;
        ctx.beginPath();
        roundRect(ctx, -len/2, -wid/2, len, wid, 4);
        ctx.fill();

        // 车窗 (后部)
        ctx.fillStyle = COLORS.vehicleGlass;
        ctx.beginPath();
        roundRect(ctx, len/2 - 8, -wid/2 + 2, 5, wid - 4, 2);
        ctx.fill();

        // 轮廓线
        ctx.strokeStyle = COLORS.vehicleOutline;
        ctx.lineWidth = 1;
        ctx.beginPath();
        roundRect(ctx, -len/2, -wid/2, len, wid, 4);
        ctx.stroke();

        // 如果是等待车辆且有速度, 显示微小抖动
        if (waiting && speed < 0.1) {
            // no extra effect
        }

        ctx.restore();
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
    }

    // ── 公开 API ──
    return { init, update, draw };
})();
