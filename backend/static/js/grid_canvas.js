/**
 * TrafficRL Cockpit — 2×2 网格 Canvas 渲染
 * =========================================
 * 四个路口: I0(NW), I1(NE), I2(SW), I3(SE)
 * 内部分隔线 + 走廊标签
 */
const GridCanvas = (() => {
    let ctx, canvas;
    let vehicles = [];
    let phases = [0, 0, 0, 0];
    let prevVehicles = [];
    let frameId = null;
    let animProgress = 0;

    const easeOutCubic = t => 1 - Math.pow(1 - t, 3);

    const W = 800, H = 640;
    // 四个路口中心
    const CX = [200, 600, 200, 600];
    const CY = [210, 210, 430, 430];
    const RW = 60;     // road width
    const STOP = 40;
    const VLEN = 14, VWID = 8;

    const C = {
        bg: '#fafafa', road: '#e8e8ed', roadLine: '#d1d1d6',
        stopLine: '#fff', crosswalk: '#e0e0e0', centerIsland: '#f5f5f7',
        greenLight: '#34c759', greenGlow: 'rgba(52,199,89,0.35)',
        redLight: '#ff3b30', redGlow: 'rgba(255,59,48,0.35)',
        yellowLight: '#ffcc00',
        vehicleBody: '#0071e3', vehicleBodyWait: '#ff9500',
        vehicleOutline: '#fff', vehicleGlass: 'rgba(255,255,255,0.6)',
        corridor: '#d5d5da',
    };

    function lerp(a, b, t) { return a + (b - a) * t; }
    function findPrev(v) { return prevVehicles.find(p => p.vid === v.vid) || v; }

    function init() {
        canvas = document.getElementById('simCanvas');
        ctx = canvas.getContext('2d');
        canvas.width = W;
        canvas.height = H;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        draw();
    }

    function update(data) {
        prevVehicles = [...vehicles];
        vehicles = data.vehicles || [];
        phases = data.phases || [0, 0, 0, 0];
        animProgress = 0;
        if (frameId) cancelAnimationFrame(frameId);
        animateFrame();
    }

    function animateFrame() {
        animProgress = Math.min(1, animProgress + 0.18);
        draw(easeOutCubic(animProgress));
        if (animProgress < 1) frameId = requestAnimationFrame(animateFrame);
    }

    function draw(t = 1.0) {
        if (!ctx) return;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, W, H);

        // 背景道路网络
        drawRoadNetwork();
        // 四个路口
        for (let i = 0; i < 4; i++) {
            drawIntersection(CX[i], CY[i], i, t);
        }
        drawVehicles(t);
    }

    function drawRoadNetwork() {
        // 水平道路: 两条
        ctx.fillStyle = C.road;
        ctx.fillRect(0, CY[0] - RW/2, W, RW);           // 上行
        ctx.fillRect(0, CY[2] - RW/2, W, RW);           // 下行

        // 垂直道路: 两条 (每列)
        for (const col of [0, 1]) {
            const cx = CX[col];
            ctx.fillRect(cx - RW/2, 0, RW, H);
        }

        // 分隔虚线 (水平)
        for (const row of [0, 1]) {
            const cy = CY[row];
            ctx.strokeStyle = C.roadLine;
            ctx.lineWidth = 1.2;
            ctx.setLineDash([10, 6]);
            ctx.beginPath();
            ctx.moveTo(0, cy);
            ctx.lineTo(W, cy);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // 分隔虚线 (垂直)
        for (const col of [0, 1]) {
            const cx = CX[col];
            ctx.strokeStyle = C.roadLine;
            ctx.lineWidth = 1.2;
            ctx.setLineDash([10, 6]);
            ctx.beginPath();
            ctx.moveTo(cx, 0);
            ctx.lineTo(cx, H);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // 走廊标签
        ctx.fillStyle = '#aaa';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        const mx = (CX[0] + CX[1]) / 2;
        const my = (CY[0] + CY[2]) / 2;
        ctx.fillText('← 走廊 →', mx, CY[0] - RW/2 - 8);
        ctx.fillText('← 走廊 →', mx, CY[2] - RW/2 - 8);
        ctx.save();
        ctx.translate(CX[0] - RW/2 - 10, my);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('↕ 走廊 ↕', 0, 0);
        ctx.restore();
        ctx.save();
        ctx.translate(CX[1] + RW/2 + 10, my);
        ctx.rotate(Math.PI/2);
        ctx.fillText('↕ 走廊 ↕', 0, 0);
        ctx.restore();
    }

    function drawIntersection(cx, cy, idx, t) {
        // 停车线
        drawStopLine(cx - STOP, 0, cx - STOP, cy - RW/2);
        drawStopLine(cx + STOP, cy + RW/2, cx + STOP, Math.min(H, cy + RW/2 + STOP + 30));
        drawStopLine(0, cy + STOP, cx - RW/2, cy + STOP);
        drawStopLine(cx + RW/2, cy - STOP, Math.min(W, cx + RW/2 + STOP + 30), cy - STOP);

        // 中央方块
        ctx.fillStyle = C.centerIsland;
        ctx.fillRect(cx - RW/2, cy - RW/2, RW, RW);

        // 斑马线
        drawXwalk(cx - STOP - 6, cy - RW/2, true);
        drawXwalk(cx + STOP + 6, cy + RW/2, true);
        drawXwalk(cx + STOP + 6, cy - RW/2, false);
        drawXwalk(cx - STOP - 6, cy + RW/2, false);

        // 信号灯
        drawTrafficLightsAt(cx, cy, phases[idx]);

        // 标签
        ctx.fillStyle = '#555';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('I' + idx, cx, cy - RW/2 - STOP - 14);
    }

    function drawStopLine(x1, y1, x2, y2) {
        ctx.strokeStyle = C.stopLine;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    function drawXwalk(cx, cy, vert) {
        ctx.fillStyle = C.crosswalk;
        const w = vert ? 5 : 24;
        const h = vert ? 24 : 5;
        for (let i = -2; i <= 2; i++) {
            const dx = vert ? 0 : i * 7;
            const dy = vert ? i * 7 : 0;
            ctx.fillRect(cx + dx - w/2, cy + dy - h/2, w, h);
        }
    }

    function drawTrafficLightsAt(cx, cy, phase) {
        const positions = [
            { x: cx - STOP + 14, y: cy - RW/2 - 6, dir: 'NS' },
            { x: cx + STOP - 14, y: cy + RW/2 + 6, dir: 'NS' },
            { x: cx + RW/2 + 6, y: cy - STOP + 14, dir: 'EW' },
            { x: cx - RW/2 - 6, y: cy + STOP - 14, dir: 'EW' },
        ];
        positions.forEach(pos => {
            const isGreen = (pos.dir === 'NS' && phase === 0) || (pos.dir === 'EW' && phase === 1);
            drawLight(pos.x, pos.y, isGreen);
        });
    }

    function drawLight(x, y, isGreen) {
        const r = 5;
        const glow = ctx.createRadialGradient(x, y, r*0.3, x, y, r*2);
        if (isGreen) {
            glow.addColorStop(0, C.greenGlow);
            glow.addColorStop(1, 'rgba(52,199,89,0)');
        } else {
            glow.addColorStop(0, C.redGlow);
            glow.addColorStop(1, 'rgba(255,59,48,0)');
        }
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, r*2, 0, Math.PI*2);
        ctx.fill();
        ctx.fillStyle = isGreen ? C.greenLight : C.redLight;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI*2);
        ctx.fill();
    }

    function drawVehicles(t) {
        vehicles.forEach(v => {
            const prev = findPrev(v);
            const x = lerp(prev.x, v.x, t);
            const y = lerp(prev.y, v.y, t);
            ctx.save();
            ctx.translate(x, y);
            // 按方向旋转
            const angles = { N: 0, S: Math.PI, E: Math.PI/2, W: -Math.PI/2 };
            ctx.rotate(angles[v.dir] || 0);
            // 车身
            ctx.fillStyle = v.waiting ? C.vehicleBodyWait : C.vehicleBody;
            ctx.fillRect(-VLEN/2, -VWID/2, VLEN, VWID);
            // 轮廓
            ctx.strokeStyle = C.vehicleOutline;
            ctx.lineWidth = 1.2;
            ctx.strokeRect(-VLEN/2, -VWID/2, VLEN, VWID);
            // 车窗
            ctx.fillStyle = C.vehicleGlass;
            ctx.fillRect(-2, -VWID/2 + 2, 6, VWID - 4);
            ctx.restore();
        });
    }

    return { init, update };
})();
