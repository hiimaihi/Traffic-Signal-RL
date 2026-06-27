/**
 * TrafficRL Cockpit — 多路口 Canvas 渲染
 * ========================================
 * 1×2 走廊: 左侧 I0, 右侧 I1, 中间连接道路。
 */
const MultiCanvas = (() => {
    let ctx, canvas;
    let vehicles = [];
    let phases = [0, 0];
    let prevVehicles = [];
    let frameId = null;
    let animProgress = 0;

    const easeOutCubic = t => 1 - Math.pow(1 - t, 3);

    const SIZE_W = 800, SIZE_H = 420;
    const I0_CX = 200, I1_CX = 600, CY = 210;
    const ROAD_W = 60, STOP = 40;
    const VEH_LEN = 16, VEH_WID = 9;

    const C = {
        bg: '#fafafa', road: '#e8e8ed', roadLine: '#d1d1d6',
        stopLine: '#ffffff', crosswalk: '#e0e0e0', centerIsland: '#f5f5f7',
        greenLight: '#34c759', greenGlow: 'rgba(52,199,89,0.35)',
        redLight: '#ff3b30', redGlow: 'rgba(255,59,48,0.35)',
        yellowLight: '#ffcc00',
        vehicleBody: '#0071e3', vehicleBodyWait: '#ff9500',
        vehicleOutline: '#ffffff', vehicleGlass: 'rgba(255,255,255,0.6)',
        corridor: '#d5d5da',
    };

    function lerp(a, b, t) { return a + (b - a) * t; }
    function lerpV(a, b, t) { return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) }; }
    function findPrev(v) { return prevVehicles.find(p => p.vid === v.vid) || v; }

    function init() {
        canvas = document.getElementById('simCanvas');
        ctx = canvas.getContext('2d');
        canvas.width = SIZE_W;
        canvas.height = SIZE_H;
        canvas.style.width = SIZE_W + 'px';
        canvas.style.height = SIZE_H + 'px';
        draw();
    }

    function update(data) {
        prevVehicles = [...vehicles];
        vehicles = data.vehicles || [];
        phases = data.phases || [0, 0];
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
        ctx.clearRect(0, 0, SIZE_W, SIZE_H);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, SIZE_W, SIZE_H);

        // 绘制两个路口及连接道路
        drawCorridor();
        drawIntersection(I0_CX, CY, 0, t);
        drawIntersection(I1_CX, CY, 1, t);
        drawVehicles(t);
    }

    function drawCorridor() {
        // I0 右侧到 I1 左侧的连接道路
        const x0 = I0_CX + ROAD_W/2 + STOP;
        const x1 = I1_CX - ROAD_W/2 - STOP;
        ctx.fillStyle = C.road;
        ctx.fillRect(x0, CY - ROAD_W/2, x1 - x0, ROAD_W);

        // 道路中线 (虚线)
        ctx.strokeStyle = C.roadLine;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([10, 6]);
        ctx.beginPath();
        ctx.moveTo(x0, CY);
        ctx.lineTo(x1, CY);
        ctx.stroke();
        ctx.setLineDash([]);

        // 标签
        ctx.fillStyle = '#aaa';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('← 走廊 (3步延迟) →', (x0+x1)/2, CY - ROAD_W/2 - 10);
    }

    function drawIntersection(cx, cy, idx, t) {
        const rw = ROAD_W;

        // 东西向主路
        ctx.fillStyle = C.road;
        ctx.fillRect(0, cy - rw/2, SIZE_W, rw);
        // 南北向主路 (每路口独立)
        ctx.fillRect(cx - rw/2, 0, rw, SIZE_H);

        // 中央分界线
        ctx.strokeStyle = C.roadLine;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([10, 6]);
        ctx.beginPath();
        ctx.moveTo(Math.max(0, cx - rw/2 - STOP - 30), cy);
        ctx.lineTo(Math.min(SIZE_W, cx + rw/2 + STOP + 30), cy);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(cx, Math.max(0, cy - rw/2 - STOP - 30));
        ctx.lineTo(cx, Math.min(SIZE_H, cy + rw/2 + STOP + 30));
        ctx.stroke();
        ctx.setLineDash([]);

        // 停车线 (白实线)
        drawStopLine(cx - STOP, 0, cx - STOP, cy - rw/2);
        drawStopLine(cx + STOP, cy + rw/2, cx + STOP, SIZE_H);
        drawStopLine(0, cy + STOP, cx - rw/2, cy + STOP);
        drawStopLine(cx + rw/2, cy - STOP, SIZE_W, cy - STOP);

        // 中央方块
        ctx.fillStyle = C.centerIsland;
        ctx.fillRect(cx - rw/2, cy - rw/2, rw, rw);

        // 斑马线
        drawXwalk(cx - STOP - 6, cy - rw/2, true);
        drawXwalk(cx + STOP + 6, cy + rw/2, true);
        drawXwalk(cx + STOP + 6, cy - rw/2, false);
        drawXwalk(cx - STOP - 6, cy + rw/2, false);

        // 信号灯
        drawTrafficLightsAt(cx, cy, phases[idx]);

        // 路口标签
        ctx.fillStyle = '#666';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('I' + idx, cx, cy - rw/2 - STOP - 16);
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
        const w = vert ? 5 : 26;
        const h = vert ? 26 : 5;
        for (let i = -2; i <= 2; i++) {
            const dx = vert ? 0 : i * 7;
            const dy = vert ? i * 7 : 0;
            ctx.fillRect(cx + dx - w/2, cy + dy - h/2, w, h);
        }
    }

    function drawTrafficLightsAt(cx, cy, phase) {
        const positions = [
            { x: cx - STOP + 16, y: cy - ROAD_W/2 - 6, dir: 'NS' },
            { x: cx + STOP - 16, y: cy + ROAD_W/2 + 6, dir: 'NS' },
            { x: cx + ROAD_W/2 + 6, y: cy - STOP + 16, dir: 'EW' },
            { x: cx - ROAD_W/2 - 6, y: cy + STOP - 16, dir: 'EW' },
        ];
        positions.forEach(pos => {
            const isGreen = (pos.dir === 'NS' && phase === 0) || (pos.dir === 'EW' && phase === 1);
            drawLight(pos.x, pos.y, isGreen);
        });
    }

    function drawLight(x, y, isGreen) {
        const r = 6;
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

        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.beginPath();
        ctx.arc(x-1, y-1, r*0.3, 0, Math.PI*2);
        ctx.fill();
    }

    function drawVehicles(t) {
        vehicles.forEach(v => {
            const prev = findPrev(v);
            const pos = lerpV(prev, v, t);
            const x = Math.max(-20, Math.min(SIZE_W+20, pos.x));
            const y = Math.max(-20, Math.min(SIZE_H+20, pos.y));
            drawSingleVehicle(x, y, v.dir, v.waiting);
        });
    }

    function drawSingleVehicle(x, y, dir, waiting) {
        ctx.save();
        ctx.translate(x, y);
        const angles = { N: 0, S: Math.PI, E: Math.PI/2, W: -Math.PI/2 };
        ctx.rotate(angles[dir] || 0);

        const len = VEH_LEN, wid = VEH_WID;
        ctx.fillStyle = 'rgba(0,0,0,0.08)';
        ctx.fillRect(-len/2+2, -wid/2+2, len, wid);

        ctx.fillStyle = waiting ? C.vehicleBodyWait : C.vehicleBody;
        ctx.beginPath();
        roundRect(ctx, -len/2, -wid/2, len, wid, 3);
        ctx.fill();

        ctx.fillStyle = C.vehicleGlass;
        ctx.beginPath();
        roundRect(ctx, len/2-7, -wid/2+2, 4, wid-4, 1.5);
        ctx.fill();

        ctx.strokeStyle = C.vehicleOutline;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        roundRect(ctx, -len/2, -wid/2, len, wid, 3);
        ctx.stroke();

        ctx.restore();
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.moveTo(x+r, y); ctx.lineTo(x+w-r, y);
        ctx.quadraticCurveTo(x+w, y, x+w, y+r);
        ctx.lineTo(x+w, y+h-r);
        ctx.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
        ctx.lineTo(x+r, y+h);
        ctx.quadraticCurveTo(x, y+h, x, y+h-r);
        ctx.lineTo(x, y+r);
        ctx.quadraticCurveTo(x, y, x+r, y);
    }

    return { init, update, draw };
})();
