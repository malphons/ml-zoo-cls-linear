/* ============================================================
   Linear Discriminant Analysis — Data & Configuration
   Generates 2-class Gaussian clusters with different means,
   computes LDA projection axis and decision boundary.
   ============================================================ */
(function () {
    'use strict';

    /* ---------- Seeded PRNG (LCG, seed=77) ---------- */
    var seed = 77;
    function rand() {
        seed = (seed * 16807 + 0) % 2147483647;
        return (seed - 1) / 2147483646;
    }
    function randn() {
        var u = rand(), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    /* ---------- Generate 50 points, 2 classes ---------- */
    var points = [];
    var i;

    /* Class 0 — cluster around (3, 3.5), slight positive correlation */
    for (i = 0; i < 25; i++) {
        var u0 = randn(), v0 = randn();
        var x0 = 3.0 + 1.2 * u0 + 0.4 * v0;
        var y0 = 3.5 + 0.4 * u0 + 1.2 * v0;
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, x0)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, y0)) * 100) / 100,
            cls: 0
        });
    }

    /* Class 1 — cluster around (7, 6.5), same covariance structure */
    for (i = 0; i < 25; i++) {
        var u1 = randn(), v1 = randn();
        var x1 = 7.0 + 1.2 * u1 + 0.4 * v1;
        var y1 = 6.5 + 0.4 * u1 + 1.2 * v1;
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, x1)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, y1)) * 100) / 100,
            cls: 1
        });
    }

    /* ---------- Compute class means ---------- */
    var mean0 = { x: 0, y: 0 }, mean1 = { x: 0, y: 0 };
    var n0 = 0, n1 = 0;
    points.forEach(function (p) {
        if (p.cls === 0) { mean0.x += p.x; mean0.y += p.y; n0++; }
        else             { mean1.x += p.x; mean1.y += p.y; n1++; }
    });
    mean0.x /= n0; mean0.y /= n0;
    mean1.x /= n1; mean1.y /= n1;

    /* ---------- Compute pooled within-class covariance ---------- */
    var Sw = [[0, 0], [0, 0]];
    points.forEach(function (p) {
        var m = p.cls === 0 ? mean0 : mean1;
        var dx = p.x - m.x, dy = p.y - m.y;
        Sw[0][0] += dx * dx;
        Sw[0][1] += dx * dy;
        Sw[1][0] += dy * dx;
        Sw[1][1] += dy * dy;
    });

    /* ---------- Compute LDA direction: w = Sw^{-1} * (mean1 - mean0) ---------- */
    var det = Sw[0][0] * Sw[1][1] - Sw[0][1] * Sw[1][0];
    var SwInv = [
        [ Sw[1][1] / det, -Sw[0][1] / det],
        [-Sw[1][0] / det,  Sw[0][0] / det]
    ];
    var dmx = mean1.x - mean0.x;
    var dmy = mean1.y - mean0.y;
    var wLDA = {
        dx: SwInv[0][0] * dmx + SwInv[0][1] * dmy,
        dy: SwInv[1][0] * dmx + SwInv[1][1] * dmy
    };

    /* Normalise */
    var wLen = Math.sqrt(wLDA.dx * wLDA.dx + wLDA.dy * wLDA.dy);
    wLDA.dx /= wLen;
    wLDA.dy /= wLen;

    /* ---------- Decision boundary ---------- */
    /* The boundary is perpendicular to wLDA through the midpoint of class means */
    var midX = (mean0.x + mean1.x) / 2;
    var midY = (mean0.y + mean1.y) / 2;

    /* Boundary line: wLDA.dx * (x - midX) + wLDA.dy * (y - midY) = 0
       => wLDA.dx * x + wLDA.dy * y - (wLDA.dx * midX + wLDA.dy * midY) = 0 */
    var boundary = {
        w0: -(wLDA.dx * midX + wLDA.dy * midY),
        w1: wLDA.dx,
        w2: wLDA.dy
    };

    function classifyFn(x, y) {
        var z = boundary.w0 + boundary.w1 * x + boundary.w2 * y;
        return z >= 0 ? 1 : 0;
    }

    /* ---------- Export ---------- */
    window.MLZoo = window.MLZoo || {};
    window.MLZoo.modelData = {
        config: {
            width: 800,
            height: 400,
            xDomain: [0, 10],
            yDomain: [0, 10],
            accentColor: '#58a6ff',
            xLabel: 'Feature x\u2081',
            yLabel: 'Feature x\u2082'
        },
        points: points,
        boundary: boundary,
        classifyFn: classifyFn,
        projectionDirection: wLDA,
        mean0: mean0,
        mean1: mean1,
        Sw: Sw
    };
})();
