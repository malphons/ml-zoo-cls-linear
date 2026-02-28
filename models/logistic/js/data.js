/* ============================================================
   Logistic Regression — Data & Configuration
   Generates 2-class 2D data with overlap for binary
   logistic regression demonstration.
   ============================================================ */
(function () {
    'use strict';

    /* ---------- Seeded PRNG (LCG, seed=42) ---------- */
    var seed = 42;
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

    /* Class 0 — cluster around (3, 6) */
    for (i = 0; i < 25; i++) {
        var x0 = 3 + randn() * 1.3;
        var y0 = 6 + randn() * 1.5;
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, x0)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, y0)) * 100) / 100,
            cls: 0
        });
    }

    /* Class 1 — cluster around (7, 4) */
    for (i = 0; i < 25; i++) {
        var x1 = 7 + randn() * 1.3;
        var y1 = 4 + randn() * 1.5;
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, x1)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, y1)) * 100) / 100,
            cls: 1
        });
    }

    /* ---------- Decision boundary coefficients ---------- */
    /* The boundary is: w0 + w1*x + w2*y = 0
       Approximate logistic regression fit:
       P(class=1) = sigmoid(w0 + w1*x + w2*y)
       With classes centred at (3,6) and (7,4):
       direction vector ~ (4, -2), normalised decision at midpoint (5, 5)
       w1 = 2.0, w2 = -1.0, w0 = -(2.0*5 + (-1.0)*5) = -5  */

    function getBoundary(C) {
        /* Simulate effect of regularisation C on boundary.
           Higher C = less regularisation = steeper coefficients.
           Lower C = more regularisation = coefficients shrink toward 0. */
        var scale = 1 - 1 / (1 + C);  /* ranges from 0 (C=0) to ~1 (C large) */
        var w1 = 2.0 * scale;
        var w2 = -1.0 * scale;
        var w0 = -(w1 * 5 + w2 * 5);
        return { w0: w0, w1: w1, w2: w2 };
    }

    function makeClassifyFn(C) {
        var b = getBoundary(C);
        return function (x, y) {
            var z = b.w0 + b.w1 * x + b.w2 * y;
            return z >= 0 ? 1 : 0;
        };
    }

    /* ---------- Sigmoid curve data for side chart ---------- */
    var sigmoidData = [];
    for (i = -60; i <= 60; i++) {
        var t = i / 10;
        sigmoidData.push({ t: t, sigma: 1 / (1 + Math.exp(-t)) });
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
        getBoundary: getBoundary,
        makeClassifyFn: makeClassifyFn,
        sigmoidData: sigmoidData
    };
})();
