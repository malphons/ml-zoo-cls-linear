/* ============================================================
   Multinomial Logistic Regression — Data & Configuration
   Generates 3-class 2D data and softmax decision boundaries.
   ============================================================ */
(function () {
    'use strict';

    /* ---------- Seeded PRNG (LCG, seed=55) ---------- */
    var seed = 55;
    function rand() {
        seed = (seed * 16807 + 0) % 2147483647;
        return (seed - 1) / 2147483646;
    }
    function randn() {
        var u = rand(), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    /* ---------- Generate 60 points, 3 classes ---------- */
    var points = [];
    var i;

    /* Class 0 — cluster around (2.5, 7) */
    for (i = 0; i < 20; i++) {
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, 2.5 + randn() * 1.1)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, 7.0 + randn() * 1.1)) * 100) / 100,
            cls: 0
        });
    }

    /* Class 1 — cluster around (7.5, 7) */
    for (i = 0; i < 20; i++) {
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, 7.5 + randn() * 1.1)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, 7.0 + randn() * 1.1)) * 100) / 100,
            cls: 1
        });
    }

    /* Class 2 — cluster around (5, 2.5) */
    for (i = 0; i < 20; i++) {
        points.push({
            x: Math.round(Math.max(0.2, Math.min(9.8, 5.0 + randn() * 1.1)) * 100) / 100,
            y: Math.round(Math.max(0.2, Math.min(9.8, 2.5 + randn() * 1.1)) * 100) / 100,
            cls: 2
        });
    }

    /* ---------- Softmax classification ---------- */
    /* Approximate weight vectors for the three classes.
       Each class k has a weight vector [bias, w1, w2].
       The class with the highest score wins. */

    var W = [
        /* class 0: favours low x, high y */
        [-2.0, -1.5,  1.2],
        /* class 1: favours high x, high y */
        [-2.0,  1.5,  1.2],
        /* class 2: favours mid x, low y */
        [ 2.0,  0.0, -1.8]
    ];

    function softmaxScores(x, y) {
        var scores = [];
        var maxS = -Infinity;
        for (var k = 0; k < W.length; k++) {
            var s = W[k][0] + W[k][1] * (x - 5) + W[k][2] * (y - 5);
            scores.push(s);
            if (s > maxS) maxS = s;
        }
        /* Numerically stable softmax */
        var expSum = 0;
        var probs = [];
        for (var k2 = 0; k2 < scores.length; k2++) {
            var e = Math.exp(scores[k2] - maxS);
            probs.push(e);
            expSum += e;
        }
        for (var k3 = 0; k3 < probs.length; k3++) {
            probs[k3] /= expSum;
        }
        return probs;
    }

    function classifyFn(x, y) {
        var probs = softmaxScores(x, y);
        var bestK = 0;
        for (var k = 1; k < probs.length; k++) {
            if (probs[k] > probs[bestK]) bestK = k;
        }
        return bestK;
    }

    /* ---------- Boundary line segments ---------- */
    /* The decision boundary between class i and class j is a line where
       score_i = score_j. We compute the pairwise boundaries. */

    function getBoundarySegments() {
        var xd = [0, 10], yd = [0, 10];
        var pairs = [[0,1], [0,2], [1,2]];
        var segments = [];

        pairs.forEach(function (pair) {
            var a = pair[0], b = pair[1];
            /* W[a][0] + W[a][1]*(x-5) + W[a][2]*(y-5) = W[b][0] + W[b][1]*(x-5) + W[b][2]*(y-5) */
            var db = W[a][0] - W[b][0];
            var dw1 = W[a][1] - W[b][1];
            var dw2 = W[a][2] - W[b][2];
            /* db + dw1*(x-5) + dw2*(y-5) = 0 */
            /* y = 5 - (db + dw1*(x-5)) / dw2 */

            if (Math.abs(dw2) > 1e-10) {
                var xStart = xd[0], xEnd = xd[1];
                var yStart = 5 - (db + dw1 * (xStart - 5)) / dw2;
                var yEnd   = 5 - (db + dw1 * (xEnd - 5)) / dw2;

                /* Clip to domain */
                var pts = clipLine(xStart, yStart, xEnd, yEnd, xd, yd);
                if (pts) {
                    segments.push({
                        x1: pts.x1, y1: pts.y1,
                        x2: pts.x2, y2: pts.y2,
                        classes: [a, b]
                    });
                }
            } else if (Math.abs(dw1) > 1e-10) {
                var xv = 5 - db / dw1;
                if (xv >= xd[0] && xv <= xd[1]) {
                    segments.push({
                        x1: xv, y1: yd[0],
                        x2: xv, y2: yd[1],
                        classes: [a, b]
                    });
                }
            }
        });

        return segments;
    }

    function clipLine(x1, y1, x2, y2, xd, yd) {
        /* Simple parametric line clipping to rectangle */
        var dx = x2 - x1, dy = y2 - y1;
        var tmin = 0, tmax = 1;

        function clip(p, q) {
            if (Math.abs(p) < 1e-12) return q >= 0;
            var r = q / p;
            if (p < 0) { if (r > tmax) return false; if (r > tmin) tmin = r; }
            else        { if (r < tmin) return false; if (r < tmax) tmax = r; }
            return true;
        }

        if (!clip(-dx, x1 - xd[0])) return null;
        if (!clip( dx, xd[1] - x1)) return null;
        if (!clip(-dy, y1 - yd[0])) return null;
        if (!clip( dy, yd[1] - y1)) return null;

        return {
            x1: x1 + tmin * dx, y1: y1 + tmin * dy,
            x2: x1 + tmax * dx, y2: y1 + tmax * dy
        };
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
        classifyFn: classifyFn,
        softmaxScores: softmaxScores,
        getBoundarySegments: getBoundarySegments,
        W: W
    };
})();
