/* ============================================================
   Ridge Classifier — 2-class data with L2-regularized boundary
   ============================================================ */
(function () {
    'use strict';
    var seed = 33;
    function rand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
    function randn() { var u = rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

    var points = [];
    for (var i = 0; i < 25; i++) { points.push({ x: Math.round(Math.max(0.3, Math.min(9.7, 3.5 + randn() * 1.5)) * 100) / 100, y: Math.round(Math.max(0.3, Math.min(9.7, 3.5 + randn() * 1.5)) * 100) / 100, cls: 0 }); }
    for (var i = 0; i < 25; i++) { points.push({ x: Math.round(Math.max(0.3, Math.min(9.7, 6.5 + randn() * 1.5)) * 100) / 100, y: Math.round(Math.max(0.3, Math.min(9.7, 6.5 + randn() * 1.5)) * 100) / 100, cls: 1 }); }

    /* Boundary for different alpha values */
    var alphas = {
        '0.01': { w0: -7.8, w1: 0.85, w2: 0.75 },
        '0.1':  { w0: -7.5, w1: 0.82, w2: 0.72 },
        '1':    { w0: -7.0, w1: 0.78, w2: 0.68 },
        '10':   { w0: -6.2, w1: 0.70, w2: 0.62 },
        '100':  { w0: -5.5, w1: 0.60, w2: 0.55 }
    };

    function makeClassifyFn(alpha) {
        var key = String(alpha);
        var b = alphas[key] || alphas['1'];
        return function (x, y) { return (b.w0 + b.w1 * x + b.w2 * y) >= 0 ? 1 : 0; };
    }

    function getBoundary(alpha) {
        var key = String(alpha);
        return alphas[key] || alphas['1'];
    }

    window.MLZoo = window.MLZoo || {};
    window.MLZoo.modelData = {
        config: { width: 800, height: 400, xDomain: [0, 10], yDomain: [0, 10], accentColor: '#58a6ff', xLabel: 'Feature x\u2081', yLabel: 'Feature x\u2082' },
        points: points,
        classifyFn: makeClassifyFn(1),
        makeClassifyFn: makeClassifyFn,
        getBoundary: getBoundary,
        alphas: alphas
    };
})();
