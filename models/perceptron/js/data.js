/* ============================================================
   Perceptron — Data with epoch-wise decision boundary updates
   ============================================================ */
(function () {
    'use strict';
    var seed = 88;
    function rand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
    function randn() { var u = rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

    var points = [];
    for (var i = 0; i < 20; i++) { points.push({ x: Math.round(Math.max(0.5, Math.min(9.5, 3 + randn() * 1.2)) * 100) / 100, y: Math.round(Math.max(0.5, Math.min(9.5, 3 + randn() * 1.2)) * 100) / 100, cls: 0 }); }
    for (var i = 0; i < 20; i++) { points.push({ x: Math.round(Math.max(0.5, Math.min(9.5, 7 + randn() * 1.2)) * 100) / 100, y: Math.round(Math.max(0.5, Math.min(9.5, 7 + randn() * 1.2)) * 100) / 100, cls: 1 }); }

    /* Simulate perceptron training — 10 epochs with progressively better boundary */
    var epochs = [
        { w1: 0.3, w2: 0.1, w0: -1.5 },
        { w1: 0.5, w2: 0.3, w0: -3.0 },
        { w1: 0.6, w2: 0.5, w0: -4.5 },
        { w1: 0.7, w2: 0.6, w0: -5.5 },
        { w1: 0.75, w2: 0.65, w0: -6.2 },
        { w1: 0.78, w2: 0.68, w0: -6.8 },
        { w1: 0.80, w2: 0.70, w0: -7.2 },
        { w1: 0.81, w2: 0.71, w0: -7.4 },
        { w1: 0.82, w2: 0.72, w0: -7.5 },
        { w1: 0.82, w2: 0.72, w0: -7.5 }
    ];

    function makeClassifyFn(epoch) {
        var e = epochs[Math.min(epoch, epochs.length - 1)];
        return function (x, y) { return (e.w0 + e.w1 * x + e.w2 * y) >= 0 ? 1 : 0; };
    }

    window.MLZoo = window.MLZoo || {};
    window.MLZoo.modelData = {
        config: { width: 800, height: 400, xDomain: [0, 10], yDomain: [0, 10], accentColor: '#58a6ff', xLabel: 'Feature x\u2081', yLabel: 'Feature x\u2082' },
        points: points,
        epochs: epochs,
        classifyFn: makeClassifyFn(9),
        makeClassifyFn: makeClassifyFn
    };
})();
