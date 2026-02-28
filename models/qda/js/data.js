/* ============================================================
   QDA — 2-class data with different covariance structures
   ============================================================ */
(function () {
    'use strict';
    var seed = 66;
    function rand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
    function randn() { var u = rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

    var points = [];
    /* Class 0: circular cluster at (3.5, 4) */
    for (var i = 0; i < 25; i++) {
        points.push({ x: Math.round(Math.max(0.2, Math.min(9.8, 3.5 + randn() * 1.0)) * 100) / 100, y: Math.round(Math.max(0.2, Math.min(9.8, 4.0 + randn() * 1.0)) * 100) / 100, cls: 0 });
    }
    /* Class 1: elongated cluster at (6.5, 6.5), rotated */
    for (var i = 0; i < 25; i++) {
        var u = randn() * 2.0, v = randn() * 0.6;
        var angle = 0.7;
        var rx = u * Math.cos(angle) - v * Math.sin(angle);
        var ry = u * Math.sin(angle) + v * Math.cos(angle);
        points.push({ x: Math.round(Math.max(0.2, Math.min(9.8, 6.5 + rx)) * 100) / 100, y: Math.round(Math.max(0.2, Math.min(9.8, 6.5 + ry)) * 100) / 100, cls: 1 });
    }

    /* QDA classify: uses Mahalanobis distance with per-class covariance */
    var m0 = { x: 3.5, y: 4.0 }, m1 = { x: 6.5, y: 6.5 };
    var cov0 = { a: 1, b: 0, c: 0, d: 1 }; /* identity-like */
    var cov1 = { a: 2.2, b: 1.4, c: 1.4, d: 1.6 }; /* elongated, rotated */

    function mahal(x, y, m, cov) {
        var det = cov.a * cov.d - cov.b * cov.c;
        var inv = { a: cov.d / det, b: -cov.b / det, c: -cov.c / det, d: cov.a / det };
        var dx = x - m.x, dy = y - m.y;
        return dx * (inv.a * dx + inv.b * dy) + dy * (inv.c * dx + inv.d * dy) + Math.log(det);
    }

    function classifyFnQDA(x, y) { return mahal(x, y, m0, cov0) > mahal(x, y, m1, cov1) ? 1 : 0; }
    /* LDA comparison: shared covariance (average) */
    var covShared = { a: 1.6, b: 0.7, c: 0.7, d: 1.3 };
    function classifyFnLDA(x, y) { return mahal(x, y, m0, covShared) > mahal(x, y, m1, covShared) ? 1 : 0; }

    window.MLZoo = window.MLZoo || {};
    window.MLZoo.modelData = {
        config: { width: 800, height: 400, xDomain: [0, 10], yDomain: [0, 10], accentColor: '#58a6ff', xLabel: 'Feature x\u2081', yLabel: 'Feature x\u2082' },
        points: points,
        classifyFn: classifyFnQDA,
        classifyFnLDA: classifyFnLDA,
        mean0: m0, mean1: m1
    };
})();
