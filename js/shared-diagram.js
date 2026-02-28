/* ============================================================
   ML Zoo — Shared 2D Classification Diagram for Linear Models
   Shows a 2D feature space with linear/quadratic decision
   boundaries, colored regions, and optional projection axis.
   ============================================================ */
(function () {
    'use strict';

    var svg, g, width, height, xScale, yScale, zoom;
    var config = {};
    var margin = { top: 20, right: 30, bottom: 45, left: 55 };

    var CLASS_COLORS = ['#58a6ff', '#f85149', '#3fb950', '#d29922'];

    /* ---------- Initialise the SVG canvas ---------- */

    function init(containerSelector, cfg) {
        config = cfg || {};
        var container = document.querySelector(containerSelector);
        if (!container) return;

        width  = config.width  || container.clientWidth || 800;
        height = config.height || 400;

        svg = d3.select(containerSelector)
            .append('svg')
            .attr('viewBox', '0 0 ' + width + ' ' + height)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%')
            .style('max-height', height + 'px');

        svg.append('defs')
            .append('clipPath')
            .attr('id', 'plot-clip')
            .append('rect')
            .attr('x', margin.left)
            .attr('y', margin.top)
            .attr('width', width - margin.left - margin.right)
            .attr('height', height - margin.top - margin.bottom);

        g = svg.append('g').attr('clip-path', 'url(#plot-clip)');

        var xDomain = config.xDomain || [0, 10];
        var yDomain = config.yDomain || [0, 10];

        xScale = d3.scaleLinear().domain(xDomain).range([margin.left, width - margin.right]);
        yScale = d3.scaleLinear().domain(yDomain).range([height - margin.bottom, margin.top]);

        /* Grid lines */
        var xGrid = svg.append('g')
            .attr('transform', 'translate(0,' + (height - margin.bottom) + ')')
            .call(d3.axisBottom(xScale).ticks(8).tickSize(-(height - margin.top - margin.bottom)).tickFormat(''));
        xGrid.attr('opacity', 0.08).select('.domain').remove();

        var yGrid = svg.append('g')
            .attr('transform', 'translate(' + margin.left + ',0)')
            .call(d3.axisLeft(yScale).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(''));
        yGrid.attr('opacity', 0.08).select('.domain').remove();

        var axisColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted') || '#6e7681';

        svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', 'translate(0,' + (height - margin.bottom) + ')')
            .call(d3.axisBottom(xScale).ticks(8))
            .selectAll('text,line,path').attr('stroke', axisColor).attr('fill', axisColor);

        svg.append('g')
            .attr('class', 'y-axis')
            .attr('transform', 'translate(' + margin.left + ',0)')
            .call(d3.axisLeft(yScale).ticks(6))
            .selectAll('text,line,path').attr('stroke', axisColor).attr('fill', axisColor);

        if (config.xLabel) {
            svg.append('text').attr('x', width / 2).attr('y', height - 5)
                .attr('text-anchor', 'middle').attr('fill', axisColor).attr('font-size', '12px')
                .text(config.xLabel);
        }
        if (config.yLabel) {
            svg.append('text').attr('x', -height / 2).attr('y', 15)
                .attr('transform', 'rotate(-90)').attr('text-anchor', 'middle')
                .attr('fill', axisColor).attr('font-size', '12px').text(config.yLabel);
        }

        zoom = d3.zoom().scaleExtent([0.5, 5])
            .on('zoom', function (event) { g.attr('transform', event.transform); });
        svg.call(zoom);
    }

    /* ---------- Draw classification points ---------- */

    function drawPoints(points, opts) {
        opts = opts || {};
        var radius = opts.radius || 5;

        g.selectAll('.data-point').remove();

        var pts = g.selectAll('.data-point')
            .data(points)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', function (d) { return xScale(d.x); })
            .attr('cy', function (d) { return yScale(d.y); })
            .attr('r', 0)
            .attr('fill', function (d) { return CLASS_COLORS[d.cls || 0]; })
            .attr('opacity', 0.8)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);

        pts.transition().duration(400).delay(function (d, i) { return i * 15; })
            .attr('r', radius);

        pts.on('mouseover', function (event, d) {
                d3.select(this).attr('r', radius + 3).attr('opacity', 1);
                showTooltip(event, d);
            })
            .on('mouseout', function () {
                d3.select(this).attr('r', radius).attr('opacity', 0.8);
                hideTooltip();
            });
    }

    /* ---------- Draw decision regions (heatmap) ---------- */

    function drawRegions(classifyFn, opts) {
        opts = opts || {};
        g.selectAll('.region-cell').remove();

        var xd = config.xDomain || [0, 10];
        var yd = config.yDomain || [0, 10];
        var res = opts.resolution || 50;
        var dx = (xd[1] - xd[0]) / res;
        var dy = (yd[1] - yd[0]) / res;
        var cellW = (width - margin.left - margin.right) / res;
        var cellH = (height - margin.top - margin.bottom) / res;

        var cells = [];
        for (var i = 0; i < res; i++) {
            for (var j = 0; j < res; j++) {
                var cx = xd[0] + (i + 0.5) * dx;
                var cy = yd[0] + (j + 0.5) * dy;
                cells.push({ x: cx, y: cy, cls: classifyFn(cx, cy), i: i, j: j });
            }
        }

        g.selectAll('.region-cell')
            .data(cells)
            .enter()
            .append('rect')
            .attr('class', 'region-cell')
            .attr('x', function (d) { return margin.left + d.i * cellW; })
            .attr('y', function (d) { return margin.top + (res - 1 - d.j) * cellH; })
            .attr('width', cellW + 0.5)
            .attr('height', cellH + 0.5)
            .attr('fill', function (d) { return CLASS_COLORS[d.cls || 0]; })
            .attr('opacity', 0)
            .transition()
            .duration(300)
            .attr('opacity', opts.opacity || 0.12);
    }

    /* ---------- Draw a linear decision boundary ---------- */
    /* Accepts coefficients w0, w1, w2 for: w0 + w1*x + w2*y = 0
       Or an array of line segments [{x1,y1,x2,y2}, ...] */

    function drawDecisionBoundary(boundary, opts) {
        opts = opts || {};
        g.selectAll('.decision-boundary').remove();

        if (Array.isArray(boundary)) {
            /* Array of line segments */
            boundary.forEach(function (seg, idx) {
                g.append('line')
                    .attr('class', 'decision-boundary')
                    .attr('x1', xScale(seg.x1)).attr('y1', yScale(seg.y1))
                    .attr('x2', xScale(seg.x1)).attr('y2', yScale(seg.y1))
                    .attr('stroke', opts.color || '#e3b341')
                    .attr('stroke-width', opts.width || 2)
                    .attr('stroke-dasharray', opts.dash || '8 4')
                    .attr('opacity', 0.85)
                    .transition().duration(500).delay(idx * 100)
                    .attr('x2', xScale(seg.x2)).attr('y2', yScale(seg.y2));
            });
        } else if (boundary && boundary.w0 !== undefined) {
            /* w0 + w1*x + w2*y = 0  =>  y = -(w0 + w1*x) / w2 */
            var xd = config.xDomain || [0, 10];
            var yd = config.yDomain || [0, 10];
            var x1 = xd[0], x2 = xd[1];
            var y1, y2;

            if (Math.abs(boundary.w2) > 1e-10) {
                y1 = -(boundary.w0 + boundary.w1 * x1) / boundary.w2;
                y2 = -(boundary.w0 + boundary.w1 * x2) / boundary.w2;
            } else {
                /* Vertical line: w0 + w1*x = 0 => x = -w0/w1 */
                var xv = -boundary.w0 / boundary.w1;
                x1 = xv; x2 = xv;
                y1 = yd[0]; y2 = yd[1];
            }

            g.append('line')
                .attr('class', 'decision-boundary')
                .attr('x1', xScale(x1)).attr('y1', yScale(y1))
                .attr('x2', xScale(x1)).attr('y2', yScale(y1))
                .attr('stroke', opts.color || '#e3b341')
                .attr('stroke-width', opts.width || 2)
                .attr('stroke-dasharray', opts.dash || '8 4')
                .attr('opacity', 0.85)
                .transition().duration(600)
                .attr('x2', xScale(x2)).attr('y2', yScale(y2));
        }
    }

    /* ---------- Draw a quadratic (curved) decision boundary ---------- */
    /* Accepts a function boundaryFn(x) => y or null, sampled across the x domain */

    function drawQuadraticBoundary(boundaryFn, opts) {
        opts = opts || {};
        g.selectAll('.decision-boundary').remove();

        var xd = config.xDomain || [0, 10];
        var yd = config.yDomain || [0, 10];
        var steps = opts.steps || 200;
        var pathPoints = [];

        for (var i = 0; i <= steps; i++) {
            var xv = xd[0] + (xd[1] - xd[0]) * i / steps;
            var yv = boundaryFn(xv);
            if (yv !== null && yv >= yd[0] && yv <= yd[1]) {
                pathPoints.push({ x: xv, y: yv });
            }
        }

        if (pathPoints.length < 2) return;

        var lineGen = d3.line()
            .x(function (d) { return xScale(d.x); })
            .y(function (d) { return yScale(d.y); })
            .curve(d3.curveBasis);

        var path = g.append('path')
            .attr('class', 'decision-boundary')
            .attr('d', lineGen(pathPoints))
            .attr('fill', 'none')
            .attr('stroke', opts.color || '#e3b341')
            .attr('stroke-width', opts.width || 2)
            .attr('stroke-dasharray', opts.dash || '8 4')
            .attr('opacity', 0.85);

        /* Animate drawing */
        var totalLen = path.node().getTotalLength();
        path.attr('stroke-dasharray', totalLen + ' ' + totalLen)
            .attr('stroke-dashoffset', totalLen)
            .transition().duration(800)
            .attr('stroke-dashoffset', 0)
            .on('end', function () {
                d3.select(this).attr('stroke-dasharray', opts.dash || '8 4');
            });
    }

    /* ---------- Draw projection axis (for LDA) ---------- */
    /* Shows projected data points along a discriminant direction */

    function drawProjection(points, direction, opts) {
        opts = opts || {};
        g.selectAll('.projection-line,.projection-point,.projection-axis').remove();

        var xd = config.xDomain || [0, 10];
        var yd = config.yDomain || [0, 10];
        var cx = (xd[0] + xd[1]) / 2;
        var cy = (yd[0] + yd[1]) / 2;

        /* direction = {dx, dy} normalised */
        var len = Math.sqrt(direction.dx * direction.dx + direction.dy * direction.dy);
        var dx = direction.dx / len;
        var dy = direction.dy / len;

        /* Draw the projection axis line across the plot */
        var ext = Math.max(xd[1] - xd[0], yd[1] - yd[0]);
        var ax1 = cx - dx * ext, ay1 = cy - dy * ext;
        var ax2 = cx + dx * ext, ay2 = cy + dy * ext;

        g.append('line')
            .attr('class', 'projection-axis')
            .attr('x1', xScale(ax1)).attr('y1', yScale(ay1))
            .attr('x2', xScale(ax2)).attr('y2', yScale(ay2))
            .attr('stroke', opts.axisColor || '#8b949e')
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '4 3')
            .attr('opacity', 0.5);

        /* Project each point onto the axis and draw connection lines + projected dots */
        points.forEach(function (p, i) {
            var relX = p.x - cx;
            var relY = p.y - cy;
            var scalar = relX * dx + relY * dy;
            var projX = cx + scalar * dx;
            var projY = cy + scalar * dy;

            g.append('line')
                .attr('class', 'projection-line')
                .attr('x1', xScale(p.x)).attr('y1', yScale(p.y))
                .attr('x2', xScale(p.x)).attr('y2', yScale(p.y))
                .attr('stroke', CLASS_COLORS[p.cls || 0])
                .attr('stroke-width', 0.8)
                .attr('opacity', 0)
                .transition().duration(400).delay(i * 10)
                .attr('x2', xScale(projX)).attr('y2', yScale(projY))
                .attr('opacity', 0.3);

            g.append('circle')
                .attr('class', 'projection-point')
                .attr('cx', xScale(projX))
                .attr('cy', yScale(projY))
                .attr('r', 0)
                .attr('fill', CLASS_COLORS[p.cls || 0])
                .attr('stroke', '#fff')
                .attr('stroke-width', 0.8)
                .attr('opacity', 0.9)
                .transition().duration(400).delay(i * 10 + 200)
                .attr('r', 3.5);
        });
    }

    /* ---------- Tooltip ---------- */

    var tooltipEl = null;

    function showTooltip(event, d) {
        if (!tooltipEl) {
            tooltipEl = document.createElement('div');
            tooltipEl.style.cssText = 'position:fixed;padding:6px 10px;background:rgba(0,0,0,.85);' +
                'color:#fff;font-size:12px;border-radius:4px;pointer-events:none;z-index:999;';
            document.body.appendChild(tooltipEl);
        }
        var label = d.label || ('Class ' + (d.cls || 0));
        tooltipEl.textContent = '(' + d.x.toFixed(2) + ', ' + d.y.toFixed(2) + ') \u2014 ' + label;
        tooltipEl.style.left = event.clientX + 12 + 'px';
        tooltipEl.style.top = event.clientY - 28 + 'px';
        tooltipEl.style.display = 'block';
    }

    function hideTooltip() {
        if (tooltipEl) tooltipEl.style.display = 'none';
    }

    /* ---------- Clear & reset ---------- */

    function clear() {
        if (g) g.selectAll('.data-point,.region-cell,.decision-boundary,.projection-line,.projection-point,.projection-axis').remove();
    }

    function resetZoom() {
        if (svg && zoom) svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    }

    /* ---------- Public API ---------- */

    window.MLZoo = window.MLZoo || {};
    window.MLZoo.diagram = {
        init: init,
        drawPoints: drawPoints,
        drawRegions: drawRegions,
        drawDecisionBoundary: drawDecisionBoundary,
        drawQuadraticBoundary: drawQuadraticBoundary,
        drawProjection: drawProjection,
        clear: clear,
        resetZoom: resetZoom,
        CLASS_COLORS: CLASS_COLORS,
        getScales: function () { return { x: xScale, y: yScale }; },
        getGroup: function () { return g; },
        getSvg: function () { return svg; }
    };
})();
