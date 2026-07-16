/** Lightweight Canvas 2D line chart. Full redraw each frame. */

var MAX_POINTS = 120;

export function initChart(canvasId) {
  var canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  var ctx = canvas.getContext("2d");
  var series = {};
  var sized = false;

  function ensureSize() {
    if (sized) return;
    var rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return; // panel hidden
    var dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    sized = true;
  }

  function addSeries(key, color) {
    series[key] = { color: color, data: [] };
  }

  function push(key, value) {
    var s = series[key];
    if (!s) return;
    s.data.push(value);
    if (s.data.length > MAX_POINTS) s.data.shift();
    ensureSize();
    redraw();
  }

  function redraw() {
    if (!sized) { ensureSize(); if (!sized) return; }

    var dpr = window.devicePixelRatio || 1;
    var w = canvas.width / dpr;
    var h = canvas.height / dpr;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "#e5e5ea";
    ctx.lineWidth = 0.5;
    for (var i = 1; i < 4; i++) {
      var gy = (h / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
    }

    // Find global min/max
    var gmin = Infinity, gmax = -Infinity;
    for (var sk in series) {
      var s = series[sk];
      for (var i = 0; i < s.data.length; i++) {
        var v = s.data[i];
        if (v < gmin) gmin = v;
        if (v > gmax) gmax = v;
      }
    }
    if (!isFinite(gmin)) { gmin = 0; gmax = 100; }
    var range = gmax - gmin || 1;
    var pad = range * 0.1;

    // Draw each series
    for (var sk2 in series) {
      var s2 = series[sk2];
      if (s2.data.length < 2) continue;
      ctx.strokeStyle = s2.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (var i = 0; i < s2.data.length; i++) {
        var x = (i / (MAX_POINTS - 1)) * w;
        var y = h - ((s2.data[i] - (gmin - pad)) / (range + pad * 2)) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  function resize() {
    sized = false;
    ensureSize();
    redraw();
  }

  window.addEventListener("resize", resize);

  return {
    addSeries: addSeries,
    push: push,
    resize: resize,
    redraw: redraw
  };
}
