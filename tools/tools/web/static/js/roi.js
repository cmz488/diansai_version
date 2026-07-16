/** ROI selection overlay on stream images. */

let active = false;
let roiCanvas = null;
let roiCtx = null;
let drawing = false;
let startX, startY, endX, endY;

function onDown(e) {
  if (!active) return;
  drawing = true;
  const rect = roiCanvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
}

function onMove(e) {
  if (!active || !drawing) return;
  const rect = roiCanvas.getBoundingClientRect();
  endX = e.clientX - rect.left;
  endY = e.clientY - rect.top;
  drawRect();
}

function onUp() {
  if (!active || !drawing) return;
  drawing = false;
  showStats();
}

function isPointerSupported() {
  return typeof window !== "undefined" && window.PointerEvent !== undefined;
}

export function initROI(canvasId) {
  roiCanvas = document.getElementById(canvasId);
  if (!roiCanvas) return;
  roiCtx = roiCanvas.getContext("2d");

  if (isPointerSupported()) {
    roiCanvas.addEventListener("pointerdown", onDown);
    roiCanvas.addEventListener("pointermove", onMove);
    roiCanvas.addEventListener("pointerup", onUp);
  } else {
    // Fallback for Safari < 13 and older browsers
    roiCanvas.addEventListener("mousedown", onDown);
    roiCanvas.addEventListener("mousemove", onMove);
    roiCanvas.addEventListener("mouseup", onUp);
    // Touch fallback for mobile
    roiCanvas.addEventListener("touchstart", function(e) {
      e.preventDefault();
      onDown(e.touches[0]);
    });
    roiCanvas.addEventListener("touchmove", function(e) {
      e.preventDefault();
      onMove(e.touches[0]);
    });
    roiCanvas.addEventListener("touchend", onUp);
  }
}

export function toggleROI() {
  active = !active;
  if (!roiCanvas) return;

  // Move canvas into the first stream cell so CSS overlay works
  if (active) {
    var cell = document.querySelector(".stream-cell");
    if (cell && roiCanvas.parentElement !== cell) {
      // Match cell dimensions
      var img = cell.querySelector("img");
      if (img) {
        roiCanvas.style.width = img.style.width || "100%";
        roiCanvas.style.height = img.style.height || "100%";
      }
      cell.appendChild(roiCanvas);
    }
  }

  roiCanvas.classList.toggle("active", active);
  if (!active && roiCtx) {
    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  }
}

function drawRect() {
  if (!roiCtx) return;
  var w = endX - startX;
  var h = endY - startY;
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  roiCtx.strokeStyle = "#0071e3";
  roiCtx.lineWidth = 2;
  roiCtx.setLineDash([6, 3]);
  roiCtx.strokeRect(startX, startY, w, h);
  roiCtx.setLineDash([]);
}

function showStats() {
  var w = Math.abs(endX - startX);
  var h = Math.abs(endY - startY);
  var el = document.querySelector(".roi-stats");
  if (!el) {
    el = document.createElement("div");
    el.className = "roi-stats";
    roiCanvas.parentElement.appendChild(el);
  }
  el.textContent = Math.round(w) + "×" + Math.round(h) + " | 选区";
  el.style.display = "";
  setTimeout(function() { el.style.display = "none"; }, 3000);
}
