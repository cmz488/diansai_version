/** Multi-view grid layout manager with responsive mobile swipe. */

var gridMode = "cols2";
var streamChannels = [0, 1, 2, 3];
var currentMobileIdx = 0;

export function setGrid(mode) {
  gridMode = mode;
  var grid = document.getElementById("streamGrid");
  if (!grid) return;
  grid.className = "grid " + mode;
  renderCells();
}

export function setChannel(cellIdx, channelId) {
  streamChannels[cellIdx] = channelId;
  renderCells();
}

function renderCells() {
  var grid = document.getElementById("streamGrid");
  if (!grid) return;
  var count = gridMode === "cols1" ? 1 : gridMode === "cols2" ? 2 : 4;
  grid.innerHTML = "";
  for (var i = 0; i < count; i++) {
    var ch = streamChannels[i] != null ? streamChannels[i] : i;
    var cell = document.createElement("div");
    cell.className = "stream-cell";
    cell.innerHTML =
      '<div class="cell-label">通道 ' + ch + '</div>' +
      '<img src="/stream/' + ch + '" alt="" />';
    grid.appendChild(cell);
  }
}

export function initMobileSwipe() {
  var vp = document.querySelector(".viewport");
  if (!vp) return;
  var sx = 0;
  vp.addEventListener("touchstart", function(e) { sx = e.touches[0].clientX; }, { passive: true });
  vp.addEventListener("touchend", function(e) {
    var dx = e.changedTouches[0].clientX - sx;
    if (Math.abs(dx) < 40) return;
    var cnt = gridMode === "cols1" ? 1 : gridMode === "cols2" ? 2 : 4;
    if (dx < 0) currentMobileIdx = Math.min(currentMobileIdx + 1, cnt - 1);
    else currentMobileIdx = Math.max(currentMobileIdx - 1, 0);
    updateVisible();
  });
  window.addEventListener("resize", updateVisible);
}

function updateVisible() {
  var isMobile = window.innerWidth <= 768;
  var cells = document.querySelectorAll(".stream-cell");
  var cnt = gridMode === "cols1" ? 1 : gridMode === "cols2" ? 2 : 4;
  for (var i = 0; i < cells.length; i++) {
    if (isMobile) cells[i].style.display = (i === currentMobileIdx) ? "" : "none";
    else cells[i].style.display = i < cnt ? "" : "none";
  }
  throttleStreams();
}

export function throttleStreams() {
  var isMobile = window.innerWidth <= 768;
  var imgs = document.querySelectorAll(".stream-cell img");
  for (var i = 0; i < imgs.length; i++) {
    var shouldSuspend;
    if (isMobile) shouldSuspend = i !== currentMobileIdx;
    else shouldSuspend = i >= 2;
    var img = imgs[i];
    if (shouldSuspend) {
      if (img.dataset.srcSaved === undefined && img.src && img.src !== "") {
        img.dataset.srcSaved = img.src;
        img.src = "";
      }
    } else {
      if (img.dataset.srcSaved) {
        img.src = img.dataset.srcSaved;
        delete img.dataset.srcSaved;
      }
    }
  }
}
