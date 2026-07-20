/** Main application controller — WebSocket, stream layout, params. */

import { $ } from "./utils.js";
import { loadParams, handleParamsMessage } from "./params.js";
import { setGrid, throttleStreams } from "./layout.js";

let ws;

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");

  ws.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case "params":  handleParamsMessage(msg); break;
      case "metrics": updateMetrics(msg);       break;
    }
  };

  ws.onclose = function() { setTimeout(connectWS, 2000); };
}

function updateMetrics(data) {
  if (data.fps !== undefined) {
    const el = $("#metricFps");
    if (el) el.textContent = data.fps;
  }
  if (data.detect !== undefined) {
    const el = $("#metricDetect");
    if (el) el.textContent = data.detect;
  }
}

function wireButtons() {
  const on = (id, ev, fn) => {
    const el = $(id);
    if (el) el.addEventListener(ev, fn);
  };

  on("#btnGrid1", "click", () => setGrid("cols1"));
  on("#btnGrid2", "click", () => setGrid("cols2"));
  on("#btnGrid4", "click", () => setGrid("cols4"));
}

async function init() {
  connectWS();
  try { await loadParams(); } catch (e) { console.warn("加载参数失败:", e); }
  setGrid("cols2");
  setInterval(throttleStreams, 5000);
  wireButtons();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
