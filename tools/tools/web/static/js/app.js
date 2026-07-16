/** Main application controller — WebSocket, tabs, orchestration. */

import { $ } from "./utils.js";
import { initLog, handleLogMessage, refreshLog } from "./log.js";
import { initChart } from "./chart.js";
import { loadParams, handleParamsMessage } from "./params.js";
import { setGrid, initMobileSwipe, throttleStreams } from "./layout.js";
import { initROI, toggleROI } from "./roi.js";

var ws;
var chart;

function connectWS() {
  var proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");

  ws.onmessage = function(e) {
    var msg = JSON.parse(e.data);
    switch (msg.type) {
      case "log":     handleLogMessage(msg); break;
      case "params":  handleParamsMessage(msg); break;
      case "metrics": updateMetrics(msg); break;
    }
  };

  ws.onclose = function() { setTimeout(connectWS, 2000); };
}

function updateMetrics(data) {
  var fps = data.fps, detect = data.detect, latency = data.latency;
  if (fps !== undefined) {
    var el = $("#metricFps"); if (el) el.textContent = fps;
    if (chart) chart.push("fps", fps);
  }
  if (detect !== undefined) {
    var el2 = $("#metricDetect"); if (el2) el2.textContent = detect;
    if (chart) chart.push("detect", detect);
  }
  if (latency !== undefined) {
    var el3 = $("#metricLatency"); if (el3) el3.textContent = latency.toFixed(1) + "ms";
  }
}

function safeOn(el, event, fn) {
  if (el) el.addEventListener(event, fn);
}

function initTabs() {
  var tabs = document.querySelectorAll(".tabbar button");
  for (var i = 0; i < tabs.length; i++) {
    (function(btn) {
      btn.addEventListener("click", function() {
        for (var j = 0; j < tabs.length; j++) tabs[j].classList.remove("active");
        btn.classList.add("active");
        showMobilePanel(btn.dataset.panel);
      });
    })(tabs[i]);
  }

  safeOn($("#btnChart"), "click", function() { toggleBottomPanel("chart"); });
  safeOn($("#btnLog"), "click", function() { toggleBottomPanel("log"); });
}

function showMobilePanel(name) {
  var panels = document.querySelectorAll(".mobile-panel");
  for (var i = 0; i < panels.length; i++) panels[i].classList.remove("show");
  if (name === "view") return; // "view" means show the stream grid, hide all panels
  var panel = $("#mobile-" + name);
  if (panel) {
    panel.classList.add("show");
    if (name === "log") refreshLog();
  }
}

function toggleBottomPanel(name) {
  var el = $(".panel-" + name);
  if (!el) return;
  var show = !el.classList.contains("show");
  if (show) { el.classList.add("show"); }
  else { el.classList.remove("show"); }
  var btn = $("#btn" + name.charAt(0).toUpperCase() + name.slice(1));
  if (btn) {
    if (show) btn.classList.add("active");
    else btn.classList.remove("active");
  }
  if (name === "log" && show) refreshLog();
  if (name === "chart" && show && chart) chart.resize();
}

function wireButtons() {
  safeOn($("#btnGrid1"), "click", function() { setGrid("cols1"); });
  safeOn($("#btnGrid2"), "click", function() { setGrid("cols2"); });
  safeOn($("#btnGrid4"), "click", function() { setGrid("cols4"); });
  safeOn($("#btnROI"), "click", toggleROI);
  safeOn($("#btnSnapshot"), "click", function() {
    fetch("/api/snapshot", { method: "POST" })
      .then(function(r) { return r.json(); })
      .then(function(d) { if (d.success) console.log("快照已保存:", d.filename); })
      .catch(function() {});
  });
  safeOn($("#btnRecord"), "click", function() {
    var btn = this;
    var recording = btn.dataset.recording === "true";
    fetch("/api/recording", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: recording ? "stop" : "start" }),
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.success) {
        btn.dataset.recording = recording ? "false" : "true";
        btn.textContent = recording ? "⏺ 录屏" : "⏹ 停止";
        if (!recording) btn.classList.add("danger");
        else btn.classList.remove("danger");
        var dot = document.querySelector(".rec-dot");
        if (dot) {
          if (!recording) dot.classList.add("active");
          else dot.classList.remove("active");
        }
      }
    })
    .catch(function() {});
  });

  var closeBtns = document.querySelectorAll(".mobile-panel .close-btn");
  for (var i = 0; i < closeBtns.length; i++) {
    (function(btn2) {
      btn2.addEventListener("click", function() {
        btn2.parentElement.classList.remove("show");
      });
    })(closeBtns[i]);
  }
}

async function init() {
  connectWS();
  try {
    await loadParams();
  } catch (e) {
    console.warn("加载参数失败，使用默认值:", e);
  }
  initLog(ws);
  chart = initChart("chartCanvas");
  if (chart) {
    chart.addSeries("fps", "#0071e3");
    chart.addSeries("detect", "#34c759");
  }
  initROI("roiCanvas");
  initMobileSwipe();
  initTabs();
  setGrid("cols2");
  setInterval(throttleStreams, 5000);
  wireButtons();
}

// Module scripts are deferred — DOMContentLoaded may have already fired.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
