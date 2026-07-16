/** Virtual-scrolling log panel connected to WebSocket. */

import { $, fmtTime } from "./utils.js";

var MAX_LINES = 500;
var VISIBLE = 20;
var entries = [];
var ws = null;

export function initLog(wsInstance) {
  ws = wsInstance;
}

export function handleLogMessage(data) {
  entries.push(data);
  if (entries.length > MAX_LINES) entries = entries.slice(-MAX_LINES);
  render();
}

function render() {
  renderInto($("#logList"));
  renderInto($("#mobileLogList"));
}

function renderInto(list) {
  if (!list) return;
  // On desktop, only render if panel is visible
  if (list.id === "logList") {
    var parent = list.parentElement;
    if (parent && !parent.classList.contains("show")) return;
  }

  var start = Math.max(0, entries.length - VISIBLE);
  var html = "";
  for (var i = start; i < entries.length; i++) {
    var e = entries[i];
    var cls = "log-row log-" + e.level;
    html += '<div class="' + cls + '">' +
      '<span class="log-ts">' + fmtTime(e.ts) + '</span>' +
      '<span class="log-tag">[' + esc(e.tag) + ']</span>' +
      '<span class="log-msg">' + esc(e.msg) + '</span>' +
      '</div>';
  }
  list.innerHTML = html;
  list.scrollTop = list.scrollHeight;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Force re-render (called when panel becomes visible or on resize). */
export function refreshLog() { render(); }
