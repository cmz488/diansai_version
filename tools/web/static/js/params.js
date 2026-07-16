/** Parameter panel — renders sliders/toggles/dropdowns, POSTs changes. */

import { $, $$, debounce } from "./utils.js";

var paramDefs = [];
var paramValues = {};

var POST = debounce(async function(name, value) {
  await fetch("/api/params", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: name, value: value }),
  });
}, 50);

export async function loadParams() {
  var r = await fetch("/api/params");
  paramDefs = await r.json();
  for (var i = 0; i < paramDefs.length; i++) {
    paramValues[paramDefs[i].name] = paramDefs[i].value;
  }
  render();
}

export function handleParamsMessage(data) {
  var incoming = data.params;
  for (var k in incoming) {
    if (incoming.hasOwnProperty(k)) paramValues[k] = incoming[k];
  }
  renderValues();
}

function render() {
  renderInto($("#paramList"));
  renderInto($("#mobileParamList"));
}

function renderInto(container) {
  if (!container) return;

  var groups = {};
  for (var i = 0; i < paramDefs.length; i++) {
    var p = paramDefs[i];
    groups[p.group] = groups[p.group] || [];
    groups[p.group].push(p);
  }

  var html = "";
  for (var group in groups) {
    if (!groups.hasOwnProperty(group)) continue;
    var params = groups[group];
    html += '<div class="sidebar-section">';
    html += '<div class="label">' + group + '</div>';
    html += '<div class="content">';
    for (var j = 0; j < params.length; j++) {
      var p2 = params[j];
      var val = paramValues[p2.name] != null ? paramValues[p2.name] : p2.default;
      html += renderParamRow(p2, val);
    }
    html += '</div></div>';
  }
  container.innerHTML = html;

  // Bind events
  for (var m = 0; m < paramDefs.length; m++) {
    var p3 = paramDefs[m];
    var el = document.querySelector('[data-param="' + p3.name + '"]');
    if (!el) continue;
    (function(p, elem) {
      elem.addEventListener("input", function() {
        var v = p.type === "bool" ? this.checked : this.value;
        paramValues[p.name] = p.type === "int" ? parseInt(v)
          : p.type === "float" ? parseFloat(v) : v;
        var label = document.querySelector('[data-param-val="' + p.name + '"]');
        if (label) label.textContent = paramValues[p.name];
        POST(p.name, paramValues[p.name]);
      });
    })(p3, el);
  }
}

function renderParamRow(p, val) {
  if (p.type === "bool") {
    return '<div class="toggle-row">' +
      '<span>' + p.name + '</span>' +
      '<input type="checkbox" data-param="' + p.name + '" ' + (val ? "checked" : "") + ' />' +
      '</div>';
  }
  if (p.type === "choice") {
    var opts = "";
    var choices = p.choices || [];
    for (var i = 0; i < choices.length; i++) {
      var c = choices[i];
      opts += '<option value="' + c + '"' + (c === val ? " selected" : "") + '>' + c + '</option>';
    }
    return '<div class="param-row">' +
      '<div class="param-label"><span>' + p.name + '</span></div>' +
      '<select data-param="' + p.name + '">' + opts + '</select>' +
      '</div>';
  }
  // int / float slider
  var range = p.range || [0, 100];
  var lo = range[0], hi = range[1];
  var step = p.step != null ? p.step : (p.type === "float" ? 0.1 : 1);
  return '<div class="param-row">' +
    '<div class="param-label">' +
    '<span>' + p.name + '</span>' +
    '<span class="val" data-param-val="' + p.name + '">' + val + '</span>' +
    '</div>' +
    '<input type="range" data-param="' + p.name + '"' +
    ' min="' + lo + '" max="' + hi + '" step="' + step + '" value="' + val + '" />' +
    '</div>';
}

function renderValues() {
  for (var i = 0; i < paramDefs.length; i++) {
    var p = paramDefs[i];
    var el = document.querySelector('[data-param="' + p.name + '"]');
    var label = document.querySelector('[data-param-val="' + p.name + '"]');
    if (!el) continue;
    var val = paramValues[p.name] != null ? paramValues[p.name] : p.default;
    if (p.type === "bool") el.checked = !!val;
    else if (p.type !== "choice") {
      el.value = val;
      if (label) label.textContent = val;
    }
  }
}
