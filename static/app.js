/* static/app.js — Noon AI resilient UI
   - Defensive selectors, event delegation (no fragile IDs)
   - Works with "web" | "data" | "analysis" modes
   - Renders "insights" (KPIs/bullets/tables) when backend returns { type:"insights", ... }
*/

/* =============== Helpers =============== */
function onceReady(fn) {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
  else fn();
}

function $(sel, root = document) { return root.querySelector(sel); }
function $all(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }
function elSafe(str) {
  if (str === null || str === undefined) return "";
  return String(str).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

// Try multiple selectors; return the first that exists
function pick(...selectors) {
  for (const s of selectors) {
    const el = $(s);
    if (el) return el;
  }
  return null;
}

/* =============== Global state =============== */
let currentMode = "web";           // "web" | "data" | "analysis"
let currentDatasets = [];          // e.g. ["orders"], ["orders","inventory"]
let fileHints = [];                // optional: filenames/paths

function getMessagesEl() {
  return pick("#messages", ".messages", "#chat-messages", "[data-messages]");
}

function getInputEl() {
  return pick("#userInput", "#input", "#composer-input", 'textarea[name="message"]', "textarea", 'input[type="text"]');
}

/* =============== Rendering =============== */
function tryRenderInsights(resp) {
  try {
    if (!resp || resp.type !== "insights") return false;

    const wrap = document.createElement("div");
    wrap.className = "msg assistant";

    const kpis = resp.kpis || {};
    const insights = resp.insights || [];
    const redFlags = resp.red_flags || [];
    const breakdowns = resp.breakdowns || {};
    const tables = resp.tables || {};

    const kpiHtml = Object.entries(kpis).map(
      ([k,v]) => `
        <div class="kpi">
          <div class="kpi-key">${elSafe(k.replaceAll('_',' '))}</div>
          <div class="kpi-val">${(v ?? '-') }</div>
        </div>`
    ).join("");

    const li = (arr) =>
      (arr && arr.length)
        ? `<ul class="insight-bullets">${arr.map(x => `<li>${elSafe(x)}</li>`).join("")}</ul>`
        : "";

    const byList = (title, list) =>
      (list && list.length)
        ? `<div class="insight-subtitle">${elSafe(title)}</div>${li(list.map(x => {
            if (typeof x === "string") return x;
            if (x && typeof x === "object" && "key" in x)
              return `${x.key} — ${(x.payout || 0).toLocaleString()}`;
            return JSON.stringify(x);
          }))}`
        : "";

    const tableTopRisk = (tables.top_risk && tables.top_risk.length)
      ? (() => {
          const cols = Object.keys(tables.top_risk[0]);
          return `
            <div class="insight-subtitle">Top at-risk products</div>
            <div class="table-scroll">
              <table class="mini">
                <thead><tr>${cols.map(c=>`<th>${elSafe(c)}</th>`).join("")}</tr></thead>
                <tbody>
                  ${tables.top_risk.map(row => `
                    <tr>${cols.map(c => `<td>${elSafe(row[c])}</td>`).join("")}</tr>
                  `).join("")}
                </tbody>
              </table>
            </div>`;
        })()
      : "";

    wrap.innerHTML = `
      <div class="insight-kpis">${kpiHtml}</div>
      ${li(insights)}
      ${redFlags?.length ? `<div class="insight-subtitle">Red flags</div>${li(redFlags)}` : ""}
      ${byList("By city", breakdowns.by_city)}
      ${byList("By vendor", breakdowns.by_vendor)}
      ${tableTopRisk}
    `;

    const msgEl = getMessagesEl() || document.body;
    msgEl.appendChild(wrap);
    msgEl.scrollTop = msgEl.scrollHeight;

    return true;
  } catch (e) {
    console.error("insights render error", e);
    return false;
  }
}

function renderTableOrText(resp) {
  const msgEl = getMessagesEl() || document.body;
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";

  if (!resp) { wrap.innerHTML = "<em>No response</em>"; msgEl.appendChild(wrap); return; }

  if (typeof resp === "string") {
    wrap.innerHTML = `<p>${elSafe(resp)}</p>`;
    msgEl.appendChild(wrap); return;
  }

  if (Array.isArray(resp) && resp.length && typeof resp[0] === "object" && !Array.isArray(resp[0])) {
    const cols = Object.keys(resp[0]);
    const thead = `<thead><tr>${cols.map(c=>`<th>${elSafe(c)}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${resp.map(row => `<tr>${cols.map(c => `<td>${elSafe(row[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;
    wrap.innerHTML = `<div class="table-scroll"><table class="mini">${thead}${tbody}</table></div>`;
    msgEl.appendChild(wrap); msgEl.scrollTop = msgEl.scrollHeight; return;
  }

  wrap.innerHTML = `<pre class="json">${elSafe(JSON.stringify(resp, null, 2))}</pre>`;
  msgEl.appendChild(wrap); msgEl.scrollTop = msgEl.scrollHeight;
}

function addMsg(role, html) {
  const msgEl = getMessagesEl() || document.body;
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = html;
  msgEl.appendChild(wrap);
  msgEl.scrollTop = msgEl.scrollHeight;
}

function addTyping() {
  const id = `typing-${Date.now()}`;
  addMsg("assistant", `<span id="${id}" class="typing">...</span>`);
  return id;
}
function removeTyping(id) {
  const t = document.getElementById(id);
  if (t && t.parentNode) t.parentNode.remove();
}

/* =============== Send flow =============== */
async function sendQuery(userText) {
  const inputEl = getInputEl();
  const text = (userText ?? (inputEl ? inputEl.value : "")).trim();
  if (!text) return;

  addMsg("user", `<p>${elSafe(text)}</p>`);
  if (inputEl) inputEl.value = "";
  const typingId = addTyping();

  // dataset-picker (optional <select multiple>) → datasets
  const picker = pick("#dataset-picker", "[data-datasets]");
  if (picker && picker.selectedOptions) {
    const opts = Array.from(picker.selectedOptions).map(o => o.value);
    if (opts.length) currentDatasets = opts;
  }

  const payload = {
    q: text,
    mode: currentMode,
    datasets: currentDatasets,
    fileHints
  };

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload)
    });
    const resp = await res.json();
    removeTyping(typingId);

    if (tryRenderInsights(resp)) return;
    renderTableOrText(resp);
  } catch (err) {
    console.error(err);
    removeTyping(typingId);
    addMsg("assistant", `<p style="color:#b00020">Error: ${elSafe(err.message || err)}</p>`);
  }
}

/* =============== Event delegation (buttons, chips, presets) =============== */
function setModeFromEl(el) {
  // priority: data-mode attribute, then ID/class mapping
  const dm = el.getAttribute?.("data-mode");
  if (dm) return dm.toLowerCase();

  const id = (el.id || "").toLowerCase();
  if (id.includes("analysis")) return "analysis";
  if (id.includes("data")) return "data";
  if (id.includes("web")) return "web";

  const cls = (el.className || "").toLowerCase();
  if (cls.includes("analysis")) return "analysis";
  if (cls.includes("data")) return "data";
  if (cls.includes("web")) return "web";

  return currentMode; // unchanged if unknown
}

function uiActive(which) {
  const candidates = [
    "#chip-web", "#chip-data", "#chip-analysis",
    "[data-mode='web']", "[data-mode='data']", "[data-mode='analysis']",
    ".chip-web", ".chip-data", ".chip-analysis"
  ];
  $all(candidates.join(",")).forEach(btn => btn.classList?.remove?.("active"));

  const targetSel = (which === "web") ? ["#chip-web", "[data-mode='web']", ".chip-web"]
                   : (which === "data") ? ["#chip-data", "[data-mode='data']", ".chip-data"]
                   : ["#chip-analysis", "[data-mode='analysis']", ".chip-analysis"];

  for (const s of targetSel) {
    const el = $(s);
    if (el) { el.classList?.add?.("active"); break; }
  }
}

function onClick(e) {
  const t = e.target;

  // Send
  const sendBtn = t.closest?.('#sendBtn, #send, #send-button, .send, [data-action="send"]');
  if (sendBtn) { e.preventDefault(); sendQuery(); return; }

  // Reset (clear messages)
  const resetBtn = t.closest?.('#resetBtn, #reset, .reset, [data-action="reset"]');
  if (resetBtn) {
    e.preventDefault();
    const msgEl = getMessagesEl();
    if (msgEl) msgEl.innerHTML = "";
    return;
  }

  // Modes
  const modeBtn = t.closest?.('[data-mode], #chip-web, #chip-data, #chip-analysis, .chip-web, .chip-data, .chip-analysis');
  if (modeBtn) {
    e.preventDefault();
    currentMode = setModeFromEl(modeBtn);
    uiActive(currentMode);
    return;
  }

  // Presets / Quick chips → send their query
  const preset = t.closest?.('[data-q], [data-preset], .preset, .quick-chip');
  if (preset) {
    e.preventDefault();
    const q = preset.getAttribute("data-q") || preset.getAttribute("data-preset") || preset.textContent.trim();
    if (q) sendQuery(q);
    return;
  }

  // Audit steps (same as presets; support data-step/data-q)
  const step = t.closest?.('[data-step], .audit-step');
  if (step) {
    e.preventDefault();
    const q = step.getAttribute("data-q") || step.getAttribute("data-step") || step.textContent.trim();
    if (q) sendQuery(q);
    return;
  }
}

function onKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    const inputEl = getInputEl();
    if (inputEl && document.activeElement === inputEl) {
      e.preventDefault();
      sendQuery();
    }
  }
}

/* =============== Boot =============== */
onceReady(() => {
  // Attach one-time delegated listeners
  document.addEventListener("click", onClick);
  document.addEventListener("keydown", onKeydown);

  // If there’s a default selected mode element, pick it; else default "web"
  const active = $('[data-mode].active, .chip-web.active, .chip-data.active, .chip-analysis.active');
  if (active) currentMode = setModeFromEl(active);

  // (Optional) initialize datasets from a multi-select
  const picker = pick("#dataset-picker", "[data-datasets]");
  if (picker && picker.selectedOptions) {
    currentDatasets = Array.from(picker.selectedOptions).map(o => o.value);
  }

  // Simple sanity log to spot ID mismatches quickly
  console.log("[Noon AI] UI wired. mode=%s datasets=%o", currentMode, currentDatasets);
});

/* =============== Minimal CSS helpers (safe) =============== */
/* Keep these in CSS file ideally; left here in case. */
