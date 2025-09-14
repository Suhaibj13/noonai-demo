/* static/app.js
   Noon AI – chat UI with Analysis mode
   - Sends { q, mode, datasets, fileHints } to /ask
   - Renders "insights" (KPIs + bullets + optional tables)
   - Falls back to your existing table/text rendering for non-insights
*/

/* ---------- Configurable selectors (adjust if your HTML IDs differ) ---------- */
const EL = {
  messages: document.getElementById("messages"),
  input: document.getElementById("userInput"),
  send: document.getElementById("sendBtn"),
  chipWeb: document.getElementById("chip-web"),
  chipData: document.getElementById("chip-data"),
  chipAnalysis: document.getElementById("chip-analysis"),
  datasetPicker: document.getElementById("dataset-picker") // optional <select multiple>
};

/* ---------- Mode / dataset state ---------- */
let currentMode = "web";          // "web" | "data" | "analysis"
let currentDatasets = [];         // e.g. ["orders"] or ["orders","inventory"]
let fileHints = [];               // optional explicit filenames/paths

/* ---------- Utilities ---------- */
function elSafe(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;");
}

function addMsg(role, html) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  wrapper.innerHTML = html;
  EL.messages.appendChild(wrapper);
  EL.messages.scrollTop = EL.messages.scrollHeight;
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

/* ---------- Insights renderer (safe APPEND) ---------- */
function tryRenderInsights(resp) {
  try {
    if (!resp || resp.type !== "insights") return false;

    const kpis = resp.kpis || {};
    const insights = resp.insights || [];
    const redFlags = resp.red_flags || [];
    const breakdowns = resp.breakdowns || {};
    const tables = resp.tables || {};

    const kpiHtml = Object.entries(kpis).map(
      ([k,v]) => `
        <div class="kpi">
          <div class="kpi-key">${elSafe(k.replaceAll('_',' '))}</div>
          <div class="kpi-val">${(v ?? '-')}</div>
        </div>`
    ).join("");

    const li = (arr) =>
      (arr && arr.length)
        ? `<ul class="insight-bullets">${arr.map(x=>`<li>${elSafe(x)}</li>`).join("")}</ul>`
        : "";

    const byList = (title, list) =>
      (list && list.length)
        ? `<div class="insight-subtitle">${elSafe(title)}</div>${li(list.map(x => {
            if (typeof x === "string") return x;
            if (x && typeof x === "object" && "key" in x)
              return `${x.key} — ${(x.payout||0).toLocaleString()}`;
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

    addMsg("assistant", `
      <div class="insight-kpis">${kpiHtml}</div>
      ${li(insights)}
      ${redFlags?.length ? `<div class="insight-subtitle">Red flags</div>${li(redFlags)}` : ""}
      ${byList("By city", breakdowns.by_city)}
      ${byList("By vendor", breakdowns.by_vendor)}
      ${tableTopRisk}
    `);

    return true;
  } catch (e) {
    console.error("insights render error", e);
    return false;
  }
}

/* ---------- Generic table/text renderer (kept for web/data) ---------- */
function renderTableOrText(resp) {
  // If backend returns html, string, array-of-obj, or obj-of-obj — handle broadly
  if (!resp) { addMsg("assistant", "<em>No response</em>"); return; }

  // String → show as text
  if (typeof resp === "string") { addMsg("assistant", `<p>${elSafe(resp)}</p>`); return; }

  // If it looks like a plain table: [{...}, {...}]
  if (Array.isArray(resp) && resp.length && typeof resp[0] === "object" && !Array.isArray(resp[0])) {
    const cols = Object.keys(resp[0]);
    const thead = `<thead><tr>${cols.map(c=>`<th>${elSafe(c)}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${resp.map(row => `<tr>${cols.map(c => `<td>${elSafe(row[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;
    addMsg("assistant", `<div class="table-scroll"><table class="mini">${thead}${tbody}</table></div>`);
    return;
  }

  // If it's an object with a "data" key or anything else → pretty JSON
  addMsg("assistant", `<pre class="json">${elSafe(JSON.stringify(resp, null, 2))}</pre>`);
}

/* ---------- Send flow ---------- */
async function sendQuery(userText) {
  if (!userText || !userText.trim()) return;
  addMsg("user", `<p>${elSafe(userText)}</p>`);
  const typingId = addTyping();

  // Gather datasets from optional <select multiple> if present
  if (EL.datasetPicker) {
    const opts = Array.from(EL.datasetPicker.selectedOptions).map(o => o.value);
    if (opts.length) currentDatasets = opts;
  }

  const payload = {
    q: userText.trim(),
    mode: currentMode,              // "web" | "data" | "analysis"
    datasets: currentDatasets,      // [] is fine — backend auto-detects
    fileHints                         // leave [] unless you want to force files
  };

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const resp = await res.json();
    removeTyping(typingId);

    // Analysis insights first; if not, fall back to table/text
    if (tryRenderInsights(resp)) return;
    renderTableOrText(resp);

  } catch (err) {
    console.error(err);
    removeTyping(typingId);
    addMsg("assistant", `<p style="color:#b00020">Error: ${elSafe(err.message || err)}</p>`);
  }
}

/* ---------- Wire UI ---------- */
function init() {
  // Send button / Enter key
  EL.send?.addEventListener("click", () => {
    const t = EL.input.value;
    EL.input.value = "";
    sendQuery(t);
  });
  EL.input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const t = EL.input.value;
      EL.input.value = "";
      sendQuery(t);
    }
  });

  // Mode chips (safe APPEND)
  EL.chipWeb?.addEventListener("click", () => { currentMode = "web";  uiActive("web"); });
  EL.chipData?.addEventListener("click", () => { currentMode = "data"; uiActive("data"); });
  EL.chipAnalysis?.addEventListener("click", () => { currentMode = "analysis"; uiActive("analysis"); });

  // (Optional) set default dataset from picker on load
  if (EL.datasetPicker) {
    const opts = Array.from(EL.datasetPicker.selectedOptions).map(o => o.value);
    currentDatasets = opts;
  }
}

// simple visual active state helper (no-op if you don’t have these classes)
function uiActive(which) {
  [EL.chipWeb, EL.chipData, EL.chipAnalysis].forEach(btn => btn?.classList.remove("active"));
  if (which === "web") EL.chipWeb?.classList.add("active");
  if (which === "data") EL.chipData?.classList.add("active");
  if (which === "analysis") EL.chipAnalysis?.classList.add("active");
}

/* ---------- Kick off ---------- */
init();
