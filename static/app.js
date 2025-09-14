/* static/app.js — Noon AI resilient UI (stable baseline) */

function onceReady(fn){ if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",fn); else fn(); }
function $(s,root=document){ return root.querySelector(s); }
function $all(s,root=document){ return Array.from(root.querySelectorAll(s)); }
function elSafe(v){ if(v===null||v===undefined)return ""; return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }

let currentMode = "web";           // "web" | "data" | "analysis"
let currentDatasets = [];          // e.g. ["orders"] or ["orders","inventory"]
let fileHints = [];                // optional hints

function msgs(){ return $("#messages") || $(".messages") || $("#chat-messages") || document.body; }
function composer(){ return $("#userInput") || $("#input") || $("#composer-input") || $('textarea[name="message"]') || $("textarea") || $('input[type="text"]'); }

function addMsg(role, html){
  const w = document.createElement("div");
  w.className = `msg ${role}`;
  w.innerHTML = html;
  msgs().appendChild(w);
  msgs().scrollTop = msgs().scrollHeight;
}
function addTyping(){ const id=`typing-${Date.now()}`; addMsg("assistant", `<span id="${id}" class="typing">...</span>`); return id; }
function removeTyping(id){ const t = document.getElementById(id); if(t&&t.parentNode) t.parentNode.remove(); }

/* -------- Analysis renderer -------- */
function tryRenderInsights(resp){
  try{
    if(!resp || resp.type!=="insights") return false;

    const kpis = resp.kpis || {};
    const insights = resp.insights || [];
    const redFlags = resp.red_flags || [];
    const breakdowns = resp.breakdowns || {};
    const tables = resp.tables || {};

    const kpiHtml = Object.entries(kpis).map(([k,v]) => `
      <div class="kpi">
        <div class="kpi-key">${elSafe(k.replaceAll('_',' '))}</div>
        <div class="kpi-val">${(v ?? '-')}</div>
      </div>`).join("");

    const li = arr => (arr && arr.length) ? `<ul class="insight-bullets">${arr.map(x=>`<li>${elSafe(x)}</li>`).join("")}</ul>` : "";

    const byList = (title, list) => (list && list.length)
      ? `<div class="insight-subtitle">${elSafe(title)}</div>${li(list.map(x=>{
          if(typeof x==="string") return x;
          if(x && typeof x==="object" && "key" in x) return `${x.key} — ${(x.payout||0).toLocaleString()}`;
          return JSON.stringify(x);
        }))}`
      : "";

    const tableTopRisk = (tables.top_risk && tables.top_risk.length) ? (() => {
      const cols = Object.keys(tables.top_risk[0]);
      return `
        <div class="insight-subtitle">Top at-risk products</div>
        <div class="table-scroll">
          <table class="mini">
            <thead><tr>${cols.map(c=>`<th>${elSafe(c)}</th>`).join("")}</tr></thead>
            <tbody>${tables.top_risk.map(r=>`<tr>${cols.map(c=>`<td>${elSafe(r[c])}</td>`).join("")}</tr>`).join("")}</tbody>
          </table>
        </div>`;
    })() : "";

    addMsg("assistant", `
      <div class="insight-kpis">${kpiHtml}</div>
      ${li(insights)}
      ${redFlags?.length ? `<div class="insight-subtitle">Red flags</div>${li(redFlags)}` : ""}
      ${byList("By city", breakdowns.by_city)}
      ${byList("By vendor", breakdowns.by_vendor)}
      ${tableTopRisk}
    `);
    return true;
  }catch(e){ console.error("insights render error", e); return false; }
}

/* -------- Fallback renderer for data/web -------- */
function renderTableOrText(resp){
  // If server replied with an object containing a "reply" string, show it as text (Web mode).
  if(resp && typeof resp==="object" && typeof resp.reply==="string" && !resp.type){
    addMsg("assistant", `<p>${elSafe(resp.reply)}</p>`);
    return;
  }

  if(!resp){ addMsg("assistant","<em>No response</em>"); return; }

  if(typeof resp==="string"){ addMsg("assistant", `<p>${elSafe(resp)}</p>`); return; }

  if(Array.isArray(resp) && resp.length && typeof resp[0]==="object" && !Array.isArray(resp[0])){
    const cols = Object.keys(resp[0]);
    const thead = `<thead><tr>${cols.map(c=>`<th>${elSafe(c)}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${resp.map(row=>`<tr>${cols.map(c=>`<td>${elSafe(row[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;
    addMsg("assistant", `<div class="table-scroll"><table class="mini">${thead}${tbody}</table></div>`);
    return;
  }

  // Any other object → pretty JSON (debug-friendly)
  addMsg("assistant", `<pre class="json">${elSafe(JSON.stringify(resp, null, 2))}</pre>`);
}

/* -------- Send -------- */
async function sendQuery(userText){
  const input = composer();
  const text = (userText ?? (input ? input.value : "")).trim();
  if(!text) return;

  addMsg("user", `<p>${elSafe(text)}</p>`);
  if(input) input.value = "";
  const typingId = addTyping();

  // optional dataset picker
  const picker = $("#dataset-picker") || $("[data-datasets]");
  if(picker && picker.selectedOptions){
    const vals = Array.from(picker.selectedOptions).map(o => o.value);
    if(vals.length) currentDatasets = vals;
  }

  const payload = { q: text, mode: currentMode, datasets: currentDatasets, fileHints };

  try{
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload)
    });
    const resp = await res.json();
    removeTyping(typingId);

    if(tryRenderInsights(resp)) return;
    renderTableOrText(resp);
  }catch(err){
    console.error(err);
    removeTyping(typingId);
    addMsg("assistant", `<p style="color:#b00020">Error: ${elSafe(err.message||err)}</p>`);
  }
}

/* -------- Delegated UI wiring -------- */
function uiActive(which){
  const all = [
    "#chip-web", "#chip-data", "#chip-analysis",
    "[data-mode='web']", "[data-mode='data']", "[data-mode='analysis']",
    ".chip-web", ".chip-data", ".chip-analysis"
  ].join(",");
  $all(all).forEach(b=>b.classList?.remove?.("active"));

  const pick = which==="web" ? ["#chip-web","[data-mode='web']", ".chip-web"]
            : which==="data" ? ["#chip-data","[data-mode='data']", ".chip-data"]
            : ["#chip-analysis","[data-mode='analysis']", ".chip-analysis"];
  for(const s of pick){ const el=$(s); if(el){ el.classList?.add?.("active"); break; } }
}

function modeFromEl(el){
  const dm = el.getAttribute?.("data-mode"); if(dm) return dm.toLowerCase();
  const id = (el.id||"").toLowerCase();
  const cl = (el.className||"").toLowerCase();
  if(id.includes("analysis") || cl.includes("analysis")) return "analysis";
  if(id.includes("data") || cl.includes("data")) return "data";
  if(id.includes("web") || cl.includes("web")) return "web";
  return currentMode;
}

function onClick(e){
  const t = e.target;

  // Send
  if(t.closest?.('#sendBtn, #send, #send-button, .send, [data-action="send"]')){
    e.preventDefault(); sendQuery(); return;
  }

  // Reset
  if(t.closest?.('#resetBtn, #reset, .reset, [data-action="reset"]')){
    e.preventDefault(); const m = msgs(); if(m) m.innerHTML = ""; return;
  }

  // Modes
  const modeBtn = t.closest?.('[data-mode], #chip-web, #chip-data, #chip-analysis, .chip-web, .chip-data, .chip-analysis');
  if(modeBtn){
    e.preventDefault(); currentMode = modeFromEl(modeBtn); uiActive(currentMode); return;
  }

  // Presets / Audit steps
  const preset = t.closest?.('[data-q], [data-preset], .preset, .quick-chip, .audit-step, [data-step]');
  if(preset){
    e.preventDefault();
    const q = preset.getAttribute("data-q") || preset.getAttribute("data-preset") || preset.getAttribute("data-step") || preset.textContent.trim();
    if(q) sendQuery(q);
  }
}

function onKeydown(e){
  if(e.key === "Enter" && !e.shiftKey){
    const input = composer();
    if(input && document.activeElement === input){
      e.preventDefault(); sendQuery();
    }
  }
}

onceReady(()=>{
  document.addEventListener("click", onClick);
  document.addEventListener("keydown", onKeydown);
  console.log("[Noon AI] UI wired.");
});
