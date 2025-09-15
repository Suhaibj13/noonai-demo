/* static/app.js — chat UI + Data→Extract schema viewer
   - Keeps existing Web/Data/Analysis send flow
   - Adds dropdown (#data-selector) + button (#extractBtn) to call /schema
*/

(function () {
  // ---------- tiny helpers ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const safe = (s) => (s == null ? "" : String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;"));

  // ---------- DOM references (match your current ids) ----------
  const el = {
    messages: $("#messages") || $(".messages") || $("#chat-messages") || document.body,
    input: $("#userInput") || $("#input") || $("#composer-input") || $("textarea") || $('input[type="text"]'),
    send: $("#sendBtn") || $("#send") || $("[data-action='send']"),
    reset: $("#reset") || $("#resetBtn") || $("[data-action='reset']"),
    chipWeb: $("#chip-web"),
    chipData: $("#chip-data"),
    chipAnalysis: $("#chip-analysis"),
    dataSelector: $("#data-selector"),
    extractBtn: $("#extractBtn")
  };

  // ---------- state ----------
  let currentMode = "web"; // "web" | "data" | "analysis"

  // ---------- rendering ----------
  function addMsg(role, html) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    wrap.innerHTML = html;
    el.messages.appendChild(wrap);
    el.messages.scrollTop = el.messages.scrollHeight;
  }
  function typingOn() {
    const id = `typing-${Date.now()}`;
    addMsg("assistant", `<span id="${id}" class="typing">...</span>`);
    return id;
  }
  function typingOff(id) {
    const t = document.getElementById(id);
    if (t && t.parentNode) t.parentNode.remove();
  }

  function renderResponse(resp) {
    // analysis/data/web responses all end up here
    if (!resp) { addMsg("assistant", "<em>No response</em>"); return; }

    // Prefer chat text
    if (typeof resp.reply === "string" && resp.reply.trim()) {
      addMsg("assistant", `<p>${safe(resp.reply)}</p>`);
    }

    // Show SQL if present
    if (resp.sql) {
      addMsg("assistant", `<pre class="sql">${safe(resp.sql)}</pre>`);
    }

    // Show preview (string) if present
    if (resp.preview && typeof resp.preview === "string") {
      addMsg("assistant", `<pre class="preview">${safe(resp.preview)}</pre>`);
    }

    // Fallback: raw JSON if nothing else
    if (!resp.reply && !resp.preview && !resp.sql) {
      addMsg("assistant", `<pre class="json">${safe(JSON.stringify(resp, null, 2))}</pre>`);
    }
  }

  // ---------- send flow ----------
  async function sendQuery(text) {
    const q = (text ?? (el.input ? el.input.value : "")).trim();
    if (!q) return;

    addMsg("user", `<p>${safe(q)}</p>`);
    if (el.input) el.input.value = "";
    const tId = typingOn();

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ q, mode: currentMode })
      });
      const json = await res.json();
      typingOff(tId);
      renderResponse(json);
    } catch (err) {
      typingOff(tId);
      addMsg("assistant", `<p style="color:#b00020">Error: ${safe(err.message || err)}</p>`);
    }
  }

  // ---------- schema: Data → Extract ----------
  async function fetchSchemaAndRender() {
    const dataset = el.dataSelector ? el.dataSelector.value : "orders";
    try {
      const res = await fetch(`/schema?dataset=${encodeURIComponent(dataset)}`, {
        headers: { "Accept": "application/json" }
      });
      const payload = await res.json();

      if (!payload.ok) {
        addMsg("assistant", `<p style="color:#b00020">Schema error: ${safe(payload.error || "Unknown error")}</p>`);
        return;
      }

      const items = (payload.columns || []).map(c =>
        `<li><span class="schema-name">${safe(c.name)}</span> <span class="schema-type">- ${safe(c.type || "string")}</span></li>`
      ).join("");

      addMsg(
        "assistant",
        `
        <div class="schema-block">
          <div class="insight-subtitle">Columns in <strong>${safe(payload.dataset)}</strong></div>
          <ul class="schema-cols">${items || "<li>(no columns)</li>"}</ul>
        </div>
        `
      );
    } catch (err) {
      addMsg("assistant", `<p style="color:#b00020">Schema error: ${safe(err.message || err)}</p>`);
    }
  }

  // ---------- modes UI ----------
  function setMode(m) {
    currentMode = m;
    // optional highlight if you have these classes
    [el.chipWeb, el.chipData, el.chipAnalysis].forEach(b => b && b.classList.remove("active"));
    if (m === "web" && el.chipWeb) el.chipWeb.classList.add("active");
    if (m === "data" && el.chipData) el.chipData.classList.add("active");
    if (m === "analysis" && el.chipAnalysis) el.chipAnalysis.classList.add("active");
  }

  // ---------- wire events ----------
  function init() {
    // send
    if (el.send) el.send.addEventListener("click", () => sendQuery());
    if (el.input) el.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
      }
    });

    // reset
    if (el.reset) el.reset.addEventListener("click", () => { el.messages.innerHTML = ""; });

    // modes
    if (el.chipWeb) el.chipWeb.addEventListener("click", () => setMode("web"));
    if (el.chipData) el.chipData.addEventListener("click", () => setMode("data"));
    if (el.chipAnalysis) el.chipAnalysis.addEventListener("click", () => setMode("analysis"));

    // data → extract
    if (el.extractBtn) el.extractBtn.addEventListener("click", (e) => {
      e.preventDefault();
      fetchSchemaAndRender();
    });

    // default mode on load
    setMode("web");
    // small console ping
    console.log("[Noon AI] UI ready");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
