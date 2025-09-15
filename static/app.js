/* static/app.js — stable bubbles + processing + Data→Extract schema viewer */

(function () {
  // ---------- helpers ----------
  const $  = (sel, root=document) => root.querySelector(sel);
  const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
  const safe = (s) => (s == null ? "" : String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"));

  // ---------- DOM ----------
  const el = {
    messages: $("#messages") || document.body,
    input: $("#input") || $("textarea") || $('input[type="text"]'),
    send: $("#send") || $("#sendBtn"),
    reset: $("#reset-btn") || $("#reset") || $("#resetBtn"),
    modeBadge: $("#mode"),
    status: $("#status"),
    modeBtns: $$('.mode-btn[data-mode]'),
    dataSelector: $("#data-selector"),
    extractBtn: $("#extractBtn"),
  };

  // ---------- state ----------
  let currentMode = "web"; // "web" | "data" | "analysis"

  // ---------- status / mode badges ----------
  function setStatus(text){ if (el.status) el.status.textContent = text; }
  function setModeBadge(mode){
    if (el.modeBadge) el.modeBadge.textContent = `Mode: ${mode[0].toUpperCase()}${mode.slice(1)}`;
  }

  // ---------- bubble rendering ----------
  function addBubble(role, innerHTML){
    const wrap = document.createElement("div");
    wrap.className = role === "user" ? "msg msg-user" : "msg msg-ai";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = innerHTML;
    wrap.appendChild(bubble);
    el.messages.appendChild(wrap);
    el.messages.scrollTop = el.messages.scrollHeight;
    return bubble; // in case caller needs id
  }

  function addUser(html){ addBubble("user", html); }

  function addAI(html){ addBubble("ai", html); }

  function typingOn(){
    const id = `typing-${Date.now()}`;
    const b = addBubble("ai",
      `<span id="${id}" class="typing">
         <span class="dot"></span><span class="dot"></span><span class="dot"></span>
       </span>`);
    return id;
  }

  function typingOff(id){
    const node = document.getElementById(id);
    if (!node) return;
    const wrap = node.closest(".msg");
    if (wrap && wrap.parentNode) wrap.parentNode.removeChild(wrap);
    else node.remove();
  }

  // ---------- render server responses ----------
  function renderResponse(resp){
    if (!resp){ addAI("<em>No response</em>"); return; }

    if (typeof resp.reply === "string" && resp.reply.trim()){
      addAI(`<p>${safe(resp.reply)}</p>`);
    }
    if (resp.sql){
      addAI(`<pre class="sql">${safe(resp.sql)}</pre>`);
    }
    if (resp.preview && typeof resp.preview === "string"){
      addAI(`<pre class="preview">${safe(resp.preview)}</pre>`);
    }
    if (!resp.reply && !resp.sql && !resp.preview){
      addAI(`<pre class="json">${safe(JSON.stringify(resp, null, 2))}</pre>`);
    }
  }

  // ---------- send flow ----------
  async function sendQuery(text){
    const q = (text ?? (el.input ? el.input.value : "")).trim();
    if (!q) return;

    addUser(`<p>${safe(q)}</p>`);
    if (el.input) el.input.value = "";
    const tId = typingOn(); setStatus("Processing…");

    try{
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type":"application/json", "Accept":"application/json" },
        body: JSON.stringify({ q, mode: currentMode })
      });
      const json = await res.json();
      typingOff(tId); setStatus("Ready");
      renderResponse(json);
    }catch(err){
      typingOff(tId); setStatus("Ready");
      addAI(`<p style="color:#b00020">Error: ${safe(err.message || err)}</p>`);
    }
  }

  // ---------- schema: Data → Extract ----------
  async function fetchSchemaAndRender(){
    const dataset = el.dataSelector ? el.dataSelector.value : "orders";
    const tId = typingOn(); setStatus("Extracting schema…");

    try{
      const res = await fetch(`/schema?dataset=${encodeURIComponent(dataset)}`, {
        headers: { "Accept":"application/json" }
      });
      const payload = await res.json();
      typingOff(tId); setStatus("Ready");

      if (!payload.ok){
        addAI(`<p style="color:#b00020">Schema error: ${safe(payload.error || "Unknown error")}</p>`);
        return;
      }
      const items = (payload.columns || []).map(c =>
        `<li><strong>${safe(c.name)}</strong> - ${safe(c.type || "string")}</li>`
      ).join("");

      addAI(`
        <div>
          <div class="insight-subtitle">Columns in <em>${safe(payload.dataset)}</em></div>
          <ul>${items || "<li>(no columns)</li>"}</ul>
        </div>
      `);
    }catch(err){
      typingOff(tId); setStatus("Ready");
      addAI(`<p style="color:#b00020">Schema error: ${safe(err.message || err)}</p>`);
    }
  }

  // ---------- modes ----------
  function setMode(m){
    currentMode = m;
    setModeBadge(m);
    el.modeBtns.forEach(b => b.classList.toggle("active", b.dataset.mode === m));
  }

  // ---------- wire events ----------
  function init(){
    // modes
    el.modeBtns.forEach(btn => btn.addEventListener("click", () => setMode(btn.dataset.mode)));

    // send
    if (el.send) el.send.addEventListener("click", () => sendQuery());
    if (el.input) el.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendQuery(); }
    });

    // reset
    if (el.reset) el.reset.addEventListener("click", () => { el.messages.innerHTML = ""; });

    // extract
    if (el.extractBtn) el.extractBtn.addEventListener("click", (e) => {
      e.preventDefault();
      fetchSchemaAndRender();
    });

    // default
    setMode("web");
    setStatus("Ready");
    console.log("[Noon AI] UI ready");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
