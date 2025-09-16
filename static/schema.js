/* static/schema.js — self-contained Schema viewer
   - Adds a click handler for #schemaViewBtn
   - Calls /schema?dataset=...
   - Renders a table: column | datatype
   - No dependencies on app.js (it won’t double-bind or conflict)
*/
(function () {
  // ---------- small, local helpers (no globals needed) ----------
  const $ = (sel, root=document) => root.querySelector(sel);
  const safe = (s) => (s == null ? "" : String(s)
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"));

  const els = {
    messages: $("#messages") || document.body,
    selector:  $("#schema-selector") || $("#data-selector"),
    viewBtn:   $("#schemaViewBtn")   || $("#extractBtn"),
    status:    $("#status")
  };

  function addAIBubble(html){
    const wrap = document.createElement("div");
    wrap.className = "msg msg-ai";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;
    wrap.appendChild(bubble);
    els.messages.appendChild(wrap);
    scrollToBottom();
  }

  function typingOn(){
    const id = `typing-${Date.now()}`;
    const wrap = document.createElement("div");
    wrap.className = "msg msg-ai";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML =
      `<span id="${id}" class="typing">
         <span class="dot"></span><span class="dot"></span><span class="dot"></span>
       </span>`;
    wrap.appendChild(bubble);
    els.messages.appendChild(wrap);
    scrollToBottom();
    if (els.status) els.status.textContent = "Fetching schema…";
    return id;
  }

  function typingOff(id){
    const node = document.getElementById(id);
    if (!node) return;
    const wrap = node.closest(".msg");
    if (wrap && wrap.parentNode) wrap.parentNode.removeChild(wrap);
    else node.remove();
    if (els.status) els.status.textContent = "Ready";
    scrollToBottom(false);
  }

  function scrollToBottom(smooth=true){
    const behavior = smooth && "scrollBehavior" in document.documentElement.style ? "smooth" : "auto";
    requestAnimationFrame(() => {
      try { els.messages.scrollTo({ top: els.messages.scrollHeight, behavior }); }
      catch { els.messages.scrollTop = els.messages.scrollHeight; }
      try { window.scrollTo({ top: document.documentElement.scrollHeight, behavior }); } catch {}
    });
  }

  async function fetchSchemaAndRender(){
    const dataset = (els.selector && els.selector.value) ? els.selector.value : "orders";
    const tId = typingOn();
    try{
      const res = await fetch(`/schema?dataset=${encodeURIComponent(dataset)}`, {
        headers: { "Accept": "application/json" }
      });
      const payload = await res.json();
      typingOff(tId);

      if (!payload.ok){
        addAIBubble(`<p style="color:#b00020">Schema error: ${safe(payload.error || "Unknown error")}</p>`);
        return;
      }
      const rows = (payload.columns || [])
        .map(c => `<tr><td><code>${safe(c.name)}</code></td><td>${safe(c.type || "string")}</td></tr>`)
        .join("");

      addAIBubble(`
        <div>
          <div class="insight-subtitle">Schema for <strong>${safe(payload.dataset)}</strong></div>
          <table class="schema-table">
            <thead><tr><th>column</th><th>datatype</th></tr></thead>
            <tbody>${rows || "<tr><td colspan='2'>(no columns)</td></tr>"}</tbody>
          </table>
        </div>
      `);
    }catch(err){
      typingOff(tId);
      addAIBubble(`<p style="color:#b00020">Schema error: ${safe(err.message || err)}</p>`);
    }
  }

  function init(){
    if (!els.viewBtn) return;
    // Guard against double-binding if app.js also attached a handler
    if (els.viewBtn.dataset.bound === "1") return;
    els.viewBtn.dataset.bound = "1";

    els.viewBtn.addEventListener("click", (e) => {
      e.preventDefault();
      fetchSchemaAndRender();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
