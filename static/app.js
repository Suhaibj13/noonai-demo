// ===== SAGE Frontend – final scroll + logo-safe settings (fixed JSON handling) =====
// --- safety: define fetchJson if the helper file didn't load ---
if (!window.fetchJson) {
  window.fetchJson = async function fetchJson(input, init) {
    const res = await fetch(input, init);
    const text = await res.text();
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const isJson = ct.includes("application/json");
    if (!res.ok || !isJson) {
      const snippet = text.slice(0, 300);
      throw new Error(`HTTP ${res.status} ${res.statusText} — Not JSON\n${snippet}`);
    }
    try {
      return JSON.parse(text);
    } catch (e) {
      throw new Error(`JSON parse error: ${e.message}\nSnippet: ${text.slice(0, 300)}`);
    }
  };
}

// ensure these exist even before user touches the dropdown
window.PROJECT = window.PROJECT || { mainTable: null, joins: [] };
window.LAST_AI_REPLY = "";


const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const inputEl = $("#input");
const sendBtn = $("#send");
const statusEl = $("#status");
const modeLabelEl = $("#mode");
const resetBtn = $("#reset-btn");
const schemaSel = document.querySelector("#schema-selector");
if (schemaSel) {
  window.PROJECT.mainTable = schemaSel.value;        // init on load
  schemaSel.addEventListener("change", () => {
    window.PROJECT.mainTable = schemaSel.value;      // update on change
  });
}

const state = { sending: false, mode: "web", history: [] };

// Mode switching
function setMode(newMode){
  state.mode = newMode;
  document.querySelectorAll(".mode-btn").forEach(btn => {
    const active = btn.dataset.mode === newMode;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  const placeholders = {
    data: "Data mode: ask for rows/filters/joins/aggregates over your tables…",
    analysis: "Analysis mode: ask for insights, trends, segments, anomalies…",
    web: "Web mode: quick factual answers (type 'observations' to get O/R/R)…"
  };
  inputEl.placeholder = placeholders[newMode] || "Type your question…";
  if (modeLabelEl) modeLabelEl.textContent = `Mode: ${newMode[0].toUpperCase()}${newMode.slice(1)}`;
}

// Chat helpers
function addMessage(role, content){
  state.history.push({ role, content });
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "user" ? "msg-user" : "msg-ai"}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scroller().scrollTop = scroller().scrollHeight;
}

function addTyping(){
  const wrap = document.createElement("div");
  wrap.className = "msg msg-ai";
  wrap.id = "typing";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scroller().scrollTop = scroller().scrollHeight;
}
function removeTyping(){ document.getElementById("typing")?.remove(); }

function scroller() {
  return document.querySelector(".chat");
}

// Ensure chat area never hides under composer
function updateComposerOffset(){
  const comp = document.querySelector(".composer");
  const px = comp ? comp.offsetHeight + 24 : 160;
  document.documentElement.style.setProperty("--composer-offset", px + "px");
  scroller().scrollTop = scroller().scrollHeight;
}

// Send handler (uses /ask)
async function send(){
  const q = (inputEl.value || "").trim();
  if (!q || state.sending) return;

  addMessage("user", q);
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendBtn.disabled = true;
  state.sending = true;
  if (statusEl) statusEl.textContent = "Processing…";
  addTyping();

  try {
    const projectTable = (window.PROJECT && window.PROJECT.mainTable) || null;
    const joins        = (window.PROJECT && window.PROJECT.joins) || [];
  
    const res = await fetchJson("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        mode: state.mode,
        projectTable,   // NEW
        joins,           // NEW
        priorReply: window.LAST_AI_REPLY || ""     // NEW
      })
    });
  
    removeTyping();
  
    // ⬇⬇⬇ CHANGE: use `res` (not `data`) and guard errors
    if (!res || res.error) {
      addMessage("ai", `Error: ${res?.error || "Unknown error"}`);
    } else {
      addMessage("ai", String((res.reply ?? "(no reply)")).trim());
      // B — store last reply
      if (res && typeof res.reply === "string") {
        window.LAST_AI_REPLY = res.reply;
      }
      
      if (res && res.preview && typeof window.renderPreview === "function") {
        window.renderPreview(res.preview);
      }
      if (res && res.sql && typeof window.renderSql === "function") {
        window.renderSql(res.sql);
      }
    }
    // ⬆⬆⬆
  
    if (statusEl) statusEl.textContent = "Ready";
  } catch (e) {
    removeTyping();
    addMessage("ai", `Error: ${e.message}`);
    if (statusEl) statusEl.textContent = "Error";
  } finally {
    state.sending = false;
    sendBtn.disabled = false;
    scroller().scrollTop = scroller().scrollHeight;
  }
}

// Reset conversation (posts "reset" via /ask)
async function resetConversation(){
  messagesEl.innerHTML = "";
  window.LAST_AI_REPLY = ""; 
  try{
    const q = ""; // <— define q for this request
    const projectTable = (window.PROJECT && window.PROJECT.mainTable) || null;
    const joins        = (window.PROJECT && window.PROJECT.joins) || [];

    const res = await fetchJson("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        mode: state.mode,
        projectTable,
        joins,
        priorReply: window.LAST_AI_REPLY || ""     // NEW
      })
    });

    // use `res`, not `data`
    addMessage("ai", String((res?.reply ?? "Started a new chat.")).trim());
  } catch(_){
    addMessage("ai", "Started a new chat.");
  }
  inputEl.focus();
}

// Events
sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); send(); }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  updateComposerOffset();
});
resetBtn.addEventListener("click", resetConversation);
document.querySelectorAll(".mode-btn").forEach(btn =>
  btn.addEventListener("click", () => setMode(btn.dataset.mode))
);

window.addEventListener("load", () => { setMode(state.mode); updateComposerOffset(); });
window.addEventListener("resize", updateComposerOffset);

// --- Audit Steps wiring ---
const projectSel = document.querySelector("#project");

// === Auto-send for Audit buttons and chips ===
document.querySelectorAll("[data-q]").forEach(btn => {
  btn.addEventListener("click", () => {
    const raw = btn.dataset.q;
    if (!raw) return;

    const q = materializePrompt(raw);

    // show it as if the user typed it
    addMessage("user", q);
    inputEl.value = "";
    sendBtn.disabled = true;
    state.sending = true;
    if (statusEl) statusEl.textContent = "Processing…";
    addTyping();

    const projectTable = (window.PROJECT && window.PROJECT.mainTable) || null;
    const joins        = (window.PROJECT && window.PROJECT.joins) || [];

    window.fetchJson("/ask", {
      method: "POST",
      headers: { "Content-Type":"application/json", "Accept":"application/json" },
      body: JSON.stringify({
        query: q,
        mode: state.mode,
        projectTable,   // NEW
        joins           // NEW
      })
    })
    .then((res) => {
      removeTyping();
      if (!res || res.error) {
        addMessage("ai", `Error: ${res?.error || "Unknown error"}`);
      } else {
        addMessage("ai", String((res.reply ?? "(no reply)")).trim());

        // B — store last reply
        if (res && typeof res.reply === "string") {
          window.LAST_AI_REPLY = res.reply;
        }
    
        // ⬇⬇⬇ 2nd point goes here (guarded preview/SQL renderers)
        if (res && res.preview && typeof window.renderPreview === "function") {
          window.renderPreview(res.preview);
        }
        if (res && res.sql && typeof window.renderSql === "function") {
          window.renderSql(res.sql);
        }
        // ⬆⬆⬆
    
        if (statusEl) statusEl.textContent = "Ready";
      }
    })

    .catch(e => {
      removeTyping();
      addMessage("ai", `Error: ${e.message}`);
      if (statusEl) statusEl.textContent = "Error";
    })
    .finally(() => {
      state.sending = false;
      sendBtn.disabled = false;
    });
  });
});


// ===========================
// Project-aware presets
// ===========================
function materializePrompt(tpl) {
  const sel = document.querySelector("#project");
  const project = sel ? sel.value : "Inventory Management";
  const filled = (tpl || "").replaceAll("{project}", project);
  if (filled === tpl) {
    return `Project: ${project}. ${tpl}`;
  }
  return filled;
}

// === Auto-send for Audit buttons and chips ===
document.querySelectorAll("[data-q]").forEach(btn => {
  btn.addEventListener("click", () => {
    const raw = btn.dataset.q;
    if (!raw) return;

    const q = materializePrompt(raw);

    // show it as if the user typed it
    addMessage("user", q);
    inputEl.value = "";
    sendBtn.disabled = true;
    state.sending = true;
    if (statusEl) statusEl.textContent = "Processing…";
    addTyping();

    window.fetchJson("/ask", {
      method: "POST",
      headers: { "Content-Type":"application/json", "Accept":"application/json" },
      body: JSON.stringify({ query: q, mode: state.mode })
    })
    .then((data) => {
      removeTyping();
      addMessage("ai", String((data.reply ?? "(no reply)")).trim());
      if (statusEl) statusEl.textContent = "Ready";
    })
    .catch(e => {
      removeTyping();
      addMessage("ai", `Error: ${e.message}`);
      if (statusEl) statusEl.textContent = "Error";
    })
    .finally(() => {
      state.sending = false;
      sendBtn.disabled = false;
    });
  });
});
