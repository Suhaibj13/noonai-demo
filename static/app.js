// ===== SAGE Frontend – final scroll + logo-safe settings =====
const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const inputEl = $("#input");
const sendBtn = $("#send");
const statusEl = $("#status");
const modeLabelEl = $("#mode");
const resetBtn = $("#reset-btn");

const state = { sending: false, mode: "web", history: [] };

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

  try{
    const res = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: q, mode: state.mode })
    });
    const data = await res.json();
    removeTyping();

    if (!res.ok || data?.reply === undefined){
      addMessage("ai", "Error: No response");
      if (statusEl) statusEl.textContent = "Error";
    }else{
      addMessage("ai", String(data.reply).trim());
      if (statusEl) statusEl.textContent = "Ready";
    }
  }catch(e){
    removeTyping();
    addMessage("ai", `Error: ${e.message}`);
    if (statusEl) statusEl.textContent = "Error";
  }finally{
    state.sending = false;
    sendBtn.disabled = false;
    scroller().scrollTop = scroller().scrollHeight;
  }
}

async function resetConversation(){
  messagesEl.innerHTML = "";
  try{
    const res = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: "reset", mode: state.mode })
    });
    const data = await res.json();
    addMessage("ai", data?.reply || "Started a new chat.");
  }catch(_){
    addMessage("ai", "Started a new chat.");
  }
  inputEl.focus();
}

/* Dynamic bottom padding so chat never hides under composer */
function updateComposerOffset(){
  const comp = document.querySelector(".composer");
  const px = comp ? comp.offsetHeight + 24 : 160;
  document.documentElement.style.setProperty("--composer-offset", px + "px");
  scroller().scrollTop = scroller().scrollHeight;
}

/* Events */
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
document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => setMode(btn.dataset.mode)));

window.addEventListener("load", () => { setMode(state.mode); updateComposerOffset(); });
window.addEventListener("resize", updateComposerOffset);

function scroller() {
  return document.querySelector(".chat");
}
function updateComposerOffset() {
  const comp = document.querySelector(".composer");
  const px = comp ? comp.offsetHeight + 24 : 160; // breathing room
  document.documentElement.style.setProperty("--composer-offset", px + "px");
  scroller().scrollTop = scroller().scrollHeight;
}
window.addEventListener("load", updateComposerOffset);
window.addEventListener("resize", updateComposerOffset);

// --- Audit Steps wiring ---
const projectSel = document.querySelector("#project");

document.querySelectorAll("#audit-steps .rail-btn[data-step]").forEach(btn => {
  btn.addEventListener("click", async () => {
    const step = (btn.dataset.step || "").trim(); // rcm | data_request | findings | report
    if (!step) return; // safety: don't fire if attribute missing

    const project = projectSel ? projectSel.value : "Inventory Management";
    // Optional: show what we're doing in the transcript
    addMessage("user", `Run "${step}" for project: ${project}`);
    if (statusEl) statusEl.textContent = `Running ${step}…`;
    addTyping();

    try {
      const res = await fetch("/run_step", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ step, project })
      });
      const data = await res.json();
      removeTyping();

      if (!res.ok) {
        addMessage("ai", data?.reply || "Error running step.");
        if (statusEl) statusEl.textContent = "Error";
      } else {
        addMessage("ai", String(data.reply || "(no reply)").trim());
        if (statusEl) statusEl.textContent = "Ready";
      }
    } catch (e) {
      removeTyping();
      addMessage("ai", `Error: ${e.message}`);
      if (statusEl) statusEl.textContent = "Error";
    }
  });
});

/* ===========================
   Project-aware presets
   =========================== */
// Helper: inject current project into {project} placeholder or prefix if absent
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

    fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: q, mode: state.mode })
    })
    .then(res => res.json().then(data => ({ ok: res.ok, data })))
    .then(({ ok, data }) => {
      removeTyping();
      if (!ok || data?.reply === undefined) {
        addMessage("ai", "Error: No response");
        if (statusEl) statusEl.textContent = "Error";
      } else {
        addMessage("ai", String(data.reply).trim());
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
