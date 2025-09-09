// static/fetchJson.js
// Exposes window.fetchJson(input, init) that safely parses JSON and surfaces HTML errors

window.fetchJson = async function fetchJson(input, init) {
  const res = await fetch(input, init);
  const text = await res.text(); // read as text first for safe diagnostics

  const isJson =
    (res.headers.get("content-type") || "").toLowerCase().includes("application/json");

  if (!res.ok || !isJson) {
    const preview = text.slice(0, 300);
    throw new Error(`HTTP ${res.status} ${res.statusText} — Not JSON\n${preview}`);
  }

  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(`JSON parse error: ${e.message}\nSnippet: ${text.slice(0, 300)}`);
  }
};
