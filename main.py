# main.py — Noon AI stable baseline (Flask)
# - Keeps your old Web/Data behavior intact
# - Adds robust Analysis mode via analysis_engine.py
# - Serves index and static safely for Render/Flask
from __future__ import annotations

import os
import logging
from typing import List, Dict, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory

# --- Flask app ---
app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO)

# -------------- Utilities --------------
def safe_json(obj, status: int = 200):
    """Return JSON consistently, no HTML error pages."""
    return jsonify(obj), status

def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

# -------------- Index / health --------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return "ok", 200

# -------------- Analysis engine --------------
# You must also add analysis_engine.py next to this file (see below).
try:
    from analysis_engine import (
        run_analysis_router,
        load_dataset_frame,        # used by Data mode helper
        detect_datasets_and_task,  # used by Data mode helper
    )
except Exception as e:
    # Don’t crash if the file is missing; Web/Data still work.
    logging.exception("analysis_engine import failed: %s", e)
    run_analysis_router = None
    load_dataset_frame = None
    detect_datasets_and_task = None

# -------------- Web / Data helpers --------------
def detect_datasets(q: str) -> List[str]:
    ql = (q or "").lower()
    ds: List[str] = []
    if "order" in ql or "sales" in ql:
        ds.append("orders")
    if "inventory" in ql or "stock" in ql:
        ds.append("inventory")
    if "mg" in ql or "minimum guarantee" in ql:
        ds.append("mg")
    # de-dupe, preserve order
    return list(dict.fromkeys(ds)) or ["orders"]

def preview_table(df, limit: int = 30):
    try:
        recs = df.head(limit).to_dict(orient="records")
        return {"columns": list(df.columns), "rows": recs}
    except Exception as e:
        return {"error": f"preview failed: {e}"}

def run_data_flow(q: str) -> Dict:
    """
    Simple data mode: pick dataset(s) from query, load CSV(s), and return a preview.
    This keeps old behavior (table-in, table-out) so the UI renders a table.
    """
    import pandas as pd

    datasets = detect_datasets(q) if detect_datasets_and_task is None else detect_datasets_and_task(q, [])[0]
    results = {}
    for name in datasets:
        # try to load via analysis_engine resolver if available; else fall back to canonical names
        try:
            if load_dataset_frame:
                df = load_dataset_frame(name)
            else:
                fname = {
                    "orders": "orders.csv",
                    "inventory": "inventory.csv",
                    "mg": "minimum_guarantee.csv",
                    "minimum_guarantee": "minimum_guarantee.csv",
                }.get(name, f"{name}.csv")
                if not file_exists(fname):
                    results[name] = {"error": f"missing file: {fname}"}
                    continue
                import pandas as pd  # local import keeps boot fast
                df = pd.read_csv(fname)
            results[name] = preview_table(df)
        except Exception as e:
            results[name] = {"error": str(e)}

    # If there is just one dataset, return its row list directly so UI renders a table.
    if len(results) == 1:
        only = next(iter(results.values()))
        if "rows" in only:
            return only["rows"]
        return only
    return results

def run_web_flow(q: str) -> str:
    """
    Web mode (non-LLM placeholder). Return a human string so the UI shows a chat bubble,
    not raw JSON. You can wire your actual web/RAG later.
    """
    if not q:
        return "Ask me about your orders, inventory, or MG payouts."
    return (
        "Got it. I’m in Web mode (demo). Try:\n"
        "• 'Analyze orders' (Data / Analysis)\n"
        "• 'Analyze order and inventory data and give me stockouts for next 10 days' (Analysis)\n"
        "• 'Generate RCM for {project}' (Web)"
    )

# -------------- /ask API --------------
@app.post("/ask")
def ask():
    payload = request.get_json(force=True) or {}
    # Accept both new 'q' and old 'query' to avoid breaking older UI
    q = (payload.get("q") or payload.get("query") or "").strip()
    mode = (payload.get("mode") or "web").lower()
    datasets = payload.get("datasets") or []
    file_hints = payload.get("fileHints") or []

    if not q:
        return safe_json({"ok": True, "reply": "Please type a question.", "mode": mode})

    # ----- Analysis (insights object) -----
    if mode == "analysis":
        if run_analysis_router is None:
            return safe_json({"type": "insights", "kpis": {}, "insights": ["Analysis engine not available."]})
        try:
            res = run_analysis_router(q, datasets=datasets, file_hints=file_hints)
            # Return raw insights object so the UI picks it up
            return safe_json(res, 200)
        except Exception as e:
            logging.exception("analysis failed")
            return safe_json({"type": "insights", "kpis": {}, "insights": [f"Error: {e}"]}, 200)

    # ----- Data (preview tables) -----
    if mode == "data":
        try:
            res = run_data_flow(q)
            # Return rows list to render a table, or object if multi-dataset
            return safe_json(res, 200)
        except Exception as e:
            logging.exception("data mode failed")
            return safe_json({"ok": False, "error": str(e)}, 500)

    # ----- Web (string reply) -----
    try:
        reply = run_web_flow(q)
        # IMPORTANT: return as an object with 'reply' so the UI can show a text bubble
        return safe_json({"ok": True, "mode": "web", "reply": reply}, 200)
    except Exception as e:
        logging.exception("web mode failed")
        return safe_json({"ok": False, "error": str(e)}, 500)

# -------------- Local dev runner --------------
if __name__ == "__main__":
    # Local dev only; Render uses gunicorn.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
