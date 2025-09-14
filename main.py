# main.py — Noon AI (restored architecture: Groq for Web/Data/Analysis)
from __future__ import annotations

import os
import re
import json
import textwrap
import logging
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="static", template_folder="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Memory for reuse in Analysis follow-ups
last_result_df: Optional[pd.DataFrame] = None
last_result_sql: Optional[str] = None
last_result_query: Optional[str] = None
REUSE_PHRASES = ["same as above", "use above", "previous result", "same dataset", "same data"]

# -----------------------------------------------------------------------------
# Groq client (lazy)
# -----------------------------------------------------------------------------
def llm_chat(messages: List[Dict], temperature: float = 0.0) -> str:
    """
    Minimal Groq chat wrapper. If GROQ_API_KEY is missing, raise.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    # Lazy import to avoid hard dependency if not used
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def safe_json(obj, status=200):
    return jsonify(obj), status

def trim_text(s: str, max_chars: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[:max_chars] + "\n..."

def extract_sql(text: str) -> str:
    """
    Extract SQL from a response that may contain ```sql blocks``` or plain SQL.
    """
    if not text:
        return ""
    m = re.search(r"```sql(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip().strip(";")
    m2 = re.search(r"```(.*?)```", text, flags=re.S | re.I)
    if m2:
        return m2.group(1).strip().strip(";")
    return text.strip().strip(";")

def df_preview_string(df: pd.DataFrame, rows: int = 50) -> str:
    try:
        return df.head(rows).to_string(index=False)
    except Exception:
        # fallback
        return str(df.head(rows))

def df_as_csv_snippet(df: pd.DataFrame, max_rows=300, max_chars=18000) -> str:
    if df is None or df.empty:
        return ""
    try:
        csv = df.head(max_rows).to_csv(index=False)
        return csv if len(csv) <= max_chars else (csv[:max_chars] + "\n...")
    except Exception:
        return df.head(50).to_string(index=False)

def pre_analyze(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0}
    numeric = df.select_dtypes(include="number").columns.tolist()
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric_cols": numeric[:10],
    }
    return summary

# -----------------------------------------------------------------------------
# Data loading (CSV)
# -----------------------------------------------------------------------------
def read_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.warning("Failed to read %s: %s", path, e)
            return pd.DataFrame()
    return pd.DataFrame()

def load_data_for_routes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (inventory_df, orders_df, mg_df)
    Looks for standard filenames at repo root.
    """
    inv = read_csv_if_exists("inventory.csv")
    ords = read_csv_if_exists("orders.csv")
    mg = read_csv_if_exists("minimum_guarantee.csv")
    # Normalize likely date columns (best effort, non-fatal)
    for df in (inv, ords, mg):
        for col in df.columns:
            if "date" in col.lower() or "created_at" in col.lower() or "sold_at" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except Exception:
                    pass
    return inv, ords, mg

# -----------------------------------------------------------------------------
# DuckDB execution
# -----------------------------------------------------------------------------
def build_memory_db(inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("inventory", inv)
    con.register("orders", ords)
    con.register("mg", mg)
    return con

def execute_sql(sql: str, inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame) -> pd.DataFrame:
    con = build_memory_db(inv, ords, mg)
    return con.execute(sql).fetch_df()

# -----------------------------------------------------------------------------
# Schema/Prompt builders
# -----------------------------------------------------------------------------
def build_schema_description(inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame) -> str:
    def one(df, name):
        cols = [f"- {c}: {str(df[c].dtype)}" for c in df.columns]
        return f"Table {name} columns:\n" + ("\n".join(cols) if cols else "(empty)")
    return "\n\n".join([one(inv, "inventory"), one(ords, "orders"), one(mg, "mg")])

def sql_generation_prompt(question: str, schema_desc: str, analysis_style: bool) -> List[Dict]:
    sys = (
        "You are a senior data analyst. Generate a single valid DuckDB SQL query that answers the user's question.\n"
        "Use only the tables and columns shown in the schema. Prefer explicit joins.\n"
        "DO NOT invent columns. Output ONLY the SQL (no commentary)."
    )
    if analysis_style:
        sys += "\n- Prefer GROUP BY, aggregations, top-N, counts, sums, averages, or time buckets when relevant."
    usr = f"Schema:\n{schema_desc}\n\nQuestion: {question}\nReturn only SQL."
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

def sql_repair_prompt(question: str, schema_desc: str, bad_sql: str, error_msg: str) -> List[Dict]:
    sys = (
        "You generated SQL that failed. Repair it to be valid DuckDB SQL given the schema.\n"
        "Return ONLY the corrected SQL. Do not add comments or explanation."
    )
    usr = f"Schema:\n{schema_desc}\n\nQuestion: {question}\n\nBad SQL:\n{bad_sql}\n\nError:\n{error_msg}\n\nCorrected SQL (only):"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

def build_answer_prompt(user_query: str, df_summary: dict, df_csv_head: str) -> List[Dict]:
    sys = (
        "You are an expert analyst. Use ONLY the provided result rows to answer the user's question precisely.\n"
        "Start your response with 'Answer: ...' on the first line.\n"
        "Then add up to 5 short bullets with supporting numbers.\n"
        "If the rows are insufficient, say exactly what data/filters are missing.\n"
        "No code blocks, no SQL."
    )
    usr = f"QUESTION: {user_query}\n\nDATA SUMMARY: {json.dumps(df_summary)}\n\nRESULT ROWS (CSV head):\n{df_csv_head}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

# -----------------------------------------------------------------------------
# Web flow (Wikipedia + Groq)
# -----------------------------------------------------------------------------
def fetch_wikipedia_snippet(q: str) -> str:
    try:
        # Very small helper; not guaranteed to find a page for every query.
        term = q.strip().split("\n")[0][:150]
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(term)}"
        r = requests.get(url, timeout=6)
        if r.ok:
            data = r.json()
            extract = data.get("extract") or ""
            return extract
    except Exception as e:
        logging.info("wiki fetch failed: %s", e)
    return ""

def run_web_flow(q: str) -> Dict:
    snippet = fetch_wikipedia_snippet(q)
    if GROQ_API_KEY:
        # Let Groq compose a concise answer with the snippet (if any)
        context = f"Wikipedia says: {snippet}" if snippet else "No web snippet available."
        msg = [
            {"role": "system", "content": "Be concise and factual. If you have a snippet, ground your answer in it."},
            {"role": "user", "content": f"Question: {q}\n\nContext: {context}\n\nAnswer briefly:"},
        ]
        try:
            answer = llm_chat(msg, temperature=0.2)
            return {"ok": True, "mode": "web", "reply": answer}
        except Exception as e:
            logging.info("Groq web answer failed: %s", e)
    # Fallback
    return {"ok": True, "mode": "web", "reply": snippet or "I couldn't fetch a web snippet, but I can still help analyze your data."}

# -----------------------------------------------------------------------------
# Data flow (Groq → SQL → DuckDB)
# -----------------------------------------------------------------------------
def run_data_flow(question: str, analysis_style: bool = False) -> Dict:
    global last_result_df, last_result_sql, last_result_query
    inv, ords, mg = load_data_for_routes()

    schema = build_schema_description(inv, ords, mg)
    if not GROQ_API_KEY:
        # No Groq: naive fallback for demo safety
        sql = "SELECT * FROM orders LIMIT 50"
        try:
            df = execute_sql(sql, inv, ords, mg)
        except Exception:
            df = pd.DataFrame()
        last_result_df, last_result_sql, last_result_query = df, sql, question
        return {
            "reply": f"(fallback) Showing first {len(df)} rows from orders.",
            "sql": sql,
            "preview": trim_text(df_preview_string(df), 2000),
        }

    # 1) Ask Groq for SQL
    sql_ask = sql_generation_prompt(question, schema, analysis_style=analysis_style)
    try:
        sql_text = llm_chat(sql_ask, temperature=0.0)
    except Exception as e:
        return {"reply": f"SQL generation error: {e}", "sql": None, "preview": None}

    sql = extract_sql(sql_text)
    # 2) Execute; if error → ask Groq to repair once
    try:
        df = execute_sql(sql, inv, ords, mg)
    except Exception as ex:
        repair_ask = sql_repair_prompt(question, schema, sql, str(ex))
        try:
            repaired = llm_chat(repair_ask, temperature=0.0)
            sql = extract_sql(repaired)
            df = execute_sql(sql, inv, ords, mg)
        except Exception as ex2:
            return {"reply": f"SQL failed: {ex2}", "sql": sql, "preview": None}

    last_result_df, last_result_sql, last_result_query = df, sql, question
    return {
        "reply": f"Executed SQL. Showing first {min(50, len(df))} rows.",
        "sql": sql,
        "preview": trim_text(df_preview_string(df), 2000),
    }

# -----------------------------------------------------------------------------
# Analysis flow (Groq → analysis SQL → answer from rows)
# -----------------------------------------------------------------------------
def run_analysis_flow(user_query: str) -> Dict:
    """
    Uses Groq to:
      1) generate analysis-oriented SQL (aggregations / group-bys) and run it,
      2) answer the user's question *directly from the result rows*.
    """
    global last_result_df, last_result_sql, last_result_query

    wants_reuse = any(p in (user_query or "").lower() for p in REUSE_PHRASES)

    # Step 1: get a dataset (reuse previous result if asked)
    if wants_reuse and last_result_df is not None:
        df = last_result_df.copy()
        sql_used = last_result_sql or "(previous result)"
    else:
        data_res = run_data_flow(user_query, analysis_style=True)
        if not data_res.get("preview"):
            # Bubble up any error/notice from data flow
            return data_res
        df = last_result_df.copy()
        sql_used = last_result_sql

    # Single-cell answer? return directly.
    try:
        if isinstance(df, pd.DataFrame) and df.shape == (1, 1):
            val = df.iloc[0, 0]
            return {
                "reply": f"Answer: {val}\n\n(derived from: {sql_used})",
                "sql": sql_used,
                "preview": trim_text(df_preview_string(df), 2000),
            }
    except Exception:
        pass

    # Step 2: Ask Groq to compose the *answer* from the actual rows
    if not GROQ_API_KEY:
        # No Groq: simple fallback narrative
        return {
            "reply": "Analysis fallback (no Groq). See the previewed rows and compute the metric you asked.",
            "sql": sql_used,
            "preview": trim_text(df_preview_string(df), 2000),
        }

    summary = pre_analyze(df)
    csv_head = df_as_csv_snippet(df)
    answer_msgs = build_answer_prompt(user_query, summary, csv_head)

    try:
        answer = llm_chat(answer_msgs, temperature=0.0)
        return {
            "reply": answer,
            "sql": sql_used,
            "preview": trim_text(df_preview_string(df), 2000),
        }
    except Exception as ex:
        logging.info("Groq analysis answer failed: %s", ex)
        return {
            "reply": f"Analysis error. Here are the rows derived from: {sql_used}",
            "sql": sql_used,
            "preview": trim_text(df_preview_string(df), 2000),
        }

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return "ok", 200

@app.post("/ask")
def ask():
    payload = request.get_json(force=True) or {}
    q = (payload.get("q") or payload.get("query") or "").strip()
    mode = (payload.get("mode") or "web").lower()

    if not q:
        return safe_json({"ok": True, "mode": mode, "reply": "Please type a question."})

    if mode == "analysis":
        res = run_analysis_flow(q)
        return safe_json(res)

    if mode == "data":
        res = run_data_flow(q, analysis_style=False)
        return safe_json(res)

    # default: web
    res = run_web_flow(q)
    return safe_json(res)

# -----------------------------------------------------------------------------
# Local dev
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
