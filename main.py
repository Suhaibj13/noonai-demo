# main.py — Noon AI (Groq for Web/Data/Analysis) + /schema + robust MG synonyms
from __future__ import annotations

import os, re, json, logging
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
)

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="static", template_folder="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# previous result cache for follow-ups
last_result_df: Optional[pd.DataFrame] = None
last_result_sql: Optional[str] = None
last_result_query: Optional[str] = None

# detect follow-ups that refer to the previous dataset
REUSE_PHRASES = [
    "same as above","use above","previous result","same dataset","same data",
    "these","those","them","out of these","from these","of these","above dataset","previous"
]

# -----------------------------------------------------------------------------
# Groq client
# -----------------------------------------------------------------------------
def llm_chat(messages: List[Dict], temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def safe_json(obj, status=200): return jsonify(obj), status

def trim_text(s: str, max_chars: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[:max_chars] + "\n..."

def extract_sql(text: str) -> str:
    if not text: return ""
    m = re.search(r"```sql(.*?)```", text, flags=re.S|re.I)
    if m: return m.group(1).strip().strip(";")
    m = re.search(r"```(.*?)```", text, flags=re.S|re.I)
    if m: return m.group(1).strip().strip(";")
    return text.strip().strip(";")

def df_preview_string(df: pd.DataFrame, rows: int = 50) -> str:
    try:    return df.head(rows).to_string(index=False)
    except: return str(df.head(rows))

def df_as_csv_snippet(df: pd.DataFrame, max_rows=300, max_chars=18000) -> str:
    if df is None or df.empty: return ""
    try:
        csv = df.head(max_rows).to_csv(index=False)
        return csv if len(csv) <= max_chars else (csv[:max_chars] + "\n...")
    except Exception:
        return df.head(50).to_string(index=False)

def pre_analyze(df: pd.DataFrame) -> Dict:
    if df is None or df.empty: return {"rows":0, "cols":0}
    numeric = df.select_dtypes(include="number").columns.tolist()
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "numeric_cols": numeric[:10]}

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def read_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:    return pd.read_csv(path)
        except Exception as e:
            logging.warning("Failed to read %s: %s", path, e)
            return pd.DataFrame()
    return pd.DataFrame()

def load_data_for_routes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inv = read_csv_if_exists("inventory.csv")
    ords = read_csv_if_exists("orders.csv")
    mg  = read_csv_if_exists("minimum_guarantee.csv")
    # soft date coercion (best-effort)
    for df in (inv, ords, mg):
        for col in df.columns:
            cl = col.lower()
            if "date" in cl or "created_at" in cl or "sold_at" in cl or "shipped" in cl or "delivered" in cl or "returned" in cl:
                try: df[col] = pd.to_datetime(df[col], errors="ignore")
                except: pass
    return inv, ords, mg

# -----------------------------------------------------------------------------
# DuckDB
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
# Schema & prompts
# -----------------------------------------------------------------------------
def build_schema_description(inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame) -> str:
    def one(df, name):
        cols = [f"- {c}: {str(df[c].dtype)}" for c in df.columns]
        return f"Table {name} columns:\n" + ("\n".join(cols) if cols else "(empty)")
    return "\n\n".join([one(inv,"inventory"), one(ords,"orders"), one(mg,"mg")])

def sql_generation_prompt(question: str, schema_desc: str, analysis_style: bool,
                          previous_sql: Optional[str] = None) -> List[Dict]:
    sys = (
        "You are a senior data analyst. Generate a single valid DuckDB SQL query that answers the user's question.\n"
        "Use only the tables/columns present in the provided schema (exact names). "
        "Prefer explicit joins. Do not invent columns.\n"
        "Return ONLY the SQL (no comments, no markdown)."
    )
    if analysis_style:
        sys += "\n- Prefer GROUP BY, aggregations, top-N, counts, sums, averages, or time buckets when relevant."
    if previous_sql:
        sys += "\n- If appropriate, MODIFY the PREVIOUS SQL to keep the same dataset scope/filters and add only the extra logic needed."
    usr = f"Schema:\n{schema_desc}\n\nQuestion: {question}\n"
    if previous_sql:
        usr += f"\nPREVIOUS SQL:\n{previous_sql}\n"
    usr += "\nReturn only SQL."
    return [{"role":"system","content":sys},{"role":"user","content":usr}]

def sql_repair_prompt(question: str, schema_desc: str, bad_sql: str, error_msg: str) -> List[Dict]:
    sys = (
        "The SQL failed to run. Repair it to valid DuckDB SQL using only columns in the schema. "
        "If you used synonyms, replace them with exact column names from the schema. "
        "Return ONLY the corrected SQL."
    )
    usr = f"Schema:\n{schema_desc}\n\nQuestion: {question}\n\nBad SQL:\n{bad_sql}\n\nError:\n{error_msg}\n\nCorrected SQL:"
    return [{"role":"system","content":sys},{"role":"user","content":usr}]

def build_answer_prompt(user_query: str, df_summary: dict, df_csv_head: str) -> List[Dict]:
    sys = (
        "You are an expert analyst. Use ONLY the provided result rows to answer the user's question precisely.\n"
        "Start with 'Answer: ...' on the first line. Then add up to 5 short supporting bullets.\n"
        "If the rows are insufficient, say exactly what data/filters are missing. No code blocks. No SQL."
    )
    usr = f"QUESTION: {user_query}\n\nDATA SUMMARY: {json.dumps(df_summary)}\n\nRESULT ROWS (CSV head):\n{df_csv_head}"
    return [{"role":"system","content":sys},{"role":"user","content":usr}]

# -----------------------------------------------------------------------------
# Column guard for MG (maps aliases → exact quoted MG column names)
# -----------------------------------------------------------------------------
SQL_RESERVED = {
    "select","from","where","group","by","order","limit","having","join","on",
    "and","or","not","as","count","sum","avg","min","max","distinct","left","right",
    "inner","outer","union","all","case","when","then","else","end","desc","asc"
}
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())
def _quoted(col: str) -> str:
    return f'"{col}"' if re.search(r"\W", col) else col

def build_mg_synonym_map(mg_df: pd.DataFrame) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for col in mg_df.columns:
        q = _quoted(col)
        lc = col.lower()
        keys = {
            _norm(col),
            lc,
            lc.replace(" ", "_"),
            lc.replace(" ", ""),
        }
        # semantic aliases
        if "payout" in lc:
            keys |= {"payout", "totalpayout", "total_payout"}
        if "eligib" in lc:
            keys |= {"eligible", "mgeligible", "mg_eligible"}
        if "user" in lc or "id_user" in lc:
            keys |= {"user", "userid", "id_user"}
        if "vendor" in lc:
            keys |= {"vendor", "vendorname", "vendor_name"}
        for k in keys:
            m[k] = q
    return m

def apply_mg_synonyms(sql: str, mg_df: pd.DataFrame) -> str:
    """Rewrite identifiers to the exact MG column names (quoted) when mg is referenced."""
    if not sql or mg_df is None or mg_df.empty:
        return sql
    if not re.search(r"\bmg\b", sql, flags=re.I):
        return sql

    syn = build_mg_synonym_map(mg_df)

    # 1) rewrite mg.<identifier>
    def repl_qualified(m: re.Match) -> str:
        ident = m.group(1)
        key = _norm(ident)
        rep = syn.get(key)
        return f'mg.{rep}' if rep else m.group(0)

    fixed = re.sub(r'\bmg\.([A-Za-z_][A-Za-z0-9_]*)\b', repl_qualified, sql)

    # 2) rewrite bare identifiers (avoid SQL keywords)
    def repl_bare(m: re.Match) -> str:
        ident = m.group(1)
        if ident.lower() in SQL_RESERVED:
            return ident
        key = _norm(ident)
        rep = syn.get(key)
        return rep if rep else ident

    fixed = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', repl_bare, fixed)

    return fixed

# -----------------------------------------------------------------------------
# Web flow
# -----------------------------------------------------------------------------
def fetch_wikipedia_snippet(q: str) -> str:
    try:
        term = q.strip().split("\n")[0][:150]
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(term)}"
        r = requests.get(url, timeout=6)
        if r.ok:
            return r.json().get("extract") or ""
    except Exception as e:
        logging.info("wiki fetch failed: %s", e)
    return ""

def run_web_flow(q: str) -> Dict:
    snippet = fetch_wikipedia_snippet(q)
    if GROQ_API_KEY:
        ctx = f"Wikipedia says: {snippet}" if snippet else "No web snippet available."
        try:
            answer = llm_chat(
                [
                    {"role":"system","content":"Be concise and factual. Ground answer in context if present."},
                    {"role":"user","content":f"Question: {q}\n\nContext: {ctx}\n\nAnswer briefly:"}
                ], temperature=0.2
            )
            return {"ok": True, "mode": "web", "reply": answer}
        except Exception as e:
            logging.info("Groq web answer failed: %s", e)
    return {"ok": True, "mode": "web", "reply": snippet or "I couldn't fetch a web snippet, but I can analyze your data."}

# -----------------------------------------------------------------------------
# Data flow (Groq → SQL → DuckDB, with column guard & repair)
# -----------------------------------------------------------------------------
def run_data_flow(question: str, analysis_style: bool = False, previous_sql: Optional[str]=None) -> Dict:
    global last_result_df, last_result_sql, last_result_query
    inv, ords, mg = load_data_for_routes()
    schema = build_schema_description(inv, ords, mg)

    if not GROQ_API_KEY:
        sql = "SELECT * FROM orders LIMIT 50"
        try: df = execute_sql(sql, inv, ords, mg)
        except Exception: df = pd.DataFrame()
        last_result_df, last_result_sql, last_result_query = df, sql, question
        return {"reply": f"(fallback) Showing first {len(df)} rows from orders.", "sql": sql, "preview": trim_text(df_preview_string(df), 2000)}

    # 1) ask Groq for SQL (with optional previous_sql context)
    try:
        sql_text = llm_chat(sql_generation_prompt(question, schema, analysis_style, previous_sql), temperature=0.0)
    except Exception as e:
        return {"reply": f"SQL generation error: {e}", "sql": None, "preview": None}
    sql = extract_sql(sql_text)

    # Column guard for mg synonyms before executing
    sql_try = apply_mg_synonyms(sql, mg)

    # 2) execute (if fail → guard again → repair with Groq)
    try:
        df = execute_sql(sql_try, inv, ords, mg)
    except Exception as ex:
        # one more synonym pass in case model returned new casing
        sql_try2 = apply_mg_synonyms(sql_try, mg)
        try:
            df = execute_sql(sql_try2, inv, ords, mg)
            sql_try = sql_try2
        except Exception as ex2:
            # ask Groq to repair with explicit error
            try:
                repaired = llm_chat(sql_repair_prompt(question, schema, sql_try, str(ex2)), temperature=0.0)
                sql_try = extract_sql(repaired)
                sql_try = apply_mg_synonyms(sql_try, mg)
                df = execute_sql(sql_try, inv, ords, mg)
            except Exception as ex3:
                mg_cols  = ", ".join([f'"{c}"' for c in mg.columns]) or "(none)"
                ord_cols = ", ".join([f'"{c}"' for c in ords.columns]) or "(none)"
                inv_cols = ", ".join([f'"{c}"' for c in inv.columns]) or "(none)"
                msg = (
                    f"SQL failed: {ex3}\n\n"
                    f"Tip: Use exact column names from the schema.\n"
                    f"- mg columns:  {mg_cols}\n- orders:      {ord_cols}\n- inventory:   {inv_cols}"
                )
                return {"reply": msg, "sql": sql_try, "preview": None}

    last_result_df, last_result_sql, last_result_query = df, sql_try, question
    return {
        "reply": f"Executed SQL. Showing first {min(50, len(df))} rows.",
        "sql": sql_try,
        "preview": trim_text(df_preview_string(df), 2000),
    }

# -----------------------------------------------------------------------------
# Analysis flow (Groq → analysis SQL → Groq answers from rows; carries follow-up context)
# -----------------------------------------------------------------------------
def run_analysis_flow(user_query: str) -> Dict:
    """
    Groq for analysis:
      1) generate analysis SQL (aggregations / group-bys) and run it,
      2) compose chat-style answer from the actual rows.
    For single-cell results, we still call Groq to make it read naturally,
    and we do NOT return the SQL in the payload.
    """
    global last_result_df, last_result_sql, last_result_query

    ql = (user_query or "").lower()
    wants_reuse = any(p in ql for p in REUSE_PHRASES)
    prev_sql = last_result_sql if wants_reuse else None

    # Step 1: get analysis-style dataset (reuse previous SQL if follow-up refers to "these/previous/above")
    data_res = run_data_flow(user_query, analysis_style=True, previous_sql=prev_sql)
    if not data_res.get("preview"):
        return data_res  # bubble up errors from data flow
    df, sql_used = last_result_df.copy(), last_result_sql

    # ---- Single-cell path: call Groq to make it conversational; omit SQL in response ----
    try:
        if isinstance(df, pd.DataFrame) and df.shape == (1, 1):
            # Build a tiny CSV so the LLM has the actual value
            csv_head = df_as_csv_snippet(df, max_rows=1, max_chars=200)
            if GROQ_API_KEY:
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert analyst. Turn the single numeric result into a concise, "
                            "human-friendly answer that directly addresses the user's question.\n"
                            "- Start with 'Answer: ...' on the first line using the exact value.\n"
                            "- Add 1–3 short supporting bullets (optional), e.g., brief context or implication.\n"
                            "- Do NOT include SQL or code. Be precise and neutral."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"QUESTION: {user_query}\n\nRESULT (CSV):\n{csv_head}",
                    },
                ]
                try:
                    answer = llm_chat(prompt, temperature=0.0)
                except Exception:
                    # If Groq hiccups, fall back to the raw value
                    answer = f"Answer: {df.iloc[0, 0]}"
            else:
                # No Groq key: deterministic fallback (still omitting SQL)
                answer = f"Answer: {df.iloc[0, 0]}"

            return {
                "reply": answer,
                "sql": None,  # <-- suppress query in UI per your request
                "preview": trim_text(df_preview_string(df), 2000),
            }
    except Exception:
        # If anything odd happens, proceed to the general multi-row path below
        pass

    # ---- Multi-row path (unchanged): ask Groq to compose from actual rows ----
    if not GROQ_API_KEY:
        return {
            "reply": "Analysis fallback (no Groq). See preview and compute manually.",
            "sql": sql_used,
            "preview": trim_text(df_preview_string(df), 2000),
        }

    summary = pre_analyze(df)
    csv_head = df_as_csv_snippet(df)
    try:
        answer = llm_chat(build_answer_prompt(user_query, summary, csv_head), temperature=0.0)
        return {
            "reply": answer,
            "sql": sql_used,  # keep SQL for multi-row results as before
            "preview": trim_text(df_preview_string(df), 2000),
        }
    except Exception as ex:
        logging.info("Groq analysis answer failed: %s", ex)
        return {
            "reply": f"Analysis error. Rows derived from: {sql_used}",
            "sql": sql_used,
            "preview": trim_text(df_preview_string(df), 2000),
        }
# -----------------------------------------------------------------------------
# /schema (for Data → Extract: list columns & simple types)
# -----------------------------------------------------------------------------
def _simple_dtype(series: pd.Series) -> str:
    try:
        if is_datetime64_any_dtype(series): return "datetime"
        if is_bool_dtype(series):           return "numeric"
        if is_numeric_dtype(series):        return "numeric"
        # quick heuristic from name
        if "date" in series.name.lower() or "time" in series.name.lower():
            try:
                pd.to_datetime(series.sample(min(50, len(series))), errors="raise")
                return "datetime"
            except Exception:
                pass
        return "string"
    except Exception:
        return "string"

def _get_dataset_frame(name: str) -> Tuple[str, pd.DataFrame]:
    inv, ords, mg = load_data_for_routes()
    key = (name or "").strip().lower()
    if key in ("order","orders"): return "orders", ords
    if key in ("inventory","inventory management","inventory_management"): return "inventory", inv
    if key in ("minimum guarantee","mg","minimum_guarantee"): return "mg", mg
    return key, pd.DataFrame()

@app.get("/schema")
def get_schema():
    dataset = (request.args.get("dataset") or "").strip()
    if not dataset:
        return safe_json({"ok": False, "error": "Missing dataset param"}, 400)
    key, df = _get_dataset_frame(dataset)
    if df is None or df.empty:
        return safe_json({"ok": False, "dataset": key, "error": "Dataset not found or empty."}, 404)
    cols = [{"name": c, "type": _simple_dtype(df[c])} for c in df.columns]
    return safe_json({"ok": True, "dataset": key, "columns": cols})

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index(): return render_template("index.html")

@app.get("/health")
def health(): return "ok", 200

@app.post("/ask")
def ask():
    payload = request.get_json(force=True) or {}
    q    = (payload.get("q") or payload.get("query") or "").strip()
    mode = (payload.get("mode") or "web").lower()

    if not q:
        return safe_json({"ok": True, "mode": mode, "reply": "Please type a question."})

    if mode == "analysis": return safe_json(run_analysis_flow(q))
    if mode == "data":     return safe_json(run_data_flow(q, analysis_style=False))
    return safe_json(run_web_flow(q))  # web

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=True)
