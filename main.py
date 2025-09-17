# main.py — Noon AI (Groq for Web/Data/Analysis) + /schema + robust MG synonyms
from __future__ import annotations

import os, re, json, logging
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from column_aliases import COLUMN_ALIASES
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
)
from analysis_planner import plan_with_groq

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

def build_answer_prompt(user_query: str, df_summary: dict, df_csv_head: str, column_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Ask Groq to write a chat-style answer using ONLY the provided rows.
    This avoids 'how to filter...' instructions and forces a direct, conversational reply.
    """
    cols_txt = ", ".join(column_names or [])
    sys = (
        "You are a helpful data analyst. Use ONLY the table of rows provided to answer the user's question.\n"
        "STYLE:\n"
        "- Conversational and concise, like ChatGPT.\n"
        "- First line MUST start with: 'Answer: ...' giving the result directly.\n"
        "- Then up to 3 short bullets with useful context (top contributors, quick ratios, brief takeaway).\n"
        "DO NOT:\n"
        "- Do not describe how to filter or what you would do.\n"
        "- Do not restate column definitions, dataset size, or say 'data is available'.\n"
        "- Do not include SQL, code, or markdown code fences.\n"
        "IF INSUFFICIENT:\n"
        "- If the rows are insufficient, say: 'Answer: I couldn't compute this from the provided rows' and name the specific missing columns needed.\n"
    )
    usr = (
        f"QUESTION: {user_query}\n\n"
        f"COLUMNS AVAILABLE: {cols_txt}\n\n"
        f"DATA SUMMARY: {json.dumps(df_summary)}\n\n"
        f"RESULT ROWS (CSV head):\n{df_csv_head}"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

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

def generate_alias_map(df: pd.DataFrame, dataset_key: str) -> Dict[str, str]:
    """
    For a dataframe + dataset key ('mg' | 'orders' | 'inventory'), produce a map:
      normalized_alias -> exact QUOTED column name
    Includes automatic variants (snake/spaceless/lower) + curated COLUMN_ALIASES.
    """
    curated = COLUMN_ALIASES.get(dataset_key, {})
    amap: Dict[str, str] = {}

    for col in df.columns:
        q = _quoted(col)
        lc = col.lower()

        # automatic variants for the actual column name
        auto = {
            _norm(col),
            lc,
            lc.replace(" ", "_"),
            lc.replace(" ", ""),
        }
        for a in auto:
            amap[a] = q

        # curated per-column aliases
        for a in curated.get(col, []):
            amap[_norm(a)] = q
            amap[a.lower()] = q
            amap[a.replace(" ", "_").lower()] = q
            amap[a.replace(" ", "").lower()] = q

        # a few dataset-specific generic hints
        if dataset_key == "mg":
            if "payout" in lc: amap.update({"payout": q, "totalpayout": q, "total_payout": q})
            if "eligib" in lc: amap.update({"eligible": q, "mgeligible": q, "mg_eligible": q})
            if "user" in lc:   amap.update({"user": q, "userid": q, "id_user": q})
            if "vendor" in lc: amap.update({"vendor": q, "vendorname": q, "vendor_name": q})
        if dataset_key == "orders" and "price" in lc:
            amap.update({"price": q, "amount": q})

    return amap
    
def build_mg_synonym_map(mg_df: pd.DataFrame) -> Dict[str, str]:
    # REPLACE body with this one-liner
    return generate_alias_map(mg_df, "mg")

def apply_mg_synonyms(sql: str, mg_df: pd.DataFrame) -> str:
    """
    Rewrite identifiers to the exact MG column names (quoted) when mg is referenced.
    Now handles:
      - qualified tokens (mg.col)
      - bare single-word tokens (col)
      - multi-word / symbol columns like: Total Attendance, FND+NDR+penalty Amount
        → mg."Total Attendance", mg."FND+NDR+penalty Amount"
    """
    if not sql or mg_df is None or mg_df.empty:
        return sql
    # Only attempt if the query references mg at all
    if not re.search(r"\bmg\b", sql, flags=re.I):
        return sql

    syn = build_mg_synonym_map(mg_df)
    fixed = sql

    # (A) FIRST: handle multi-word / symbol-containing columns by direct phrase replacement
    #     Replace unquoted occurrences with the exact quoted column name (optionally qualified).
    multi_cols = [c for c in mg_df.columns if re.search(r"\W", c)]  # spaces or punctuation
    for col in multi_cols:
        col_esc = re.escape(col)

        # mg.<phrase>  →  mg."<phrase>"
        pat_qual = re.compile(rf'\bmg\s*\.\s*{col_esc}\b', flags=re.I)
        fixed = pat_qual.sub(f'mg.{_quoted(col)}', fixed)

        # bare <phrase> (not already quoted) →  "<phrase>"
        # - negative look-behind and ahead to avoid already quoted instances
        pat_bare = re.compile(rf'(?<!")\b{col_esc}\b(?!")', flags=re.I)
        fixed = pat_bare.sub(_quoted(col), fixed)

    # (B) THEN: mg.<identifier> (single-word identifiers)
    def repl_qualified(m: re.Match) -> str:
        ident = m.group(1)
        key = _norm(ident)
        rep = syn.get(key)
        return f'mg.{rep}' if rep else m.group(0)

    fixed = re.sub(r'\bmg\.([A-Za-z_][A-Za-z0-9_]*)\b', repl_qualified, fixed)

    # (C) FINALLY: bare identifiers (single words), avoid SQL keywords
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
    Analysis flow with Plan → Execute → Answer:
    1) Try planner (Groq → JSON plan → compiled SQL).
    2) If planner confident and no clarification needed, run compiled SQL.
    3) Otherwise, fall back to existing Groq→SQL path (run_data_flow with analysis_style=True).
    """
    global last_result_df, last_result_sql, last_result_query

    # --- Try planner first ---
    inv, ords, mg = load_data_for_routes()
    try:
        plan_res = plan_with_groq(user_query, inv, ords, mg) if GROQ_API_KEY else {"ok": False, "error": "no_groq"}
    except Exception as e:
        plan_res = {"ok": False, "error": f"planner error: {e}"}

    if plan_res.get("ok") and not plan_res.get("needs_clarification") and plan_res.get("sql"):
        # Execute compiled SQL deterministically
        sql_used = plan_res["sql"]
        try:
            df = execute_sql(sql_used, inv, ords, mg)
        except Exception as ex:
            # If compiled SQL still fails for any reason, fall back to old path
            return run_data_flow(user_query, analysis_style=True, previous_sql=None)

        last_result_df, last_result_sql, last_result_query = df, sql_used, user_query

        # Single-cell shortcut (keep current behavior)
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

        # Multi-row: compose answer with existing prompt
        summary  = pre_analyze(df)
        csv_head = df_as_csv_snippet(df)
        try:
            answer = llm_chat(build_answer_prompt(user_query, summary, csv_head), temperature=0.0)
            return {"reply": answer, "sql": sql_used, "preview": trim_text(df_preview_string(df), 2000)}
        except Exception:
            return {"reply": "Analysis ready. See the first rows below.", "sql": sql_used, "preview": trim_text(df_preview_string(df), 2000)}

    # --- Clarification from planner ---
    if plan_res.get("ok") and plan_res.get("needs_clarification"):
        return {"reply": plan_res.get("clarify") or "Please clarify your request.", "sql": None, "preview": None}

    # --- Fallback: use your existing Groq→SQL analysis path ---
    # Keep previous-follow-up handling exactly as before
    ql = (user_query or "").lower()
    wants_reuse = any(p in ql for p in REUSE_PHRASES)
    prev_sql = last_result_sql if wants_reuse else None

    data_res = run_data_flow(user_query, analysis_style=True, previous_sql=prev_sql)
    if not data_res.get("preview"):
        return data_res

    df, sql_used = last_result_df.copy(), last_result_sql
    try:
        if isinstance(df, pd.DataFrame) and df.shape == (1,1)):
            val = df.iloc[0,0]
            return {"reply": f"Answer: {val}\n\n(derived from: {sql_used})", "sql": sql_used, "preview": trim_text(df_preview_string(df), 2000)}
    except Exception:
        pass

    if not GROQ_API_KEY:
        return {"reply":"Analysis fallback (no Groq). See preview and compute manually.", "sql": sql_used, "preview": trim_text(df_preview_string(df), 2000)}

    summary  = pre_analyze(df)
    csv_head = df_as_csv_snippet(df)
    try:
        answer = llm_chat(build_answer_prompt(user_query, summary, csv_head), temperature=0.0)
        return {"reply": answer, "sql": sql_used, "preview": trim_text(df_preview_string(df), 2000)}
    except Exception as ex:
        return {"reply": f"Analysis error. Rows derived from: {sql_used}", "sql": sql_used, "preview": trim_text(df_preview_string(df), 2000)}
        
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

@app.get("/schema/aliases")
def get_aliases():
    dataset = (request.args.get("dataset") or "").strip().lower()
    inv, ords, mg = load_data_for_routes()

    if dataset in ("order", "orders"):
        key, df = "orders", ords
    elif dataset in ("inventory", "inventory management", "inventory_management"):
        key, df = "inventory", inv
    elif dataset in ("mg", "minimum guarantee", "minimum_guarantee"):
        key, df = "mg", mg
    else:
        return safe_json({"ok": False, "error": "Unknown dataset"}, 400)

    amap = generate_alias_map(df, key)
    # invert map for human-friendly view: canonical col -> aliases
    inverted: Dict[str, List[str]] = {}
    for alias, quoted_col in amap.items():
        col = quoted_col.strip('"')
        inverted.setdefault(col, []).append(alias)
    for col in inverted:
        inverted[col] = sorted(list(set(inverted[col])))

    return safe_json({"ok": True, "dataset": key, "aliases": inverted})
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=True)
