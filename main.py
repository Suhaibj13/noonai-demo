# main.py
from flask import Flask, render_template, request, jsonify
import os, re, difflib, duckdb, requests, math, logging, traceback
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from werkzeug.exceptions import HTTPException

# ===============================
# Env & clients
# ===============================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
BASE_DIR = Path(__file__).resolve().parent  # /opt/render/project/src (Render)

# ===============================
# JSON helpers & error handling
# ===============================
def _fix_nans(obj):
    if isinstance(obj, float) and math.isnan(obj): return None
    if isinstance(obj, dict):  return {k: _fix_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_fix_nans(v) for v in obj]
    return obj

def safe_json(data, status=200):
    return jsonify(_fix_nans(data)), status

def _wants_json() -> bool:
    try:
        return request.path in ("/ask", "/run_step") or \
               "application/json" in (request.headers.get("Accept") or "")
    except Exception:
        return False

@app.errorhandler(404)
def _err_404(e):
    if _wants_json(): return safe_json({"ok": False, "error": "Not Found", "code": 404}, 404)
    return ("Not Found", 404)

@app.errorhandler(405)
def _err_405(e):
    if _wants_json(): return safe_json({"ok": False, "error": "Method Not Allowed", "code": 405}, 405)
    return ("Method Not Allowed", 405)

@app.errorhandler(500)
def _err_500(e):
    logging.exception("Internal Server Error")
    if _wants_json(): return safe_json({"ok": False, "error": "Server Error", "code": 500}, 500)
    return ("Server Error", 500)

# Catch-all: never send HTML to chat endpoints
@app.errorhandler(Exception)
def _err_any(e):
    code = getattr(e, "code", 500) if isinstance(e, HTTPException) else 500
    if _wants_json():
        logging.exception("Unhandled exception")
        return safe_json({"ok": False, "error": str(e), "code": code}, code)
    raise e

# ===============================
# Conversation cache
# ===============================
conversation_history = []
MAX_TURNS = 10

last_result_df = None
last_result_sql = None
last_result_query = None

def trim_text(t: str, limit=500) -> str:
    t = str(t or "").strip()
    return t if len(t) <= limit else t[:limit] + "…"

def get_recent_history():
    msgs = []
    for m in conversation_history[-MAX_TURNS * 2:]:
        msgs.append({"role": m["role"], "content": trim_text(m["content"], 500)})
    return msgs

def push_history(user_msg: str, assistant_msg: str):
    conversation_history.append({"role": "user", "content": user_msg})
    conversation_history.append({"role": "assistant", "content": assistant_msg})
    if len(conversation_history) > MAX_TURNS * 2:
        del conversation_history[: len(conversation_history) - MAX_TURNS * 2]

# ===============================
# Schemas, aliases & synonyms
# ===============================
SCHEMA_CONFIG = {
    "orders": {
        "columns": [
            "id","order_id","user_id","product_id","inventory_item_id",
            "status","created_at","shipped_at","delivered_at","returned_at","sale_price"
        ],
        "dtypes": {
            "id":"Int64","order_id":"Int64","user_id":"Int64","product_id":"Int64",
            "inventory_item_id":"Int64","status":"string",
            "created_at":"datetime64[ns]","shipped_at":"datetime64[ns]",
            "delivered_at":"datetime64[ns]","returned_at":"datetime64[ns]",
            "sale_price":"Float64"
        },
        "aliases": {"price":"sale_price","customer_id":"user_id","buyer_id":"user_id","client_id":"user_id"}
    },
    "inventory": {
        "columns": [
            "id","product_id","created_at","sold_at","cost","product_category","product_name",
            "product_brand","product_retail_price","product_department","product_sku",
            "product_distribution_center_id"
        ],
        "dtypes": {
            "id":"Int64","product_id":"Int64","created_at":"datetime64[ns]","sold_at":"datetime64[ns]",
            "cost":"Float64","product_category":"string","product_name":"string","product_brand":"string",
            "product_retail_price":"Float64","product_department":"string","product_sku":"string",
            "product_distribution_center_id":"Int64"
        },
        "aliases": {"unit_cost":"cost","buy_price":"cost"}
    },
    "mg": {
        "columns": [
            "month","id_user","name","fleet","city","vendor_name","rate","joining_date",
            "total_calendar_days","attendance","total_attendance","perfect_attendance","pa_needed",
            "mg_eligible","total_delivered_month","monthly_mg","eligible_mg","mg_month",
            "payout","mg_amount","total_payout","mot","final_mg_check","fnd_ndr_penalty_amount",
            "da_level_final_amount"
        ],
        "dtypes": {
            "month":"string","id_user":"Int64","name":"string","fleet":"string","city":"string",
            "vendor_name":"string","rate":"Float64","joining_date":"datetime64[ns]",
            "total_calendar_days":"Int64","attendance":"Int64","total_attendance":"Int64",
            "perfect_attendance":"Int64","pa_needed":"Int64","mg_eligible":"Int64",
            "total_delivered_month":"Int64","monthly_mg":"Float64","eligible_mg":"Float64",
            "mg_month":"Int64","payout":"Float64","mg_amount":"Float64","total_payout":"Float64",
            "mot":"string","final_mg_check":"string","fnd_ndr_penalty_amount":"Float64",
            "da_level_final_amount":"Float64"
        },
        "aliases": {"vendor":"vendor_name","final_amount":"da_level_final_amount","penalty":"fnd_ndr_penalty_amount"}
    }
}

COLUMN_SYNONYMS = {
    "orders": {"user_id":["customer","customer_id","buyer_id","client_id"], "sale_price":["price","unit_price","amount"], "order_id":["oid","sales_order_id"]},
    "inventory": {"product_id":["pid","item_id"], "cost":["unit_cost","buy_price"]},
    "mg": {"vendor_name":["vendor"], "fnd_ndr_penalty_amount":["penalty","fnd_ndr_penalty"], "da_level_final_amount":["final_amount","da_final_amount"]}
}

# ===============================
# Schema helpers
# ===============================
def _apply_aliases(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    to_rename, lower_cols = {}, {c.lower(): c for c in df.columns}
    for alias, canonical in aliases.items():
        if alias.lower() in lower_cols:
            to_rename[lower_cols[alias.lower()]] = canonical
    return df.rename(columns=to_rename) if to_rename else df

def _enforce_schema(df: pd.DataFrame, table: str) -> pd.DataFrame:
    spec = SCHEMA_CONFIG[table]
    for col in spec["columns"]:
        if col not in df.columns:
            df[col] = pd.NA
    for col, dt in spec["dtypes"].items():
        if col not in df.columns: continue
        try:
            if dt == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
            elif dt in ("Int64","Float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dt)
            elif dt == "string":
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype(dt)
        except Exception:
            pass
    return df[spec["columns"]]

def _read_first(path_list):
    for p in path_list:
        p_abs = (BASE_DIR / p)
        if p_abs.exists():
            return pd.read_csv(p_abs)
    return None

# ===============================
# Data loading (FIXED)
# ===============================
def load_data():
    # Read from project root (Render/Linux is case-sensitive)
    inv = pd.read_csv(BASE_DIR / "inventory.csv")
    ords = pd.read_csv(BASE_DIR / "orders.csv")
    mg_raw = _read_first(["minimum_guarantee.csv", "Minimum_Guarantee.csv", "mg.csv"])

    # Normalize columns
    inv.columns  = [c.strip().lower() for c in inv.columns]
    ords.columns = [c.strip().lower() for c in ords.columns]
    if mg_raw is not None:
        mg_raw.columns = [c.strip().lower().replace(" ", "_") for c in mg_raw.columns]

    # Apply aliases
    inv  = _apply_aliases(inv,  SCHEMA_CONFIG["inventory"]["aliases"])
    ords = _apply_aliases(ords, SCHEMA_CONFIG["orders"]["aliases"])
    if mg_raw is not None:
        mg_raw = _apply_aliases(mg_raw, SCHEMA_CONFIG["mg"]["aliases"])

    # Enforce schemas
    inv = _enforce_schema(inv, "inventory")
    ords = _enforce_schema(ords, "orders")
    mg  = _enforce_schema(mg_raw, "mg") if mg_raw is not None else pd.DataFrame(columns=SCHEMA_CONFIG["mg"]["columns"])

    return inv, ords, mg

# ===============================
# SQL helpers & rules
# ===============================
def normalize_time_literals(sql: str) -> str:
    sql = re.sub(r"\bCURRENT_TIMESTAMP\b", "CAST(CURRENT_TIMESTAMP AS TIMESTAMP)", sql, flags=re.I)
    sql = re.sub(r"\bNOW\(\)", "CAST(NOW() AS TIMESTAMP)", sql, flags=re.I)
    sql = re.sub(r"\bCURRENT_DATE\b", "CAST(CURRENT_DATE AS TIMESTAMP)", sql, flags=re.I)
    return sql

def validate_sql_columns(sql: str, inv_cols: list, ord_cols: list, mg_cols: list):
    missing = []
    inv_lower = [c.lower() for c in inv_cols]
    ord_lower = [c.lower() for c in ord_cols]
    mg_lower  = [c.lower() for c in mg_cols]
    for alias, cols in (("i", inv_lower), ("inventory", inv_lower),
                        ("o", ord_lower), ("orders", ord_lower),
                        ("m", mg_lower), ("mg", mg_lower)):
        for m in re.findall(rf"\b{alias}\.(\w+)\b", sql, flags=re.I):
            if m.lower() not in cols:
                t = "inventory" if alias in ("i","inventory") else "orders" if alias in ("o","orders") else "mg"
                missing.append(f"{t}.{m}")
    return (len(missing) == 0, missing)

def suggest_similar_columns(missing_cols: list, table_cols: dict, tables_to_dfs: dict, synonyms_map: dict, max_suggestions=5):
    suggestions = {}
    def first_non_null(df, col):
        try:
            s = df[col].dropna()
            return "" if s.empty else str(s.iloc[0])[:80]
        except Exception:
            return ""
    for miss in missing_cols:
        table, col = (miss.split(".", 1) if "." in miss else ("orders", miss))
        table, col = table.lower().strip(), col.lower().strip()
        cands = [c for c in table_cols.get(table, [])]
        df = tables_to_dfs.get(table)
        syn_hits = []
        for canon, alts in synonyms_map.get(table, {}).items():
            if canon in cands and (col == canon or col in [a.lower() for a in alts]):
                syn_hits.append(canon)
        sub_hits = [c for c in cands if col in c or c in col]
        fuzzy_hits = difflib.get_close_matches(col, cands, n=max_suggestions, cutoff=0.6)
        merged = []
        for arr in (syn_hits, sub_hits, fuzzy_hits):
            for x in arr:
                if x not in merged:
                    merged.append(x)
        merged = merged[:max_suggestions]
        enriched = [{"col": c, "sample": first_non_null(df, c) if df is not None else ""} for c in merged]
        suggestions[miss] = enriched
    return suggestions

def rule_orders_by_value_desc(user_query: str) -> str | None:
    q = (user_query or "").lower()
    if "orders" not in q: return None
    if "value" in q and any(k in q for k in ("decreas","desc","descending","highest")):
        m = re.search(r"\b(top|first|limit|give me|show me)\s*(\d+)", q)
        n = int(m.group(2)) if m else 500
        return f"SELECT * FROM orders ORDER BY sale_price DESC LIMIT {n}"
    return None

def rule_top_customers_sql(user_query: str) -> str | None:
    q = (user_query or "").lower()
    if not (("month" in q or "monthly" in q) and any(x in q for x in ("customer","customers","user_id"))):
        return None
    m = re.search(r"\btop\s+(\d+)\b", q)
    top_n = int(m.group(1)) if m else 5
    return f"""
WITH customer_totals AS (
  SELECT o.user_id, SUM(COALESCE(o.sale_price, 0)) AS total_revenue
  FROM orders o
  WHERE o.user_id IS NOT NULL
  GROUP BY 1
),
top_customers AS (
  SELECT user_id
  FROM customer_totals
  ORDER BY total_revenue DESC
  LIMIT {top_n}
),
monthly AS (
  SELECT DATE_TRUNC('month', o.created_at) AS year_month, o.user_id,
         SUM(COALESCE(o.sale_price, 0)) AS sale_price
  FROM orders o
  WHERE o.user_id IN (SELECT user_id FROM top_customers)
    AND o.created_at IS NOT NULL
  GROUP BY 1,2
)
SELECT year_month, user_id, sale_price
FROM monthly
ORDER BY year_month, user_id
""".strip()

def _parse_top_n(query: str, default_n=5) -> int:
    m = re.search(r"\btop\s+(\d+)\b", query.lower())
    return int(m.group(1)) if m else default_n

def _parse_threshold(query: str, default_th=10) -> int:
    q = query.lower()
    m = re.search(r"(threshold|below|less than)\s+(\d+)", q)
    return int(m.group(2)) if m else default_th

def rule_inventory_sql(user_query: str) -> str | None:
    q = (user_query or "").lower()
    if not any(k in q for k in ["inventory","stockout","stock out","overstock","on hand","reorder","slow-moving","slow moving"]):
        return None
    want_monthly = ("month" in q or "monthly" in q)
    want_slow = any(k in q for k in ["slow","overstock"])
    want_reorder = any(k in q for k in ["reorder","low stock","low inventory","stockout","stock out"])
    want_top_sold = any(k in q for k in ["most sold","top sold","best seller","bestseller","most selling","top sales","top sold items"])
    top_n = _parse_top_n(q, 5); th = _parse_threshold(q, 10)

    base_ctes = """
WITH sold AS (
  SELECT o.product_id, COUNT(*) AS units_sold
  FROM orders o
  WHERE o.product_id IS NOT NULL
  GROUP BY 1
),
stock AS (
  SELECT i.product_id,
         SUM(CASE WHEN i.sold_at IS NULL THEN 1 ELSE 0 END) AS on_hand
  FROM inventory i
  WHERE i.product_id IS NOT NULL
  GROUP BY 1
),
dim AS (
  SELECT
    coalesce(s.product_id, st.product_id) AS product_id,
    coalesce(st.on_hand, 0) AS on_hand,
    coalesce(s.units_sold, 0) AS units_sold,
    max(inv.product_name) AS product_name,
    max(inv.product_brand) AS product_brand,
    max(inv.product_category) AS product_category
  FROM sold s
  FULL OUTER JOIN stock st ON s.product_id = st.product_id
  LEFT JOIN inventory inv ON inv.product_id = coalesce(s.product_id, st.product_id)
  GROUP BY 1,2,3
)
"""
    if want_top_sold and not want_monthly:
        return f"{base_ctes}\nSELECT product_id, product_name, product_brand, product_category, units_sold, on_hand FROM dim ORDER BY units_sold DESC LIMIT {top_n}"
    if want_reorder and not want_monthly:
        return f"{base_ctes}\nSELECT product_id, product_name, product_brand, product_category, units_sold, on_hand FROM dim WHERE on_hand < {th} ORDER BY on_hand ASC, units_sold DESC LIMIT 100"
    if want_slow and not want_monthly:
        return f"{base_ctes}\nSELECT product_id, product_name, product_brand, product_category, units_sold, on_hand FROM dim WHERE on_hand >= {th} AND units_sold <= 1 ORDER BY on_hand DESC LIMIT 100"
    if want_monthly:
        return """
WITH monthly AS (
  SELECT DATE_TRUNC('month', o.created_at) AS year_month, o.product_id, COUNT(*) AS units_sold
  FROM orders o
  WHERE o.product_id IS NOT NULL AND o.created_at IS NOT NULL
  GROUP BY 1,2
),
stock AS (
  SELECT i.product_id, SUM(CASE WHEN i.sold_at IS NULL THEN 1 ELSE 0 END) AS on_hand
  FROM inventory i
  WHERE i.product_id IS NOT NULL
  GROUP BY 1
),
dim AS (
  SELECT m.year_month, m.product_id, m.units_sold, coalesce(st.on_hand,0) AS on_hand,
         max(inv.product_name) AS product_name, max(inv.product_brand) AS product_brand,
         max(inv.product_category) AS product_category
  FROM monthly m
  LEFT JOIN stock st ON st.product_id = m.product_id
  LEFT JOIN inventory inv ON inv.product_id = m.product_id
  GROUP BY 1,2,3,4
)
SELECT year_month, product_id, product_name, product_brand, product_category, units_sold, on_hand
FROM dim
ORDER BY year_month DESC, units_sold DESC
""".strip()
    return f"{base_ctes}\nSELECT product_id, product_name, product_brand, product_category, units_sold, on_hand FROM dim ORDER BY units_sold DESC, on_hand ASC LIMIT 100"

def rule_mg_sql(user_query: str) -> str | None:
    q = (user_query or "").lower()
    if "minimum guarantee" in q or "mg" in q or "logistics" in q:
        return """
SELECT month, city, vendor_name, fleet, id_user, name,
       rate, attendance, perfect_attendance, mg_eligible,
       total_delivered_month, monthly_mg, eligible_mg,
       fnd_ndr_penalty_amount, da_level_final_amount, total_payout
FROM mg
ORDER BY total_payout DESC NULLS LAST
LIMIT 200
""".strip()
    return None

# ===============================
# LLM SQL prompts
# ===============================
def build_schema_manifest(inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame) -> str:
    def col_line(df, col):
        s = df[col].dropna(); sample = "" if s.empty else f" sample={repr(str(s.iloc[0])[:40])}"
        return f"{col} {str(df[col].dtype)} nullable={df[col].isna().any()}{sample}"
    lines = ["Tables and columns (DuckDB dtypes):"]
    lines.append("[inventory]");  lines += ["  - " + col_line(inv, c) for c in SCHEMA_CONFIG["inventory"]["columns"] if c in inv.columns]
    lines.append("[orders]");     lines += ["  - " + col_line(ords, c) for c in SCHEMA_CONFIG["orders"]["columns"] if c in ords.columns]
    lines.append("[mg]");         lines += ["  - " + col_line(mg, c) for c in SCHEMA_CONFIG["mg"]["columns"] if c in mg.columns]
    return "\n".join(lines)

SQL_SYSTEM_PROMPT = """
You are SAGE-SQL for DuckDB.
Return EXACTLY ONE valid SQL SELECT statement and nothing else.
Use ONLY the provided tables and columns. No DDL/DML. No comments. No code fences.
When comparing to TIMESTAMP columns, cast NOW()/CURRENT_TIMESTAMP/CURRENT_DATE to TIMESTAMP.
"""

ANALYSIS_SQL_SYSTEM_PROMPT = """
You are SAGE-SQL for DuckDB.
Return EXACTLY ONE valid SQL SELECT statement for analysis (aggregations/trends).
Prefer GROUP BY (e.g., by month, user_id) over raw row dumps. No comments or code fences.
"""

FEW_SHOTS = """
Q: top 50 orders by sale_price
A: SELECT * FROM orders ORDER BY sale_price DESC LIMIT 50

Q: monthly units by product_id
A: SELECT date_trunc('month', created_at) AS month, product_id, COUNT(*) AS units
   FROM orders
   WHERE created_at IS NOT NULL AND product_id IS NOT NULL
   GROUP BY 1,2
   ORDER BY month DESC
""".strip()

def llm_generate_sql_with_prompt(user_query: str, manifest: str, system_prompt: str) -> str:
    msgs = get_recent_history() + [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{manifest}\n\n{FEW_SHOTS}\n\nQ: {user_query}\nA:"}
    ]
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0)
    return resp.choices[0].message.content.strip()

def llm_repair_sql(bad_sql: str, error_text: str, manifest: str) -> str:
    repair_prompt = f"""
The following SQL failed in DuckDB. Fix it. Keep ONLY one valid SELECT.

Manifest:
{manifest}

Failed SQL:
{bad_sql}

Error:
{error_text}

Return only the corrected SQL:
""".strip()
    msgs = [{"role": "system", "content": SQL_SYSTEM_PROMPT}, {"role": "user", "content": repair_prompt}]
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0)
    return resp.choices[0].message.content.strip()

# ===============================
# Pre-analysis (stats for insights)
# ===============================
def human_td(td: pd.Timedelta | timedelta | float | int) -> str:
    if isinstance(td, (float, int, np.floating, np.integer)):
        total_days = float(td)
        if total_days >= 365: return f"~{total_days/365:.1f} years"
        if total_days >= 30:  return f"~{total_days/30:.1f} months"
        if total_days >= 7:   return f"~{total_days/7:.1f} weeks"
        return f"{total_days:.0f} days"
    if pd.isna(td): return "n/a"
    days = td.days + td.seconds/86400
    return human_td(days)

def pct(part, whole) -> str:
    try:
        if whole == 0 or whole is None: return "0.0%"
        return f"{(part/whole)*100:.1f}%"
    except Exception:
        return "0.0%"

def pre_analyze(df: pd.DataFrame) -> dict:
    out = {"shape": df.shape, "columns": list(df.columns)}
    n = len(df)
    for col in ["created_at","shipped_at","delivered_at","returned_at","status","sale_price"]:
        if col in df.columns:
            na = int(df[col].isna().sum()); out[f"missing_{col}"] = {"missing": na, "pct": pct(na, n)}
    df_local = df.copy()
    if all(c in df_local.columns for c in ["created_at","shipped_at"]):
        df_local["crt_to_ship"] = (df_local["shipped_at"] - df_local["created_at"]).dt.days
    if all(c in df_local.columns for c in ["shipped_at","delivered_at"]):
        df_local["ship_to_deliv"] = (df_local["delivered_at"] - df_local["shipped_at"]).dt.days
    if all(c in df_local.columns for c in ["created_at","delivered_at"]):
        df_local["crt_to_deliv"] = (df_local["delivered_at"] - df_local["created_at"]).dt.days
    for mcol in ["crt_to_ship","ship_to_deliv","crt_to_deliv"]:
        if mcol in df_local.columns:
            s = df_local[mcol].dropna()
            if not s.empty:
                out[mcol] = {"avg": human_td(s.mean()), "p50": human_td(s.quantile(0.50)),
                             "p90": human_td(s.quantile(0.90)), "p99": human_td(s.quantile(0.99)),
                             "max": human_td(s.max())}
    if "created_at" in df.columns:
        df_local["month"] = df_local["created_at"].dt.to_period("M").dt.to_timestamp()
        trend = (df_local.groupby("month", dropna=True)
                 .agg(orders=("order_id","count"), revenue=("sale_price","sum"))
                 .reset_index().sort_values("month"))
        if not trend.empty:
            out["trend"] = trend.tail(12).to_dict(orient="records")
    if "sale_price" in df.columns:
        s = df["sale_price"].dropna()
        if not s.empty:
            out["revenue"] = {"avg": float(s.mean()), "p50": float(s.quantile(0.50)),
                              "p90": float(s.quantile(0.90)), "max": float(s.max())}
    return out

# Prompts
ANALYSIS_ASSISTANT_PROMPT = "You are SAGE-Analysis. Provide concise, factual insights grounded in the provided data."

AUDIT_FORMAT_PROMPT = """
You are SAGE-Audit.
When the user asks for observations, output MUST follow exactly:

Observation 1
Risk
Recommendation

Observation 2
Risk
Recommendation
"""

INFO_SYSTEM_PROMPT = """
You are SAGE-Info, an information expert.
Provide a comprehensive, well-structured, neutral explanation for the user's question.
Include when relevant: definition/overview, key concepts, step-by-step examples, caveats/trade-offs, and practical tips.
Avoid auditspeak. Be thorough but clear.
"""

def build_insight_prompt(user_query: str, df_summary: dict, df_sample: pd.DataFrame) -> str:
    sample_txt = df_sample.to_string(index=False)
    return (f"QUESTION: {user_query}\n\nDATA SUMMARY:\n{df_summary}\n\n"
            f"SAMPLE ROWS (up to 50):\n{trim_text(sample_txt, 2000)}\n\n"
            "Provide a concise narrative of key findings, patterns, anomalies, segments, and trends.")

def build_audit_prompt(user_query: str, df_summary: dict, df_sample: pd.DataFrame) -> str:
    sample_txt = df_sample.to_string(index=False)
    return (f"USER REQUEST: {user_query}\n\nDATASET SUMMARY:\n{df_summary}\n\n"
            f"SAMPLE ROWS (up to 50):\n{trim_text(sample_txt, 2000)}\n\n"
            "Now return output STRICTLY in the Observation/Risk/Recommendation pattern above.")

# ===============================
# Data exec & flows
# ===============================
SQL_START_RE = re.compile(r"\bselect\b", re.I)

def clean_sql_output(text: str) -> str:
    if not text: return ""
    s = (str(text).replace("```sql","").replace("```","")
                   .replace("Here is the SQL:","").replace("Here’s the SQL:","")
                   .replace("Here is the query:","").replace("SQL:","").replace("Query:","")).strip()
    m = SQL_START_RE.search(s);  s = s[m.start():] if m else s
    s = re.sub(r"--.*?$", "", s, flags=re.M); s = re.sub(r"/\*.*?\*/", "", s, flags=re.S).strip()
    parts = [p.strip() for p in s.split(";") if p.strip()]
    first = next((p for p in parts if p.lower().startswith("select")), s)
    return first if first.lower().startswith("select") else ""

def default_orders_sql() -> str:
    return "SELECT * FROM orders ORDER BY created_at DESC NULLS LAST"

def run_sql(sql: str, inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("orders", ords); con.register("inventory", inv); con.register("mg", mg)
    return con.execute(sql).fetchdf()

def build_schema_and_sql(user_query: str, analysis_style: bool):
    inv, ords, mg = load_data()
    manifest = build_schema_manifest(inv, ords, mg)
    sql = (rule_mg_sql(user_query) or rule_inventory_sql(user_query) or
           rule_top_customers_sql(user_query) or rule_orders_by_value_desc(user_query))
    if not sql:
        try:
            sys_prompt = ANALYSIS_SQL_SYSTEM_PROMPT if analysis_style else SQL_SYSTEM_PROMPT
            sql = llm_generate_sql_with_prompt(user_query, manifest, sys_prompt)
        except Exception:
            logging.exception("LLM SQL generation failed; using default query")
            sql = default_orders_sql()
    return inv, ords, mg, manifest, normalize_time_literals(clean_sql_output(sql)) or default_orders_sql()

def run_data_flow(user_query: str, *, analysis_style: bool = False):
    global last_result_df, last_result_sql, last_result_query
    inv, ords, mg, manifest, sql = build_schema_and_sql(user_query, analysis_style)

    ok, missing = validate_sql_columns(sql,
                                       SCHEMA_CONFIG["inventory"]["columns"],
                                       SCHEMA_CONFIG["orders"]["columns"],
                                       SCHEMA_CONFIG["mg"]["columns"])
    if not ok:
        suggestions = suggest_similar_columns(sorted(set(missing)),
            table_cols={"orders": ords.columns.tolist(),"inventory": inv.columns.tolist(),"mg": mg.columns.tolist()},
            tables_to_dfs={"orders": ords, "inventory": inv, "mg": mg},
            synonyms_map=COLUMN_SYNONYMS, max_suggestions=5)
        try:
            repaired = llm_repair_sql(sql, "Missing columns: " + ", ".join(sorted(set(missing))), manifest)
            repaired = normalize_time_literals(clean_sql_output(repaired)) or sql
        except Exception: repaired = sql
        ok2, _ = validate_sql_columns(repaired,
                                      SCHEMA_CONFIG["inventory"]["columns"],
                                      SCHEMA_CONFIG["orders"]["columns"],
                                      SCHEMA_CONFIG["mg"]["columns"])
        if ok2: sql = repaired
        else:
            lines = []
            for miss in sorted(set(missing)):
                lines.append(f"{miss} missing.")
                for item in suggestions.get(miss, []):
                    sample = f" (e.g., {item['sample']})" if item.get("sample") else ""
                    lines.append(f" - {item['col']}{sample}")
            return {"reply": "\n".join(lines) or "Missing columns.", "sql": sql, "preview": None}

    try:
        df = run_sql(sql, inv, ords, mg)
    except Exception as ex:
        try:
            repaired = llm_repair_sql(sql, str(ex), manifest)
            repaired = normalize_time_literals(clean_sql_output(repaired)) or sql
        except Exception: repaired = sql
        try:
            df = run_sql(repaired, inv, ords, mg); sql = repaired
        except Exception as ex2:
            return {"reply": f"Query error: {ex2}", "sql": sql, "preview": None}

    if df.empty:
        return {"reply": "No matching data found for your request.", "sql": sql, "preview": None}

    last_result_df, last_result_sql, last_result_query = df, sql, user_query
    preview_rows = 50 if len(df) > 50 else len(df)
    preview = df.head(preview_rows).to_string(index=False)
    reply_text = f"{preview}\n\n(Showing first {preview_rows} of {len(df)} rows)" if len(df) > 50 else preview
    return {"reply": reply_text, "sql": sql, "preview": preview}

REUSE_PHRASES = ("based on above","based on previous","use previous","from this","based on this","from the last result","above result")

def run_analysis_flow(user_query: str):
    global last_result_df, last_result_sql
    wants_reuse = any(p in (user_query or "").lower() for p in REUSE_PHRASES)
    if wants_reuse and last_result_df is not None:
        df, sql_used = last_result_df.copy(), (last_result_sql or "(previous result)")
    else:
        data_res = run_data_flow(user_query, analysis_style=True)
        if not data_res.get("preview"):
            return {"reply": data_res["reply"], "sql": data_res.get("sql"), "preview": data_res.get("preview")}
        df, sql_used = last_result_df.copy(), last_result_sql
    summary = pre_analyze(df); sample = df.head(50)
    prompt = build_insight_prompt(user_query, summary, sample)
    msgs = get_recent_history() + [{"role":"system","content":ANALYSIS_ASSISTANT_PROMPT},{"role":"user","content":prompt}]
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0)
        analysis = resp.choices[0].message.content.strip()
        return {"reply": analysis, "sql": sql_used, "preview": trim_text(sample.to_string(index=False), 2000)}
    except Exception as ex:
        return {"reply": f"Analysis error: {ex}", "sql": sql_used, "preview": None}

def fetch_online_answer(query: str, max_tokens: int = 900) -> str:
    try:
        msgs = get_recent_history() + [{"role":"system","content":INFO_SYSTEM_PROMPT},{"role":"user","content":query}]
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback: quick Wikipedia
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php",
                             params={"action":"opensearch","search":query,"limit":1,"namespace":0,"format":"json"},
                             timeout=6)
            arr = r.json()
            if len(arr) >= 4 and arr[1]:
                title = arr[1][0]
                r2 = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", timeout=6)
                js = r2.json(); text = (js.get("extract") or "").strip()
                if text: return text[:1400]
        except Exception:
            pass
        return "I couldn’t fetch an online answer right now."

def run_web_flow(user_query: str):
    if "observation" in (user_query or "").lower():
        df = last_result_df if isinstance(last_result_df, pd.DataFrame) and not last_result_df.empty else None
        df_summary = pre_analyze(df) if isinstance(df, pd.DataFrame) else {}
        df_sample = df.head(50) if isinstance(df, pd.DataFrame) else pd.DataFrame()
        prompt = build_audit_prompt(user_query, df_summary, df_sample)
        msgs = get_recent_history() + [{"role":"system","content":AUDIT_FORMAT_PROMPT},{"role":"user","content":prompt}]
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0)
        return {"reply": resp.choices[0].message.content.strip(), "sql": None, "preview": None}
    return {"reply": fetch_online_answer(user_query, max_tokens=900), "sql": None, "preview": None}

# ===============================
# Auto prompts for /run_step
# ===============================
AUTO_PROMPTS_BY_STEP = {
    "rcm": ("Project: {project}. Generate a complete Risk & Control Matrix tailored for this process. "
            "Include: Objective, In-scope subprocesses, Key Risks, Key Controls (preventive/detective), "
            "Control Frequency/Owner/Evidence, and concise Test Steps."),
    "data_request": ("Project: {project}. Prepare a precise Data Request list for the audit. "
                     "Group by source/system/team, list table/file names, fields, filters, window, owners, and format."),
    "findings": ("Project: {project}. Draft 5–8 potential audit findings with Title, Root Cause, Impact, "
                 "Risk Rating, Recommendation, and Owner."),
    "report": ("Project: {project}. Draft an executive summary outline — Background, Scope, Method, "
               "Overall Rating, Key Themes, High/Med findings, Management Responses, Next Steps.")
}

# ===============================
# Routes
# ===============================
@app.get("/")
def index():
    tpl = (BASE_DIR / "templates" / "index.html")
    if tpl.exists(): return render_template("index.html")
    f = BASE_DIR / "index.html"
    return f.read_text(encoding="utf-8") if f.exists() else "<h1>Noon AI</h1>"

@app.get("/healthz")
def healthz():
    try:
        inv, ords, mg = load_data()
        shapes = {
            "inventory": list(inv.shape) if isinstance(inv, pd.DataFrame) else None,
            "orders": list(ords.shape) if isinstance(ords, pd.DataFrame) else None,
            "mg": list(mg.shape) if isinstance(mg, pd.DataFrame) else None
        }
        ok = all(shapes.get(k) for k in ("orders","inventory","mg"))
        return safe_json({"ok": ok, "shapes": shapes}, 200 if ok else 500)
    except Exception as e:
        logging.exception("healthz failed")
        return safe_json({"ok": False, "error": str(e)}, 500)

@app.get("/diagz")
def diagz():
    here = Path(__file__).resolve().parent
    files = sorted([p.name for p in here.glob("*")])
    return safe_json({"cwd": str(here), "files": files})

@app.post("/run_step")
def run_step():
    payload = request.json or {}
    step = (payload.get("step") or "").lower().strip()
    project = (payload.get("project") or "Inventory Management").strip()
    tmpl = AUTO_PROMPTS_BY_STEP.get(step)
    if not tmpl:
        return safe_json({"ok": False, "error": f"Unknown step '{step}'."}, 400)
    query = tmpl.format(project=project)
    res = run_web_flow(query)
    push_history(query, trim_text(res.get("reply", ""), 500))
    return safe_json({"ok": True, "reply": res.get("reply",""), "sql": None, "preview": None, "mode": "web"})

@app.post("/ask")
def ask():
    body = request.get_json(force=True) or {}
    user_query = (body.get("query") or "").strip()
    mode = (body.get("mode") or "data").lower()
    if not user_query:
        return safe_json({"ok": True, "reply": "Please enter a question."})
    if user_query.lower() in {"new chat","/new","reset"}:
        conversation_history.clear()
        global last_result_df, last_result_sql, last_result_query
        last_result_df = last_result_sql = last_result_query = None
        return safe_json({"ok": True, "reply": "Started a new chat."})

    # Specific quick intent: total customer count
    if mode in ("data","analysis") and "total customer" in user_query.lower():
        try:
            inv, ords, mg = load_data()
            count = int(ords["user_id"].dropna().nunique())
            push_history(user_query, f"Total distinct customers: {count}")
            return safe_json({"ok": True, "reply": f"Total distinct customers: {count}", "sql": None})
        except Exception as e:
            logging.exception("fallback total customer count failed")
            return safe_json({"ok": False, "error": f"Failed to compute: {e}"}, 500)

    try:
        if mode == "data":
            res = run_data_flow(user_query, analysis_style=False)
        elif mode == "analysis":
            res = run_analysis_flow(user_query)
        elif mode == "web":
            res = run_web_flow(user_query)
        else:
            res = run_data_flow(user_query, analysis_style=False)
            res["reply"] = f"(demo) Unknown mode '{mode}'. Defaulted to Data.\n\n{res['reply']}"
        push_history(user_query, trim_text(res.get("reply",""), 500))
        return safe_json({"ok": True, "reply": res.get("reply",""), "sql": res.get("sql"), "preview": res.get("preview"), "mode": mode})
    except Exception as e:
        logging.exception("ask failed")
        return safe_json({"ok": False, "error": str(e)}, 500)

# ===============================
# Entrypoint
# ===============================
if __name__ == "__main__":
    # pip install flask pandas numpy duckdb python-dotenv groq requests
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
