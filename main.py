# main.py
from flask import Flask, render_template, request, jsonify
import os, re, difflib, duckdb, requests, math, logging
from dotenv import load_dotenv
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# ============ Optional LLM client (only used if key is set) ============
try:
    from groq import Groq
except Exception:
    Groq = None

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if Groq and key:
        return Groq(api_key=key)
    return None

# ============ Flask setup ============
load_dotenv()
app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
BASE_DIR = Path(__file__).resolve().parent  # /opt/render/project/src on Render

# --------------- JSON helpers ---------------
def _fix_nans(obj):
    if isinstance(obj, float) and math.isnan(obj): return None
    if isinstance(obj, dict):  return {k: _fix_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_fix_nans(v) for v in obj]
    return obj
def safe_json(data, status=200):
    return jsonify(_fix_nans(data)), status

API_PATHS = {"/ask", "/run_step", "/healthz"}

def _wants_json() -> bool:
    try:
        return request.path in API_PATHS or "application/json" in (request.headers.get("Accept") or "")
    except Exception:
        return False

# --------------- Never send HTML to API paths ---------------
@app.errorhandler(Exception)
def _err_any(e):
    # Turn absolutely any error into JSON for API paths
    if _wants_json():
        logging.exception("Unhandled exception")
        code = getattr(e, "code", 500)
        return safe_json({"ok": False, "error": str(e), "code": code}, code)
    # Non-API paths can keep Flask's default HTML page
    raise e

@app.errorhandler(404)
def _err_404(e):
    return safe_json({"ok": False, "error": "Not Found", "code": 404}, 404) if _wants_json() else ("Not Found", 404)

@app.errorhandler(405)
def _err_405(e):
    return safe_json({"ok": False, "error": "Method Not Allowed", "code": 405}, 405) if _wants_json() else ("Method Not Allowed", 405)

# ============ Schema + aliases (same as before) ============
SCHEMA_CONFIG = {
    "orders": {
        "columns": [
            "id","order_id","user_id","product_id","inventory_item_id",
            "status","created_at","shipped_at","delivered_at","returned_at","sale_price"
        ],
        "dtypes": {
            "id":"Int64","order_id":"Int64","user_id":"Int64","product_id":"Int64",
            "inventory_item_id":"Int64","status":"string",
            "created_at":"datetime64[ns]","shipped_at":"datetime64[ns]","delivered_at":"datetime64[ns]","returned_at":"datetime64[ns]",
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

# --------------- Schema helpers ---------------
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

# ============ Data loading (Render-safe) ============
def read_csv_safe(path: Path, name: str):
    """
    Try multiple encodings and parsers so we don't 500 on slightly messy CSVs.
    Returns (df, meta) or raises RuntimeError with a concise trail.
    """
    attempts = [
        dict(encoding="utf-8-sig", engine="c", low_memory=False),
        dict(encoding="utf-8",     engine="c", low_memory=False),
        dict(encoding="latin-1",   engine="c", low_memory=False),
        # very forgiving fallback (slower): skips bad rows rather than dying
        dict(encoding="utf-8",     engine="python", on_bad_lines="skip", low_memory=False),
        dict(encoding="latin-1",   engine="python", on_bad_lines="skip", low_memory=False),
    ]
    errors = []
    for kw in attempts:
        try:
            df = pd.read_csv(path, **kw)
            meta = {
                "name": name,
                "path": str(path),
                "encoding": kw.get("encoding"),
                "engine": kw.get("engine"),
                "on_bad_lines": kw.get("on_bad_lines", "error"),
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            }
            return df, meta
        except Exception as e:
            errors.append(f"{kw.get('encoding')}/{kw.get('engine')}/{kw.get('on_bad_lines','error')}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Failed to read {name} at {path}. Tried -> " + " | ".join(errors))


def read_first_safe(candidates, name: str):
    """
    First existing path from candidates, read it via read_csv_safe.
    Returns (df, meta) or (None, {'name':..., 'missing': True, 'tried': [...]})
    """
    tried = []
    for fname in candidates:
        p = (BASE_DIR / fname)
        if p.exists():
            df, meta = read_csv_safe(p, name)
            meta["resolved"] = fname
            return df, meta
        tried.append(str(p))
    return None, {"name": name, "missing": True, "tried": tried}

def load_data():
    # inventory.csv
    inv_df, inv_meta = read_csv_safe(BASE_DIR / "inventory.csv", "inventory.csv")
    # orders.csv
    ord_df, ord_meta = read_csv_safe(BASE_DIR / "orders.csv", "orders.csv")
    # minimum_guarantee.csv (or tolerated alternatives)
    mg_df, mg_meta = read_first_safe(
        ["minimum_guarantee.csv", "Minimum_Guarantee.csv", "mg.csv"],
        "minimum_guarantee.csv"
    )

    # Normalize/alias
    inv_df.columns = [c.strip().lower() for c in inv_df.columns]
    ord_df.columns = [c.strip().lower() for c in ord_df.columns]
    if mg_df is not None:
        mg_df.columns = [c.strip().lower().replace(" ", "_") for c in mg_df.columns]

    inv_df  = _apply_aliases(inv_df,  SCHEMA_CONFIG["inventory"]["aliases"])
    ord_df  = _apply_aliases(ord_df,  SCHEMA_CONFIG["orders"]["aliases"])
    if mg_df is not None:
        mg_df = _apply_aliases(mg_df, SCHEMA_CONFIG["mg"]["aliases"])

    # Enforce schemas (never crash)
    inv = _enforce_schema(inv_df, "inventory")
    ords = _enforce_schema(ord_df, "orders")
    mg  = _enforce_schema(mg_df, "mg") if mg_df is not None else pd.DataFrame(columns=SCHEMA_CONFIG["mg"]["columns"])

    # Helpful logs once at startup/use
    logging.info("CSV OK • inventory=%s rows • orders=%s rows • mg=%s rows",
                 inv.shape[0], ords.shape[0], (mg.shape[0] if mg is not None else 0))

    # Return dataframes + lightweight meta so /healthz can show what happened
    return inv, ords, mg, {"inventory": inv_meta, "orders": ord_meta, "mg": mg_meta}

def load_data_for_routes():
    inv, ords, mg, _ = load_data()
    return inv, ords, mg
    
# ============ SQL helpers & rules ============
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

# ============ LLM SQL (optional) ============
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
    client = get_groq_client()
    if not client:
        raise RuntimeError("Groq client not configured")
    msgs = [{"role":"system","content":system_prompt},
            {"role":"user","content":f"{manifest}\n\n{FEW_SHOTS}\n\nQ: {user_query}\nA:"}]
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0)
    return resp.choices[0].message.content.strip()

def llm_repair_sql(bad_sql: str, error_text: str, manifest: str) -> str:
    client = get_groq_client()
    if not client:
        raise RuntimeError("Groq client not configured")
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
    msgs = [{"role":"system","content":SQL_SYSTEM_PROMPT},{"role":"user","content":repair_prompt}]
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0)
    return resp.choices[0].message.content.strip()

# ============ Data exec & flows ============
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
    inv, ords, mg = load_data_for_routes()
    manifest = build_schema_manifest(inv, ords, mg)
    sql = (rule_mg_sql(user_query) or rule_inventory_sql(user_query) or rule_orders_by_value_desc(user_query))
    if not sql:
        try:
            sys_prompt = ANALYSIS_SQL_SYSTEM_PROMPT if analysis_style else SQL_SYSTEM_PROMPT
            sql = llm_generate_sql_with_prompt(user_query, manifest, sys_prompt)
        except Exception:
            logging.exception("LLM SQL generation unavailable; using default query")
            sql = default_orders_sql()
    return inv, ords, mg, manifest, normalize_time_literals(clean_sql_output(sql)) or default_orders_sql()

def run_data_flow(user_query: str, *, analysis_style: bool = False):
    inv, ords, mg, manifest, sql = build_schema_and_sql(user_query, analysis_style)

    ok, missing = validate_sql_columns(sql,
                                       SCHEMA_CONFIG["inventory"]["columns"],
                                       SCHEMA_CONFIG["orders"]["columns"],
                                       SCHEMA_CONFIG["mg"]["columns"])
    if not ok:
        try:
            repaired = llm_repair_sql(sql, "Missing columns: " + ", ".join(sorted(set(missing))), manifest)
            repaired = normalize_time_literals(clean_sql_output(repaired)) or sql
            ok2, _ = validate_sql_columns(repaired,
                                          SCHEMA_CONFIG["inventory"]["columns"],
                                          SCHEMA_CONFIG["orders"]["columns"],
                                          SCHEMA_CONFIG["mg"]["columns"])
            if ok2:
                sql = repaired
        except Exception:
            pass  # keep original sql or fall through

    try:
        df = run_sql(sql, inv, ords, mg)
    except Exception as ex:
        # final fallback: simple safe default
        logging.exception("SQL execution failed; falling back to default")
        df = run_sql(default_orders_sql(), inv, ords, mg)
        sql = default_orders_sql()

    if df.empty:
        return {"reply": "No matching data found for your request.", "sql": sql, "preview": None}

    preview_rows = 50 if len(df) > 50 else len(df)
    preview = df.head(preview_rows).to_string(index=False)
    reply_text = f"{preview}\n\n(Showing first {preview_rows} of {len(df)} rows)" if len(df) > 50 else preview
    return {"reply": reply_text, "sql": sql, "preview": preview}

# ============ Info (web) ============
INFO_SYSTEM_PROMPT = """
You are SAGE-Info, an information expert.
Provide a comprehensive, well-structured, neutral explanation for the user's question.
Include when relevant: definition/overview, key concepts, step-by-step examples, caveats/trade-offs, and practical tips.
Avoid auditspeak. Be thorough but clear.
"""

def fetch_online_answer(query: str, max_tokens: int = 900) -> str:
    client = get_groq_client()
    if client:
        try:
            msgs = [{"role":"system","content":INFO_SYSTEM_PROMPT},{"role":"user","content":query}]
            resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except Exception:
            logging.exception("LLM web answer failed; falling back to Wikipedia")
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
        logging.exception("Wikipedia fallback failed")
    return "I couldn’t fetch an online answer right now."

# ============ Routes ============
@app.get("/")
def index():
    tpl = (BASE_DIR / "templates" / "index.html")
    if tpl.exists(): return render_template("index.html")
    f = BASE_DIR / "index.html"
    return f.read_text(encoding="utf-8") if f.exists() else "<h1>Noon AI</h1>"

@app.get("/diagz")
def diagz():
    here = Path(__file__).resolve().parent
    files = sorted([p.name for p in here.glob("*")])
    return safe_json({"cwd": str(here), "files": files})

@app.get("/healthz")
def healthz():
    try:
        inv, ords, mg, meta = load_data()  # uses the robust reader above
        shapes = {
            "inventory": list(inv.shape) if isinstance(inv, pd.DataFrame) else None,
            "orders":    list(ords.shape) if isinstance(ords, pd.DataFrame) else None,
            "mg":        list(mg.shape)  if isinstance(mg,  pd.DataFrame) else None,
        }
        ok = all(shapes.get(k) for k in ("orders", "inventory"))  # mg can be legitimately missing
        return safe_json({
            "ok": ok,
            "shapes": shapes,
            "meta": meta
        }, 200 if ok else 500)
    except Exception as e:
        logging.exception("/healthz failed")
        # Return a terse but informative JSON error (no HTML)
        return safe_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

@app.post("/run_step")
def run_step():
    body = request.get_json(silent=True) or {}
    step = (body.get("step") or "").lower().strip()
    project = (body.get("project") or "Inventory Management").strip()
    if not step:
        return safe_json({"ok": False, "error": "Missing 'step'."}, 400)
    # Keep your current audit-generation flow short & simple for now
    reply = f"Ran step '{step}' for project '{project}'."
    return safe_json({"ok": True, "reply": reply, "sql": None, "preview": None, "mode": "web"})

@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    user_query = (body.get("query") or "").strip()
    mode = (body.get("mode") or "data").lower()
    if not user_query:
        return safe_json({"ok": True, "reply": "Please enter a question."})
    if user_query.lower() in {"new chat","/new","reset"}:
        return safe_json({"ok": True, "reply": "Started a new chat."})

    # Quick intent that must always work
    if mode in ("data","analysis") and "total customer" in user_query.lower():
        inv, ords, mg = load_data_for_routes()
        count = int(ords["user_id"].dropna().nunique())
        return safe_json({"ok": True, "reply": f"Total distinct customers: {count}", "sql": None})

    try:
        if mode == "data":
            res = run_data_flow(user_query, analysis_style=False)
        elif mode == "analysis":
            # reuse same dataset generation but you can add narrative later
            res = run_data_flow(user_query, analysis_style=True)
        elif mode == "web":
            return safe_json({"ok": True, "reply": fetch_online_answer(user_query), "sql": None, "preview": None, "mode": "web"})
        else:
            res = run_data_flow(user_query, analysis_style=False)
            res["reply"] = f"(demo) Unknown mode '{mode}'. Defaulted to Data.\n\n{res['reply']}"
        return safe_json({"ok": True, "reply": res.get("reply",""), "sql": res.get("sql"), "preview": res.get("preview"), "mode": mode})
    except Exception as e:
        logging.exception("ask failed")
        return safe_json({"ok": False, "error": str(e)}, 500)

# ============ Entrypoint ============
if __name__ == "__main__":
    # pip install flask pandas numpy duckdb python-dotenv groq requests
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
