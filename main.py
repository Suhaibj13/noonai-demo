from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os, re, difflib, duckdb, requests
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from datetime import timedelta

# ===============================
# Load env & init client
# ===============================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)

# ===============================
# Conversation & cache (demo)
# ===============================
conversation_history = []           # [{"role": "user"|"assistant", "content": "..."}]
MAX_TURNS = 10

last_result_df: pd.DataFrame | None = None
last_result_sql: str | None = None
last_result_query: str | None = None

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
# Fixed Schema Config
# ===============================
SCHEMA_CONFIG = {
    "orders": {
        "columns": [
            "id", "order_id", "user_id", "product_id", "inventory_item_id",
            "status", "created_at", "shipped_at", "delivered_at", "returned_at",
            "sale_price"
        ],
        "dtypes": {
            "id": "Int64", "order_id": "Int64", "user_id": "Int64",
            "product_id": "Int64", "inventory_item_id": "Int64",
            "status": "string",
            "created_at": "datetime64[ns]", "shipped_at": "datetime64[ns]",
            "delivered_at": "datetime64[ns]", "returned_at": "datetime64[ns]",
            "sale_price": "Float64"
        },
        "aliases": {
            "price": "sale_price",
            "customer_id": "user_id", "buyer_id": "user_id", "client_id": "user_id"
        }
    },
    "inventory": {
        "columns": [
            "id", "product_id", "created_at", "sold_at", "cost",
            "product_category", "product_name", "product_brand",
            "product_retail_price", "product_department",
            "product_sku", "product_distribution_center_id"
        ],
        "dtypes": {
            "id": "Int64", "product_id": "Int64",
            "created_at": "datetime64[ns]", "sold_at": "datetime64[ns]",
            "cost": "Float64",
            "product_category": "string", "product_name": "string",
            "product_brand": "string", "product_retail_price": "Float64",
            "product_department": "string", "product_sku": "string",
            "product_distribution_center_id": "Int64"
        },
        "aliases": {"unit_cost": "cost", "buy_price": "cost"}
    }
}

SCHEMA_CONFIG["mg"] = {
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
    "aliases": {
        "vendor":"vendor_name","final_amount":"da_level_final_amount","penalty":"fnd_ndr_penalty_amount"
    }
}

INVENTORY_PATH = "inventory.csv"
ORDERS_PATH    = "orders.csv"
MG_PATHS = ["minimum_guarantee.csv", "minimum_guarantee.cv", "mg.csv", "mg.cv"]

COLUMN_SYNONYMS = {
    "orders": {
        "user_id": ["customer", "customer_id", "buyer_id", "client_id"],
        "sale_price": ["price", "unit_price", "amount"],
        "order_id": ["oid", "sales_order_id"]
    },
    "inventory": {
        "product_id": ["pid", "item_id"],
        "cost": ["unit_cost", "buy_price"]
    }
}

COLUMN_SYNONYMS["mg"] = {
    "vendor_name": ["vendor"],
    "fnd_ndr_penalty_amount": ["penalty", "fnd_ndr_penalty"],
    "da_level_final_amount": ["final_amount", "da_final_amount"]
}

# ===============================
# Load & enforce schema
# ===============================
def _apply_aliases(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    to_rename = {}
    lower_cols = {c.lower(): c for c in df.columns}
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
        if col not in df.columns:
            continue
        try:
            if dt == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
            elif dt in ("Int64", "Float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dt)
            elif dt == "string":
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype(dt)
        except Exception:
            pass
    df = df[spec["columns"]]
    return df

from pathlib import Path

def _read_first(path_list):
    for p in path_list:
        if Path(p).exists():
            return pd.read_csv(p)
    return None

def load_data():
    inv = pd.read_csv(INVENTORY_PATH)
    ords = pd.read_csv(ORDERS_PATH)
    mg_raw = _read_first(MG_PATHS)

    inv.columns = [c.strip().lower() for c in inv.columns]
    ords.columns = [c.strip().lower() for c in ords.columns]
    if mg_raw is not None:
        mg_raw.columns = [c.strip().lower().replace(" ", "_") for c in mg_raw.columns]

    inv = _apply_aliases(inv, SCHEMA_CONFIG["inventory"]["aliases"])
    ords = _apply_aliases(ords, SCHEMA_CONFIG["orders"]["aliases"])
    if mg_raw is not None:
        mg_raw = _apply_aliases(mg_raw, SCHEMA_CONFIG["mg"]["aliases"])

    inv = _enforce_schema(inv, "inventory")
    ords = _enforce_schema(ords, "orders")
    mg  = _enforce_schema(mg_raw, "mg") if mg_raw is not None else pd.DataFrame(columns=SCHEMA_CONFIG["mg"]["columns"])
    return inv, ords, mg


# ===============================
# SQL helpers
# ===============================
def normalize_time_literals(sql: str) -> str:
    sql = re.sub(r"\bCURRENT_TIMESTAMP\b", "CAST(CURRENT_TIMESTAMP AS TIMESTAMP)", sql, flags=re.I)
    sql = re.sub(r"\bNOW\(\)", "CAST(NOW() AS TIMESTAMP)", sql, flags=re.I)
    sql = re.sub(r"\bCURRENT_DATE\b", "CAST(CURRENT_DATE AS TIMESTAMP)", sql, flags=re.I)
    return sql

def validate_sql_columns(sql: str, inv_cols: list, ord_cols: list,mg_cols: list):
    missing = []
    inv_lower = [c.lower() for c in inv_cols]
    ord_lower = [c.lower() for c in ord_cols]
    mg_lower  = [c.lower() for c in mg_cols]
    for alias, cols in (("i", inv_lower), ("inventory", inv_lower),
                        ("o", ord_lower), ("orders", ord_lower),
                        ("m", mg_lower), ("mg", mg_lower)):
        for m in re.findall(rf"\b{alias}\.(\w+)\b", sql, flags=re.I):
            if m.lower() not in cols:
                t = "inventory" if alias in ("i", "inventory") else "orders" if alias in ("o","orders") else "mg"
                missing.append(f"{t}.{m}")
    return (len(missing) == 0, missing)

def suggest_similar_columns(missing_cols: list, table_cols: dict,
                            tables_to_dfs: dict, synonyms_map: dict, max_suggestions=5):
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

# ===============================
# Rule-based SQL (few helpers)
# ===============================
def rule_orders_by_value_desc(user_query: str) -> str | None:
    q = (user_query or "").lower()
    if "orders" not in q:
        return None
    if "value" in q and (("decreas" in q) or ("desc" in q) or ("descending" in q) or ("highest" in q)):
        m = re.search(r"\b(top|first|limit|give me|show me)\s*(\d+)", q)
        n = int(m.group(2)) if m else 500
        return f"SELECT * FROM orders ORDER BY sale_price DESC LIMIT {n}"
    return None

def rule_top_customers_sql(user_query: str) -> str | None:
    q = (user_query or "").lower()
    if not (("month" in q or "monthly" in q) and ("customer" in q or "customers" in q or "user_id" in q)):
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
    inventory_like = any(k in q for k in [
        "inventory", "stockout", "stock out", "overstock", "on hand", "reorder",
        "slow-moving", "slow moving"
    ])
    if not inventory_like:
        return None
    want_monthly = ("month" in q or "monthly" in q)
    want_slow = any(k in q for k in ["slow", "overstock"])
    want_reorder = any(k in q for k in ["reorder", "low stock", "low inventory", "stockout", "stock out"])
    want_top_sold = any(k in q for k in ["most sold", "top sold", "best seller", "bestseller",
                                         "most selling", "top sales", "top sold items"])
    top_n = _parse_top_n(q, 5)
    th = _parse_threshold(q, 10)

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

# ===============================
# LLM prompts (SQL + Analysis + Audit/Web)
# ===============================
def build_schema_manifest(inv: pd.DataFrame, ords: pd.DataFrame,mg: pd.DataFrame) -> str:
    def col_line(df, col):
        s = df[col].dropna()
        sample = "" if s.empty else f" sample={repr(str(s.iloc[0])[:40])}"
        return f"{col} {str(df[col].dtype)} nullable={df[col].isna().any()}{sample}"
    lines = ["Tables and columns (DuckDB dtypes):"]
    lines.append("[inventory]")
    for c in SCHEMA_CONFIG["inventory"]["columns"]:
        if c in inv.columns: lines.append("  - " + col_line(inv, c))
    lines.append("[orders]")
    for c in SCHEMA_CONFIG["orders"]["columns"]:
        if c in ords.columns: lines.append("  - " + col_line(ords, c))
    lines.append("[mg]")
    for c in SCHEMA_CONFIG["mg"]["columns"]:
        if c in mg.columns: lines.append("  - " + col_line(mg, c))
    return "\n".join(lines)

SQL_SYSTEM_PROMPT = """
You are SAGE-SQL for DuckDB.
Return EXACTLY ONE valid SQL SELECT statement and nothing else.
Ignore phrasing like "give me" or "analyze": always produce the best SELECT over the provided schema.
Use ONLY the provided tables and columns. No DDL/DML. No comments. No code fences.
If no LIMIT is present, do not add one.
When comparing to TIMESTAMP columns, cast NOW()/CURRENT_TIMESTAMP/CURRENT_DATE to TIMESTAMP.
"""

ANALYSIS_SQL_SYSTEM_PROMPT = """
You are SAGE-SQL for DuckDB.
Return EXACTLY ONE valid SQL SELECT statement and nothing else.
Goal: produce a compact, analysis-ready dataset (aggregations/trends) for the question.
Prefer GROUP BY (e.g., by month and/or user_id) rather than raw row dumps.
Use ONLY the provided tables and columns. No DDL/DML. No comments. No code fences.
If no LIMIT is present, do not add one.
When comparing to TIMESTAMP columns, cast NOW()/CURRENT_TIMESTAMP/CURRENT_DATE to TIMESTAMP.
"""

FEW_SHOTS = """
Examples (DuckDB):
Q: top 50 orders by sale_price
A: SELECT * FROM orders ORDER BY sale_price DESC LIMIT 50

Q: monthly units by product_id
A: SELECT date_trunc('month', created_at) AS month, product_id, COUNT(*) AS units
   FROM orders
   WHERE created_at IS NOT NULL AND product_id IS NOT NULL
   GROUP BY 1,2
   ORDER BY month DESC
""".strip()

# ---- Analysis style (no O/R/R) ----
ANALYSIS_ASSISTANT_PROMPT = """
You are SAGE-Analysis, a data insights assistant.
Write concise, factual insights grounded in the provided summary and sample rows.
Highlight trends, seasonality, top/bottom drivers, notable outliers, and data-quality gaps.
Avoid "Observation/Risk/Recommendation" formatting. No bullet labels. Be crisp.
"""

# ---- Web mode audit formatter (only when 'observation(s)' requested) ----
AUDIT_FORMAT_PROMPT = """
You are SAGE-Audit.
When the user asks for observations, output MUST follow exactly:

Observation 1
Risk
Recommendation

Observation 2
Risk
Recommendation

Make each observation short and factual; risks should describe real impact; recommendations must be actionable.
No extra headings or prose outside that structure.
"""

ONLINE_SYSTEM_PROMPT = """
You are SAGE, a concise assistant. Provide a brief, accurate answer using general knowledge or reputable public sources.
Be direct and avoid filler.
"""

# ===============================
# LLM wrappers
# ===============================
def llm_generate_sql_with_prompt(user_query: str, manifest: str, system_prompt: str) -> str:
    msgs = get_recent_history() + [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{manifest}\n\n{FEW_SHOTS}\n\nQ: {user_query}\nA:"}
    ]
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=msgs, temperature=0
    )
    return resp.choices[0].message.content.strip()

def llm_generate_sql(user_query: str, manifest: str) -> str:
    return llm_generate_sql_with_prompt(user_query, manifest, SQL_SYSTEM_PROMPT)

def llm_repair_sql(bad_sql: str, error_text: str, manifest: str) -> str:
    repair_prompt = f"""
The following SQL failed in DuckDB. Fix it. Keep ONLY one valid SELECT statement, no comments.

Manifest:
{manifest}

Failed SQL:
{bad_sql}

Error:
{error_text}

Return only the corrected SQL:
""".strip()
    msgs = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": repair_prompt}
    ]
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=msgs, temperature=0
    )
    return resp.choices[0].message.content.strip()

# ===============================
# Analysis Pre-processor
# ===============================
def human_td(td: pd.Timedelta | timedelta | float | int) -> str:
    """Human-friendly timedelta or number of days."""
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

def safe_q(df: pd.DataFrame, col: str, q: float):
    try:
        return df[col].quantile(q)
    except Exception:
        return np.nan

def pre_analyze(df: pd.DataFrame) -> dict:
    """Compute summary stats, outliers and trend snippets for the LLM."""
    out = {"shape": df.shape, "columns": list(df.columns)}
    n = len(df)

    # 1) Data quality (selected columns if present)
    for col in ["created_at", "shipped_at", "delivered_at", "returned_at", "status", "sale_price"]:
        if col in df.columns:
            na = int(df[col].isna().sum())
            out[f"missing_{col}"] = {"missing": na, "pct": pct(na, n)}

    # 2) Cycle times if dates exist
    df_local = df.copy()
    if all(c in df_local.columns for c in ["created_at", "shipped_at"]):
        df_local["crt_to_ship"] = (df_local["shipped_at"] - df_local["created_at"]).dt.days
    if all(c in df_local.columns for c in ["shipped_at", "delivered_at"]):
        df_local["ship_to_deliv"] = (df_local["delivered_at"] - df_local["shipped_at"]).dt.days
    if all(c in df_local.columns for c in ["created_at", "delivered_at"]):
        df_local["crt_to_deliv"] = (df_local["delivered_at"] - df_local["created_at"]).dt.days

    for mcol in ["crt_to_ship", "ship_to_deliv", "crt_to_deliv"]:
        if mcol in df_local.columns:
            s = df_local[mcol].dropna()
            if not s.empty:
                out[mcol] = {
                    "avg": human_td(s.mean()),
                    "p50": human_td(s.quantile(0.50)),
                    "p90": human_td(s.quantile(0.90)),
                    "p99": human_td(s.quantile(0.99)),
                    "max": human_td(s.max())
                }

    # 3) Outliers (top 5 slowest deliveries)
    if "crt_to_deliv" in df_local.columns:
        slow = df_local[["user_id","order_id","crt_to_deliv"]].dropna().sort_values("crt_to_deliv", ascending=False).head(5)
        out["top_slowest"] = slow.to_dict(orient="records")

    # 4) Trend by month if created_at present
    if "created_at" in df.columns:
        df_local["month"] = df_local["created_at"].dt.to_period("M").dt.to_timestamp()
        trend = (df_local
                 .groupby("month", dropna=True)
                 .agg(orders=("order_id","count"),
                      revenue=("sale_price","sum"))
                 .reset_index()
                 .sort_values("month"))
        if not trend.empty:
            out["trend"] = trend.tail(12).to_dict(orient="records")
            out["trend_growth_last3"] = (
                trend["orders"].tail(3).pct_change().dropna().mean()
                if trend["orders"].tail(3).shape[0] >= 2 else np.nan
            )

    # 5) Revenue distribution
    if "sale_price" in df.columns:
        s = df["sale_price"].dropna()
        if not s.empty:
            out["revenue"] = {
                "avg": float(s.mean()),
                "p50": float(s.quantile(0.50)),
                "p90": float(s.quantile(0.90)),
                "max": float(s.max())
            }

    return out

def build_audit_prompt(user_query: str, df_summary: dict, df_sample: pd.DataFrame) -> str:
    """For Web mode 'observations' requests — strict O/R/R format."""
    sample_txt = df_sample.to_string(index=False)
    return (
        f"USER REQUEST: {user_query}\n\n"
        f"DATASET SUMMARY (pre-analysis):\n{df_summary}\n\n"
        f"SAMPLE ROWS (up to 50):\n{trim_text(sample_txt, 2000)}\n\n"
        "Now return output STRICTLY in this format:\n\n"
        "Observation 1\nRisk\nRecommendation\n\n"
        "Observation 2\nRisk\nRecommendation\n"
    )

def build_insight_prompt(user_query: str, df_summary: dict, df_sample: pd.DataFrame) -> str:
    """For Analysis mode — pure insights (no O/R/R)."""
    sample_txt = df_sample.to_string(index=False)
    return (
        f"QUESTION: {user_query}\n\n"
        f"DATA SUMMARY:\n{df_summary}\n\n"
        f"SAMPLE ROWS (up to 50):\n{trim_text(sample_txt, 2000)}\n\n"
        "Provide a concise narrative of key findings, patterns, anomalies, segments, and trends. "
        "Avoid 'Observation/Risk/Recommendation' formatting."
    )

# ===============================
# Data executors
# ===============================
SQL_START_RE = re.compile(r"\bselect\b", re.I)

def clean_sql_output(text: str) -> str:
    if not text: return ""
    s = str(text)
    s = s.replace("```sql", "").replace("```", "")
    s = s.replace("Here is the SQL:", "").replace("Here’s the SQL:", "")
    s = s.replace("Here is the query:", "").replace("SQL:", "").replace("Query:", "")
    m = SQL_START_RE.search(s)
    if m: s = s[m.start():]
    s = re.sub(r"--.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S).strip()
    parts = [p.strip() for p in s.split(";") if p.strip()]
    first = next((p for p in parts if p.lower().startswith("select")), s)
    return first if first.lower().startswith("select") else ""

def default_orders_sql() -> str:
    return "SELECT * FROM orders ORDER BY created_at DESC NULLS LAST"

def run_sql(sql: str, inv: pd.DataFrame, ords: pd.DataFrame, mg: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("orders", ords)
    con.register("inventory", inv)
    con.register("mg", mg)  # now valid because mg is a parameter
    return con.execute(sql).fetchdf()

def run_data_flow(user_query: str, *, analysis_style: bool = False):
    sql = (rule_mg_sql(user_query)
       or rule_inventory_sql(user_query)
       or rule_top_customers_sql(user_query)
       or rule_orders_by_value_desc(user_query))
    """
    Returns { reply, sql, preview }.
    analysis_style=True → ask LLM for grouped/trend-friendly SQL (no LIMITs injected).
    """
    global last_result_df, last_result_sql, last_result_query

    inv, ords, mg = load_data()
    manifest = build_schema_manifest(inv, ords,mg)

    # Rules → else LLM
    if not sql:
        sys_prompt = ANALYSIS_SQL_SYSTEM_PROMPT if analysis_style else SQL_SYSTEM_PROMPT
        sql = llm_generate_sql_with_prompt(user_query, manifest, sys_prompt)

    sql = normalize_time_literals(clean_sql_output(sql)) or default_orders_sql()

    # Schema validation
    is_valid, missing_cols = validate_sql_columns(sql,
        SCHEMA_CONFIG["inventory"]["columns"], SCHEMA_CONFIG["orders"]["columns"],SCHEMA_CONFIG["mg"]["columns"])
    if not is_valid:
        suggestions = suggest_similar_columns(
            missing_cols=sorted(set(missing_cols)),
            table_cols={"orders": ords.columns.tolist(), "inventory": inv.columns.tolist(),"mg": mg.columns.tolist()},
            tables_to_dfs={"orders": ords, "inventory": inv, "mg": mg},
            synonyms_map=COLUMN_SYNONYMS, max_suggestions=5
        )
        repaired = llm_repair_sql(sql, "Missing columns: " + ", ".join(sorted(set(missing_cols))), manifest)
        repaired = normalize_time_literals(clean_sql_output(repaired)) or sql
        ok2, _ = validate_sql_columns(repaired,
            SCHEMA_CONFIG["inventory"]["columns"], SCHEMA_CONFIG["orders"]["columns"],SCHEMA_CONFIG["mg"]["columns"])
        if ok2:
            sql = repaired
        else:
            lines = []
            for miss in sorted(set(missing_cols)):
                lines.append(f"{miss} missing.")
                for item in suggestions.get(miss, []):
                    sample = f" (e.g., {item['sample']})" if item.get("sample") else ""
                    lines.append(f" - {item['col']}{sample}")
            return {"reply": "\n".join(lines) or "Missing columns.", "sql": sql, "preview": None}

    # Execute with one-shot repair
    try:
        df = run_sql(sql, inv, ords, mg)
    except Exception as ex:
        repaired = llm_repair_sql(sql, str(ex), manifest)
        repaired = normalize_time_literals(clean_sql_output(repaired)) or sql
        try:
            df = run_sql(repaired, inv, ords,mg)
            sql = repaired
        except Exception as ex2:
            return {"reply": f"Query error: {ex2}", "sql": sql, "preview": None}

    if df.empty:
        return {"reply": "No matching data found for your request.", "sql": sql, "preview": None}

    # Cache full df (no artificial LIMIT here)
    last_result_df = df
    last_result_sql = sql
    last_result_query = user_query

    # For display: show up to 50 rows if very large
    preview_rows = 50 if len(df) > 50 else len(df)
    preview = df.head(preview_rows).to_string(index=False)
    if len(df) > 50:
        reply_text = f"{preview}\n\n(Showing first {preview_rows} of {len(df)} rows)"
    else:
        reply_text = preview

    return {"reply": reply_text, "sql": sql, "preview": preview}

# Phrases that mean: reuse previous result
REUSE_PHRASES = ("based on above", "based on previous", "use previous", "from this",
                 "based on this", "from the last result", "above result")

def run_analysis_flow(user_query: str):
    """
    If user explicitly says to reuse → analyze cached df.
    Otherwise, produce a new analysis-style dataset first (no forced LIMIT),
    then compute pre-analysis & ask LLM for concise insights (no O/R/R).
    """
    global last_result_df, last_result_sql, last_result_query

    wants_reuse = any(p in (user_query or "").lower() for p in REUSE_PHRASES)

    if wants_reuse and last_result_df is not None:
        df = last_result_df.copy()
        sql_used = last_result_sql or "(previous result)"
    else:
        data_res = run_data_flow(user_query, analysis_style=True)
        if not data_res.get("preview"):
            return {"reply": data_res["reply"], "sql": data_res.get("sql"), "preview": data_res.get("preview")}
        df = last_result_df.copy()
        sql_used = last_result_sql

    # Pre-analysis + analysis prompt (no O/R/R)
    summary = pre_analyze(df)
    sample = df.head(50)
    prompt = build_insight_prompt(user_query, summary, sample)
    msgs = get_recent_history() + [
        {"role": "system", "content": ANALYSIS_ASSISTANT_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=msgs, temperature=0
        )
        analysis = resp.choices[0].message.content.strip()
        return {
            "reply": analysis,
            "sql": sql_used,
            "preview": trim_text(sample.to_string(index=False), 2000)
        }
    except Exception as ex:
        return {"reply": f"Analysis error: {ex}", "sql": sql_used, "preview": None}

# ===============================
# Web (Wikipedia + audit formatter fallback)
# ===============================
def fetch_online_answer(query: str, max_chars=700) -> str:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"opensearch","search":query,"limit":1,"namespace":0,"format":"json"},
            timeout=6,
        )
        arr = r.json()
        if len(arr) >= 4 and arr[1]:
            title = arr[1][0]
            r2 = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", timeout=6
            )
            js = r2.json()
            text = (js.get("extract") or "").strip()
            if text:
                return text[:max_chars]
    except Exception:
        pass
    msgs = get_recent_history() + [
        {"role": "system", "content": ONLINE_SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=msgs, temperature=0
    )
    return resp.choices[0].message.content.strip()

def run_web_flow(user_query: str):
    # If the user explicitly asks for "observations", produce O/R/R using recent data if available
    if "observation" in (user_query or "").lower():
        df = last_result_df if last_result_df is not None and not last_result_df.empty else None
        df_summary = pre_analyze(df) if isinstance(df, pd.DataFrame) else {}
        df_sample = df.head(50) if isinstance(df, pd.DataFrame) else pd.DataFrame()
        prompt = build_audit_prompt(user_query, df_summary, df_sample)
        msgs = get_recent_history() + [
            {"role": "system", "content": AUDIT_FORMAT_PROMPT},
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(  # <-- fixed below to .chat.completions
            model="llama-3.3-70b-versatile", messages=msgs, temperature=0
        )
        return {"reply": resp.choices[0].message.content.strip(), "sql": None, "preview": None}
    # Otherwise, normal concise web answer
    return {"reply": fetch_online_answer(user_query), "sql": None, "preview": None}

# ===============================
# Auto-prompts for Audit Steps
# ===============================
AUTO_PROMPTS_BY_STEP = {
    "rcm": (
        "Project: {project}. Generate a complete Risk & Control Matrix tailored for this process. "
        "Include: Objective, In-scope subprocesses, Key Risks, Key Controls (preventive/detective), "
        "Control Frequency/Owner/Evidence, and concise Test Steps. Keep it table-friendly and pragmatic."
    ),
    "data_request": (
        "Project: {project}. Prepare a precise Data Request list for the audit. "
        "Group by source/system/team, list table/file names, mandatory fields, filters, time windows, "
        "owners, and delivery format. Add data quality checks we will perform on receipt."
    ),
    "findings": (
        "Project: {project}. Draft potential audit findings (5–8) we commonly see in this area. "
        "For each: Title, Root Cause, Impact, Risk Rating, Recommendation, and Owner/Action Party."
    ),
    "report": (
        "Project: {project}. Draft an executive summary outline for the audit report. "
        "Include: Background, Scope, Method, Overall Rating, Key Themes, High/Med findings, "
        "Management Responses (placeholders), and Next Steps."
    ),
}

# ===============================
# Routes
# ===============================
@app.route("/")
def root():
    tpl = Path("templates/index.html")
    if tpl.exists():
        return render_template("index.html")
    f = Path("index.html")
    return f.read_text(encoding="utf-8") if f.exists() else "<h1>SAGE</h1>"

@app.post("/run_step")
def run_step():
    payload = request.json or {}
    step = (payload.get("step") or "").lower().strip()
    project = (payload.get("project") or "Inventory Management").strip()

    tmpl = AUTO_PROMPTS_BY_STEP.get(step)
    if not tmpl:
        return jsonify({"reply": f"Unknown step '{step}'."}), 400

    query = tmpl.format(project=project)

    # Use the concise web/LLM path (no SQL), but keep recent history for context
    res = run_web_flow(query)  # returns {"reply": "...", "sql": None, "preview": None}
    push_history(query, trim_text(res.get("reply", ""), 500))
    return jsonify({
        "reply": res.get("reply", ""),
        "sql": None,
        "preview": None,
        "mode": "web",
        "sent_query": query
    })

@app.post("/ask")
def ask():
    user_query = (request.json.get("query") or "").strip()
    mode = (request.json.get("mode") or "data").lower()

    if not user_query:
        return jsonify({"reply": "Please enter a question."})

    if user_query.lower() in {"new chat", "/new", "reset"}:
        conversation_history.clear()
        global last_result_df, last_result_sql, last_result_query
        last_result_df = None
        last_result_sql = None
        last_result_query = None
        return jsonify({"reply": "Started a new chat."})

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

        push_history(user_query, trim_text(res.get("reply", ""), 500))
        return jsonify({
            "reply": res.get("reply", ""),
            "sql": res.get("sql"),
            "preview": res.get("preview"),
            "mode": mode
        })
    except Exception as e:
        err = f"⚠️ Error: {str(e)}"
        push_history(user_query, err)
        return jsonify({"reply": err})
    
# ===============================
# Entrypoint
# ===============================
if __name__ == "__main__":
    # pip install flask pandas numpy duckdb python-dotenv groq requests
    app.run(host="0.0.0.0", port=5000, debug=True)
