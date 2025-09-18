# analysis_planner.py
# Plan → Execute → Answer helper used by Analysis mode.
# Produces a structured plan (JSON) with Groq, validates, and compiles deterministic SQL.

from __future__ import annotations
import os, re, json
from typing import Dict, List, Optional, Tuple

# --- Groq client (self-contained; no circular import with main.py) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def _llm(messages: List[Dict], temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# --- utils ---
_SQL_OPS = {"=", "!=", ">", ">=", "<", "<="}

def _sql_quote(val) -> str:
    """Return a SQL string literal with single quotes escaped."""
    s = str(val)
    return "'" + s.replace("'", "''") + "'"
    
def _quote_ident(name: str) -> str:
    # Quote if it contains any non-word or starts with a digit
    if not name: return '""'
    if re.search(r"\W", name) or re.match(r"^\d", name):
        return f'"{name}"'
    return name

def _qual(table: str, col: str) -> str:
    return f'{table}.{_quote_ident(col)}'

def _json_only(text: str) -> str:
    # Try to extract the outermost JSON object/array from the response
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    return m.group(1).strip() if m else text

def _schema_map(inv, ords, mg) -> Dict[str, List[str]]:
    return {
        "inventory": list(inv.columns),
        "orders": list(ords.columns),
        "mg": list(mg.columns),
    }

# --- prompt builders ---
def _human_cols(cols):
    # Quote columns that contain spaces/symbols for display in the schema block
    return ", ".join([f'"{c}"' if re.search(r"\W", c) else c for c in cols])
    
def _plan_prompt(question: str, schema_cols: Dict[str, List[str]]) -> List[Dict]:
    # Render compact schema list for the model
    schema_txt = "\n".join(f"{k}: {_human_cols(cols)}" for k, cols in schema_cols.items())

    sys = (
        "You are a data analyst. Given a QUESTION and the SCHEMA (tables: mg, orders, inventory), "
        "return a STRICT JSON plan describing the analysis. Use only exact column names from the schema. "
        "If a column has spaces or symbols, show it quoted in the JSON (e.g., \"DA Level Final Amount\").\n\n"
        "Return ONLY JSON with the keys:\n"
        "{\n"
        '  "dataset": ["mg"],            // one or more of: "mg", "orders", "inventory"\n'
        '  "goal": "short description",\n'
        '  "metrics": [{"name":"total_paid","expr":"sum(\\"DA Level Final Amount\\")"}],\n'
        '  "dimensions": ["Vendor Name"],\n'
        '  "filters": [{"column":"MG Eligible","op":"=","value":0}],\n'
        '  "time_window": null,\n'
        '  "topn": 10,\n'
        '  "sort": [{"by":"total_paid","dir":"desc"}],\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "needs_clarification": false,\n'
        '  "ambiguities": []\n'
        "}\n"
        "Rules:\n"
        "- Use only columns from the schema (exact names). If unsure, set needs_clarification=true and describe in ambiguities.\n"
        '- In metric expr, QUOTE multi-word columns, e.g., sum("DA Level Final Amount").\n'
        "- Prefer single table when possible; join logic is out of scope for now.\n"
        "- Do not include markdown or commentary; return pure JSON."
    )
    usr = f"SCHEMA:\n{schema_txt}\n\nQUESTION:\n{question}\n"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

# --- planner + compiler ---
def generate_plan(question: str, inv, ords, mg) -> Dict:
    schema_cols = _schema_map(inv, ords, mg)
    msg = _plan_prompt(question, schema_cols)
    raw = _llm(msg, temperature=0.0)
    try:
        plan = json.loads(_json_only(raw))
    except Exception as e:
        return {"ok": False, "error": f"Plan JSON parse error: {e}", "raw": raw}

    # light validation
    ds = plan.get("dataset") or []
    if isinstance(ds, str): ds = [ds]
    plan["dataset"] = ds

    # only support single table for now; otherwise ask for clarification
    if not ds:
        plan["needs_clarification"] = True
        plan.setdefault("ambiguities", []).append("Dataset not identified (mg/orders/inventory).")
    elif len(ds) > 1:
        plan["needs_clarification"] = True
        plan.setdefault("ambiguities", []).append("Multiple datasets requested; joins are out of scope for now.")

    # verify columns exist
    cols_by_table = schema_cols.get(ds[0], []) if ds else []
    colset = set(cols_by_table)
    def _check_col(c):
        return c in colset

    bad_filters = []
    for f in plan.get("filters", []):
        if not (isinstance(f, dict) and f.get("column") and f.get("op") in _SQL_OPS):
            bad_filters.append(str(f))
            continue
        if not _check_col(f["column"]):
            bad_filters.append(f['column'])
    if bad_filters:
        plan["needs_clarification"] = True
        plan.setdefault("ambiguities", []).append(f"Unknown or invalid filters: {', '.join(bad_filters)}")

    # ok to return
    return {"ok": True, "plan": plan, "confidence": float(plan.get("confidence") or 0.0)}

def compile_plan_to_sql(plan: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """Deterministically compile a simple plan (single dataset) to DuckDB SQL."""
    ds = plan.get("dataset") or []
    if not ds or len(ds) != 1:
        return False, None, "Unsupported dataset selection."
    table = ds[0]  # 'mg' | 'orders' | 'inventory'

    dims: List[str] = plan.get("dimensions") or []
    metrics: List[Dict] = plan.get("metrics") or []
    filters: List[Dict] = plan.get("filters") or []
    sort: List[Dict] = plan.get("sort") or []
    topn = plan.get("topn")

    # SELECT list
    sel_parts: List[str] = []
    group_by: List[str] = []

    for d in dims:
        sel_parts.append(f"{_qual(table, d)} AS {_quote_ident(d)}")
        group_by.append(_qual(table, d))

    for m in metrics:
        name = m.get("name") or "m"
        expr = (m.get("expr") or "").strip()
        if not expr:
            return False, None, "Metric expression missing."
        # We expect expr already quoted by the prompt (e.g., sum("DA Level Final Amount"))
        sel_parts.append(f"{expr} AS {_quote_ident(name)}")

    if not sel_parts:
        return False, None, "No dimensions or metrics provided."

    sql = f"SELECT\n  " + ",\n  ".join(sel_parts) + f"\nFROM {table}\n"

    # WHERE
    where_parts = []
    for f in filters:
        col = f.get("column"); op = f.get("op"); val = f.get("value")
        if col and op in _SQL_OPS:
            colq = _qual(table, col)
            if isinstance(val, str):
            # if looks like a number, keep as number; else quote as string
            if re.fullmatch(r"-?\d+(\.\d+)?", val):
                where_parts.append(f"{colq} {op} {val}")
            else:
                safe_val = _sql_quote(val)  # pre-escape, no backslashes in f-string
                where_parts.append(f"{colq} {op} {safe_val}")
            elif val is None:
                # treat NULL/IS comparisons separately if needed; simplest: skip
                continue
            else:
                where_parts.append(f"{colq} {op} {val}")
    if where_parts:
        sql += "WHERE " + " AND ".join(where_parts) + "\n"

    # GROUP BY
    if group_by:
        sql += "GROUP BY " + ", ".join(group_by) + "\n"

    # ORDER BY
    if sort:
        s = []
        for srt in sort:
            by = srt.get("by")
            direction = (srt.get("dir") or "desc").upper()
            if by:
                s.append(f"{_quote_ident(by)} {('DESC' if direction.startswith('D') else 'ASC')}")
        if s:
            sql += "ORDER BY " + ", ".join(s) + "\n"

    # LIMIT
    if isinstance(topn, int) and topn > 0:
        sql += f"LIMIT {int(topn)}\n"

    return True, sql.strip(), None

# --- public entry point ---
def plan_with_groq(question: str, inv, ords, mg) -> Dict:
    """
    Returns:
      { ok, needs_clarification, clarify, confidence, sql, plan, error }
    """
    if not GROQ_API_KEY:
        return {"ok": False, "error": "GROQ_API_KEY not set"}
    p = generate_plan(question, inv, ords, mg)
    if not p.get("ok"):
        return p

    plan = p["plan"]
    if plan.get("needs_clarification"):
        ambiguities = plan.get("ambiguities") or []
        clarify = "To proceed, please confirm: " + "; ".join(ambiguities)
        return {
            "ok": True,
            "needs_clarification": True,
            "clarify": clarify,
            "confidence": float(plan.get("confidence") or 0.0),
            "plan": plan,
            "sql": None,
        }

    ok, sql, err = compile_plan_to_sql(plan)
    if not ok:
        return {"ok": False, "error": err or "Failed to compile plan", "plan": plan}

    return {
        "ok": True,
        "needs_clarification": False,
        "confidence": float(plan.get("confidence") or 0.0),
        "sql": sql,
        "plan": plan,
    }
