# demo_intents.py
# Demo answers driven by your real CSVs (first 1,000 rows).
# Intercepts a few exact questions per mode.
from __future__ import annotations
import os, re
from typing import Any, Dict, Callable, Tuple, Optional
import pandas as pd

# ---------- load & normalize ---------------------------------------------------
_DEMO_CACHE: Dict[str, pd.DataFrame] = {}

def _read_head(fname: str, n: int = 1000) -> pd.DataFrame:
    if fname in _DEMO_CACHE:
        return _DEMO_CACHE[fname]
    if not os.path.exists(fname):
        _DEMO_CACHE[fname] = pd.DataFrame()
        return _DEMO_CACHE[fname]
    df = pd.read_csv(fname).head(n).copy()
    # best-effort parse common date columns
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "created" in cl or "time" in cl or "sold" in cl or "delivered" in cl:
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception:
                pass
    _DEMO_CACHE[fname] = df
    return df

def _orders() -> pd.DataFrame:   return _read_head("orders.csv")
def _inventory() -> pd.DataFrame:return _read_head("inventory.csv")
def _mg() -> pd.DataFrame:       return _read_head("minimum_guarantee.csv")

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _df_preview_string(df: pd.DataFrame, rows: int = 50) -> str:
    if df is None or df.empty:
        return "(no rows)"
    try:
        return df.head(rows).to_string(index=False)
    except Exception:
        return str(df.head(rows))

# ---------- light column sniffers ---------------------------------------------
def _find_col(df: pd.DataFrame, *cands: str, numeric: bool | None = None) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    # first pass: name contains any candidate
    for name in cols:
        nm = name.lower()
        if any(k in nm for k in cands):
            if numeric is True and not pd.api.types.is_numeric_dtype(df[name]): 
                continue
            if numeric is False and pd.api.types.is_numeric_dtype(df[name]): 
                continue
            return name
    # fallback: exact lowercase match
    for name in cols:
        if name.lower() in cands:
            return name
    return None

# ---------- DATA mode demos (return preview only) -----------------------------
def _data_show(df: pd.DataFrame) -> Dict[str, Any]:
    preview = _df_preview_string(df, 50)
    return {
        "reply": f"Executed SQL. Showing first {min(50, len(df)) if df is not None else 0} rows.",
        "sql": "",                # intentionally blank for demo
        "preview": preview,
    }

DATA_DEMOS: Dict[str, Callable[[], Dict[str, Any]]] = {
    _norm("Give me total purchase for each customer"):       lambda: _data_show(_orders()),
    _norm("Show total orders by day"):                       lambda: _data_show(_orders()),
    _norm("Top 5 sku by revenue"):                           lambda: _data_show(_orders()),
    _norm("Compare orders first half vs second half of month"): lambda: _data_show(_orders()),
    _norm("Show minimum guarantee payout by vendor"):        lambda: _data_show(_mg()),
    # You can add synonyms like:
    _norm("show orders table"):                              lambda: _data_show(_orders()),
    _norm("show inventory table"):                           lambda: _data_show(_inventory()),
    _norm("show minimum guarantee table"):                   lambda: _data_show(_mg()),
}

# ---------- ANALYSIS mode demos (compute numbers) -----------------------------
def _fmt_num(x) -> str:
    try:
        return f"{float(x):,.2f}" if abs(float(x) - int(float(x))) > 1e-9 else f"{int(float(x)):,}"
    except Exception:
        return str(x)

def _analysis_orders_totals() -> Dict[str, Any]:
    df = _orders()
    if df is None or df.empty:
        return {"reply": "I couldn't compute this from the demo rows.", "sql": None, "preview": None}

    val_col = _find_col(df, "order_value", "amount", "price", "revenue", "total")
    if not val_col:
        return {"reply": "I couldn't compute totals (no amount/price column found).", "sql": None, "preview": _df_preview_string(df)}

    n_orders = len(df)
    total_rev = df[val_col].fillna(0).astype(float).sum()
    aov = total_rev / n_orders if n_orders else 0

    reply = (
        f"Answer: total revenue = { _fmt_num(total_rev) }, AOV = { _fmt_num(aov) } "
        f"(over { _fmt_num(n_orders) } orders).\n"
        "- Topline looks at first 1,000 demo rows only.\n"
        "- Use Data mode for full detail.\n"
    )
    return {"reply": reply, "sql": None, "preview": _df_preview_string(df)}

def _analysis_first_vs_second_half() -> Dict[str, Any]:
    df = _orders()
    if df is None or df.empty:
        return {"reply": "I couldn't compute this from the demo rows.", "sql": None, "preview": None}
    date_col = _find_col(df, "date", "order_date", "created", "time")
    val_col  = _find_col(df, "order_value", "amount", "price", "revenue", "total")
    if not date_col:
        return {"reply": "I couldn't find a date column for the split.", "sql": None, "preview": _df_preview_string(df)}
    tmp = df[[date_col]].copy()
    tmp["__day"] = pd.to_datetime(df[date_col], errors="coerce").dt.day
    tmp["__period"] = tmp["__day"].apply(lambda d: "First half" if pd.notna(d) and d <= 15 else "Second half")
    if val_col:
        tmp["__val"] = pd.to_numeric(df[val_col], errors="coerce")
    g = tmp.groupby("__period", dropna=False)
    cnt = g["__day"].count()
    rev = g["__val"].sum() if "__val" in tmp else None

    reply = f"Answer: orders — First half = { _fmt_num(cnt.get('First half',0)) }, Second half = { _fmt_num(cnt.get('Second half',0)) }."
    if rev is not None:
        reply += f" Revenue — First half = { _fmt_num(rev.get('First half',0)) }, Second half = { _fmt_num(rev.get('Second half',0)) }."
    reply += "\n- Split computed on first 1,000 demo rows."
    return {"reply": reply, "sql": None, "preview": _df_preview_string(df)}

def _analysis_top5_sku_share() -> Dict[str, Any]:
    df = _orders()
    if df is None or df.empty:
        return {"reply": "I couldn't compute this from the demo rows.", "sql": None, "preview": None}
    sku_col = _find_col(df, "sku", "sku_id", "product", "product_id", "item_id")
    val_col = _find_col(df, "order_value", "amount", "price", "revenue", "total")
    if not (sku_col and val_col):
        return {"reply": "I couldn't find SKU and value columns.", "sql": None, "preview": _df_preview_string(df)}
    g = df.groupby(sku_col, dropna=False)[val_col].sum().sort_values(ascending=False)
    top5 = g.head(5)
    share = (top5.sum() / g.sum()) * 100 if g.sum() else 0.0
    reply = (
        f"Answer: top-5 SKU revenue share = { _fmt_num(share) }%.\n"
        f"- Top SKU: { top5.index[0] if len(top5) else 'n/a' } with { _fmt_num(top5.iloc[0] if len(top5) else 0) }.\n"
        f"- Total SKUs in demo: { _fmt_num(g.shape[0]) }."
    )
    return {"reply": reply, "sql": None, "preview": _df_preview_string(df)}

def _analysis_mg_utilization() -> Dict[str, Any]:
    df = _mg()
    if df is None or df.empty:
        return {"reply": "I couldn't compute this from the demo rows.", "sql": None, "preview": None}
    mg_amt   = _find_col(df, "mg", "mg_amount", "minimum", "guarantee", "committed")
    payout   = _find_col(df, "payout", "actual_payout", "paid", "disbursed")
    if not (mg_amt and payout):
        return {"reply": "I couldn't find MG amount and payout columns.", "sql": None, "preview": _df_preview_string(df)}
    total_mg  = pd.to_numeric(df[mg_amt], errors="coerce").fillna(0).sum()
    total_pay = pd.to_numeric(df[payout], errors="coerce").fillna(0).sum()
    util = (total_pay / total_mg * 100) if total_mg else 0.0
    reply = (
        f"Answer: MG utilization = { _fmt_num(util) }% "
        f"(committed { _fmt_num(total_mg) } vs paid { _fmt_num(total_pay) }).\n"
        "- Computed on first 1,000 demo rows."
    )
    return {"reply": reply, "sql": None, "preview": _df_preview_string(df)}

def _analysis_inventory_health() -> Dict[str, Any]:
    df = _inventory()
    if df is None or df.empty:
        return {"reply": "I couldn't compute this from the demo rows.", "sql": None, "preview": None}
    qty_col = _find_col(df, "qty", "quantity", "stock", "on_hand", "onhand", "available", numeric=True)
    sku_col = _find_col(df, "sku", "sku_id", "product", "product_id", "item_id")
    total_skus = df[sku_col].nunique() if sku_col else len(df)
    avg_stock = pd.to_numeric(df[qty_col], errors="coerce").mean() if qty_col else None
    stockouts = int((pd.to_numeric(df[qty_col], errors="coerce") <= 0).sum()) if qty_col else None
    reply = f"Answer: total SKUs = { _fmt_num(total_skus) }"
    if avg_stock is not None:
        reply += f", average stock = { _fmt_num(avg_stock) }"
    if stockouts is not None:
        reply += f", stock-outs = { _fmt_num(stockouts) }"
    reply += ".\n- Based on first 1,000 demo rows."
    return {"reply": reply, "sql": None, "preview": _df_preview_string(df)}

ANALYSIS_DEMOS: Dict[str, Callable[[], Dict[str, Any]]] = {
    _norm("What is total revenue and average order value?"): _analysis_orders_totals,
    _norm("First vs second half of month — orders and revenue"): _analysis_first_vs_second_half,
    _norm("Top 5 SKUs by revenue and % share"): _analysis_top5_sku_share,
    _norm("Minimum guarantee utilization summary"): _analysis_mg_utilization,
    _norm("Inventory health summary"): _analysis_inventory_health,
    # Simple ASCII variant of the en-dash:
    _norm("First vs second half of month - orders and revenue"): _analysis_first_vs_second_half,
}

# ---------- public entry ------------------------------------------------------
def get_demo_response(mode: str, user_input: str) -> Dict[str, Any] | None:
    key = _norm(user_input)
    m = (mode or "").strip().lower()
    if m == "data":
        fn = DATA_DEMOS.get(key)
        return fn() if fn else None
    if m == "analysis":
        fn = ANALYSIS_DEMOS.get(key)
        return fn() if fn else None
    return None
