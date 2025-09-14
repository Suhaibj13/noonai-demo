# analysis_engine.py
# Noon AI — generic + domain analyzers for "Analysis" mode.
# Works even if column names differ slightly (fuzzy resolver).
# Safe to import; does nothing until you call run_analysis_router().

from __future__ import annotations
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Canonical dataset -> default CSV filename in repo
DATASET_FILES = {
    "orders": "orders.csv",
    "inventory": "inventory.csv",
    "mg": "minimum_guarantee.csv",
    "minimum_guarantee": "minimum_guarantee.csv",
}

# Column name synonyms (soft/fuzzy matching)
COLS = {
    "orders": {
        "order_id": ["order_id", "id_order", "orderid"],
        "customer_id": ["customer_id", "id_customer", "customer"],
        "product_id": ["product_id", "sku", "item_id", "product"],
        "qty": ["quantity", "qty", "units", "qty_sold"],
        "value": ["order_value", "amount", "total", "value"],
        "timestamp": ["order_date", "order_datetime", "timestamp", "created_at", "date"],
    },
    "inventory": {
        "product_id": ["product_id", "sku", "item_id", "product"],
        "on_hand": ["on_hand", "stock", "qty_on_hand", "quantity", "available_qty", "balance"],
        "inbound": ["inbound", "incoming", "po_inbound", "on_the_way", "expected_receipts"],
        "updated_at": ["updated_at", "asof", "snapshot_date", "date"],
    },
    "mg": {
        "id_user": ["id_user", "rider_id", "da_id"],
        "vendor_name": ["vendor_name", "vendor", "fleet_partner"],
        "city": ["city", "area", "zone"],
        "rate": ["rate"],
        "attendance": ["attendance"],
        "perfect_attendance": ["perfect_attendance", "pa_needed", "pa"],
        "mg_eligible": ["mg_eligible", "eligible_mg", "mg_month"],
        "total_payout": ["total_payout", "payout", "amount"],
    },
}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column by exact or contains match (case-insensitive)."""
    if df is None or df.empty:
        return None
    lookup = {c.lower().strip(): c for c in df.columns}
    # exact match
    for cand in candidates:
        c = lookup.get(cand.lower())
        if c:
            return c
    # soft contains
    for cand in candidates:
        cl = cand.lower()
        for k, v in lookup.items():
            if cl in k:
                return v
    return None


def load_dataset_frame(name: str, file_hints: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Resolve and load a CSV for a dataset name.

    Order of precedence:
    1) file_hints (exact filenames/paths)
    2) default mapping DATASET_FILES
    3) any CSV in CWD with dataset keyword in its filename
    4) "name" itself if it's a path to a CSV
    """
    # 1) explicit hints
    if file_hints:
        for f in file_hints:
            if isinstance(f, str) and os.path.exists(f):
                return pd.read_csv(f)

    # 2) repo default filename
    default_file = DATASET_FILES.get(name.lower())
    if default_file and os.path.exists(default_file):
        return pd.read_csv(default_file)

    # 3) fuzzy filename search
    for f in os.listdir("."):
        if f.lower().endswith(".csv") and name.lower() in f.lower():
            return pd.read_csv(f)

    # 4) direct path
    if os.path.exists(name):
        return pd.read_csv(name)

    raise FileNotFoundError(
        f"[analysis] No CSV found for dataset '{name}'. "
        f"Place a CSV in repo or pass fileHints."
    )


def detect_datasets_and_task(q: str, provided: Optional[List[str]] = None) -> Tuple[List[str], str, Dict]:
    """Very small NL router. You can add more tasks later."""
    ql = (q or "").lower()
    datasets = list(provided) if provided else []

    # auto dataset detection
    if not datasets:
        if any(w in ql for w in ["order", "sales"]):
            datasets.append("orders")
        if any(w in ql for w in ["inventory", "stock", "out of stock", "stockout"]):
            datasets.append("inventory")
        if any(w in ql for w in ["mg", "minimum guarantee", "minimum_guarantee", "payout"]):
            datasets.append("mg")

    # extract horizon for stockout requests
    horizon_days = None
    m = re.search(r"next\s+(\d+)\s+days", ql)
    if m:
        horizon_days = int(m.group(1))

    # task detection
    if "out of stock" in ql or "stockout" in ql:
        task = "forecast_stockouts"
        params = {"horizon_days": horizon_days or 10}
    elif any(k in ql for k in ["unique customers", "aov", "repeat"]):
        task = "orders_kpis"
        params = {}
    elif any(k in ql for k in ["mg", "minimum guarantee", "payout"]):
        task = "mg_insights"
        params = {}
    else:
        task = "generic_profile"
        params = {}

    # dedupe while preserving order
    datasets = list(dict.fromkeys(datasets))
    return datasets, task, params


# ---------- ANALYZERS ----------

def analyze_generic_profile(df: pd.DataFrame) -> dict:
    kpis = {"rows": int(len(df)), "cols": int(df.shape[1])}
    insights: List[str] = []

    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        desc = df[numeric].describe().T.reset_index()
        for _, r in desc.head(5).iterrows():
            try:
                insights.append(f"{r['index']}: mean={r['mean']:.2f}, p50={r['50%']:.2f}")
            except Exception:
                continue
    else:
        for c in df.columns[:5]:
            vc = df[c].astype(str).value_counts().head(1)
            if len(vc):
                insights.append(f"Most common {c} = {vc.index[0]} ({int(vc.iloc[0])} rows)")

    return {"type": "insights", "kpis": kpis, "insights": insights}


def analyze_orders_kpis(df_orders: pd.DataFrame) -> dict:
    oid = _find_col(df_orders, COLS["orders"]["order_id"]) or df_orders.columns[0]
    cid = _find_col(df_orders, COLS["orders"]["customer_id"])
    val = _find_col(df_orders, COLS["orders"]["value"])
    ts  = _find_col(df_orders, COLS["orders"]["timestamp"])

    total_orders = int(df_orders.shape[0])
    uniq_cust = int(df_orders[cid].nunique()) if cid in df_orders else None
    aov = float(df_orders[val].mean()) if val in df_orders else None

    repeat_rate = None
    if cid in df_orders and oid in df_orders:
        g = df_orders.groupby(cid)[oid].nunique()
        repeat_rate = float((g > 1).mean())

    insights: List[str] = []
    if aov is not None: insights.append(f"AOV ≈ {aov:,.2f}")
    if repeat_rate is not None: insights.append(f"Repeat-customer rate ≈ {repeat_rate*100:.1f}%")
    if ts in df_orders:
        d = df_orders.copy()
        d[ts] = pd.to_datetime(d[ts], errors="coerce")
        if d[ts].notna().any():
            insights.append(f"Range: {d[ts].min().date()} → {d[ts].max().date()}")

    return {
        "type": "insights",
        "kpis": {
            "total_orders": total_orders,
            "unique_customers": uniq_cust,
            "aov": aov,
            "repeat_rate": repeat_rate,
        },
        "insights": insights,
    }


def analyze_mg(df_mg: pd.DataFrame) -> dict:
    elig = _find_col(df_mg, COLS["mg"]["mg_eligible"])
    payout = _find_col(df_mg, COLS["mg"]["total_payout"])
    city = _find_col(df_mg, COLS["mg"]["city"])
    vendor = _find_col(df_mg, COLS["mg"]["vendor_name"])
    rate = _find_col(df_mg, COLS["mg"]["rate"])

    total_da = int(df_mg.shape[0])
    elig_rate = float(df_mg[elig].astype(float).mean()) if elig in df_mg else None
    total_payout = float(df_mg[payout].sum()) if payout in df_mg else None
    avg_rate = float(df_mg[rate].mean()) if rate in df_mg else None

    insights: List[str] = []
    if elig_rate is not None: insights.append(f"MG eligibility rate ≈ {elig_rate*100:.1f}%")
    if total_payout is not None: insights.append(f"Total payout ≈ {total_payout:,.0f}")

    def topk(col):
        if col in df_mg and payout in df_mg:
            s = df_mg.groupby(col)[payout].sum().sort_values(ascending=False).head(3)
            return [{"key": str(i), "payout": float(v)} for i, v in s.items()]
        return None

    return {
        "type": "insights",
        "kpis": {"total_da": total_da, "eligible_rate": elig_rate, "total_payout": total_payout, "avg_rate": avg_rate},
        "breakdowns": {"by_city": topk(city), "by_vendor": topk(vendor)},
        "insights": insights,
    }


def analyze_stockouts(df_orders: pd.DataFrame, df_inventory: pd.DataFrame, horizon_days: int = 10) -> dict:
    pid_o = _find_col(df_orders, COLS["orders"]["product_id"])
    qty_o = _find_col(df_orders, COLS["orders"]["qty"])
    ts_o  = _find_col(df_orders, COLS["orders"]["timestamp"])

    pid_i = _find_col(df_inventory, COLS["inventory"]["product_id"])
    onh   = _find_col(df_inventory, COLS["inventory"]["on_hand"])
    inbound = _find_col(df_inventory, COLS["inventory"]["inbound"])

    if not (pid_o and qty_o and ts_o and pid_i and onh):
        return {"type": "insights", "kpis": {}, "insights": ["Missing required columns for stockout forecast."]}

    o = df_orders.copy()
    o[ts_o] = pd.to_datetime(o[ts_o], errors="coerce")
    o = o.dropna(subset=[ts_o])

    # use last 30 days for daily velocity
    if o.empty:
        return {"type": "insights", "kpis": {}, "insights": ["No recent orders available for velocity."]}

    cutoff = o[ts_o].max() - pd.Timedelta(days=30)
    window = o[o[ts_o] >= cutoff]

    daily = (
        window.groupby([pid_o, window[ts_o].dt.date])[qty_o].sum()
        .groupby(level=0)
        .mean()
        .rename("avg_daily_units")
        .reset_index()
    )

    inv_cols = [pid_i, onh] + ([inbound] if inbound in df_inventory.columns else [])
    inv = df_inventory[inv_cols].copy()
    inv = inv.rename(columns={pid_i: "product_id", onh: "on_hand"})
    if inbound in inv.columns:
        inv = inv.rename(columns={inbound: "inbound"})

    merged = inv.merge(daily.rename(columns={pid_o: "product_id"}), on="product_id", how="left")
    merged["avg_daily_units"] = merged["avg_daily_units"].fillna(0.0)
    merged["projected_consumption"] = merged["avg_daily_units"] * float(horizon_days)
    merged["projected_balance"] = merged["on_hand"] + merged.get("inbound", 0) - merged["projected_consumption"]

    will_stockout = int((merged["projected_balance"] < 0).sum())
    top_risk = merged.sort_values("projected_balance").head(10)
    table = top_risk[["product_id", "on_hand", "avg_daily_units", "projected_balance"]].to_dict(orient="records")

    insights = [
        f"Products expected to stock out in next {horizon_days} days: {will_stockout}",
        f"Median daily velocity across catalog: {merged['avg_daily_units'].median():.2f} units/day",
    ]

    return {
        "type": "insights",
        "kpis": {"risk_stockouts": will_stockout, "horizon_days": int(horizon_days)},
        "insights": insights,
        "tables": {"top_risk": table},
    }


def run_analysis_router(q: str, datasets: Optional[List[str]] = None, file_hints: Optional[List[str]] = None) -> dict:
    """
    Central dispatcher. Safe to call.
    Only handles Analysis mode. For Data/Web modes, let your existing pipeline run.
    """
    ds, task, params = detect_datasets_and_task(q, provided=datasets)

    # load frames
    frames: Dict[str, pd.DataFrame] = {}
    for name in ds:
        frames[name] = load_dataset_frame(name, file_hints=file_hints)

    # tasks
    if task == "forecast_stockouts":
        if "orders" in frames and "inventory" in frames:
            return analyze_stockouts(frames["orders"], frames["inventory"], params.get("horizon_days", 10))
        return {"type": "insights", "insights": ["Need both orders and inventory datasets for stockout forecast."]}

    if task == "orders_kpis" and "orders" in frames:
        return analyze_orders_kpis(frames["orders"])

    if task == "mg_insights" and ("mg" in frames or "minimum_guarantee" in frames):
        return analyze_mg(frames.get("mg") or frames.get("minimum_guarantee"))

    # Fallback: generic profiling of the first dataset (works for any CSV)
    if ds:
        return analyze_generic_profile(frames[ds[0]])

    return {"type": "insights", "insights": ["No dataset detected. Try: 'analyze orders' or 'analyze inventory'."]}
