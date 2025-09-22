# demo_snapshots.py — Snapshot demo (bullet-proof answers, no SQL)
from __future__ import annotations
import json, os, re
from typing import Dict, List, Optional

_FACTS_PATH = os.getenv("DEMO_FACTS_PATH", "demo_facts_mg.json")

# --------- token helpers ----------
def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())

def _match_groups(text: str, groups: List[List[str]]) -> bool:
    t = _norm(text)
    for g in groups:
        if not any(_norm(tok) in t for tok in g):
            return False
    return True

# --------- patterns (10 demo stories) ----------
# Each "groups" entry means: at least one token from EACH sublist must appear
PATTERNS = [
    ("mg_non_eligible_paid", [
        ["vendor","vendors"], ["not eligible","non eligible","ineligible"], ["mg"], ["paid","payout","amount"]
    ]),
    ("mg_attendance_below_threshold", [
        ["attendance","working days","required days","pa"], ["below","short","fell short"]
    ]),
    ("mg_top5_share", [
        ["top","top 5","five","rank","share","percent"], ["mg"]
    ]),
    ("mg_non_eligible_avg_per_da", [
        ["not eligible","non eligible","ineligible"], ["mg"], ["average","avg","per da","per agent"]
    ]),
    ("mg_monthly_spike_drop", [
        ["spike","drop","increase","decrease","change"], ["month","monthly"]
    ]),
    ("mg_top_vendors_in_swing_months", [
        ["top vendors"], ["those months","swing months","spike","drop","increase","decrease"]
    ]),
    ("mg_over_mg_outliers", [
        ["final payout","final amount"], ["exceed","more than","over"], ["mg"]
    ]),
    ("mg_early_tenure_repeated_mg", [
        ["first 30 days","first month","within 30"], ["repeated mg","multiple mg","mg more than once","got mg"]
    ]),
    ("mg_rate_outliers", [
        ["rate"], ["outlier","unusual","anomaly","zscore","z score"]
    ]),
    ("mg_multi_vendor_same_month", [
        ["same month","in the same month"], ["multiple vendors","more than one vendor","two vendors"]
    ]),
    ("mg_city_outliers", [
        ["city","cities"], ["unusual","high","low","higher","lower"], ["eligibility","payout","mg rate","mg payment"]
    ]),
]

# --------- facts loader ----------
_FACTS: Dict[str, dict] = {}
def _load_facts() -> Dict[str, dict]:
    global _FACTS
    if _FACTS:
        return _FACTS
    if not os.path.exists(_FACTS_PATH):
        return {}
    with open(_FACTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # make id → card map
    _FACTS = {card["id"]: card for card in data.get("cards", [])}
    return _FACTS

# --------- public API ----------
def detect_snapshot_card(user_question: str) -> Optional[dict]:
    """Return the facts card dict if the question matches; else None."""
    facts = _load_facts()
    if not facts:
        return None
    for card_id, groups in PATTERNS:
        if _match_groups(user_question, groups):
            return facts.get(card_id)
    return None

def _format_table(rows: List[List]) -> str:
    if not rows:
        return ""
    # very simple monospace table: header then up to 10 rows
    header = rows[0]
    body   = rows[1:11]
    widths = [max(len(str(x)) for x in col) for col in zip(*([header] + body))]
    def line(r): return " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
    out = [line(header)]
    out.append("-+-".join("-"*w for w in widths))
    out.extend(line(r) for r in body)
    return "\n".join(out)

def render_snapshot_answer(card: dict) -> Dict[str, str]:
    """
    Build a natural-language answer purely from the JSON card.
    Falls back gracefully if some fields are missing.
    """
    if not card:
        return {}
    title = card.get("title", "Analysis")
    narrative = card.get("narrative") or {}
    headline  = narrative.get("headline") or f"Snapshot: {title}"
    bullets   = narrative.get("bullets") or []
    reply = "Answer: " + headline.strip()
    for b in bullets[:3]:
        reply += f"\n- {b.strip()}"
    preview = ""
    st = card.get("short_table")
    if isinstance(st, list) and st:
        preview = _format_table(st)
    return {"reply": reply, "preview": preview}
