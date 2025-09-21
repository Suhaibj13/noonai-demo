# mg_canned.py
# Canned MG demo queries + lightweight detector.
from __future__ import annotations
import re
from typing import Dict, List, Optional

# ---------- tiny matcher helpers ----------
def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())

def _matches(text: str, must_groups: List[List[str]]) -> bool:
    """
    true if text contains at least one token from EACH group in must_groups.
    Example: [["vendor","vendors"], ["not eligible","ineligible"], ["mg"]]
    """
    t = _norm(text)
    for group in must_groups:
        ok = False
        for token in group:
            if _norm(token) in t:
                ok = True
                break
        if not ok:
            return False
    return True

# ---------- canned cards ----------
# Each card has:
#   id: stable key
#   patterns: list of token-groups (see _matches)
#   sql: exact DuckDB SQL for table mg
#   hint: short explainer hint to steer answer tone
CARDS: List[Dict] = [
    {
        "id": "mg_non_eligible_paid",
        "patterns": [
            [["vendor","vendors"], ["not eligible","non eligible","ineligible"], ["mg"], ["paid","payout","amount"]],
        ],
        "sql": """
SELECT
  "Vendor Name" AS vendor,
  COUNT(DISTINCT id_user) AS das,
  SUM("MG Amount")        AS mg_paid,
  SUM("DA Level Final Amount") AS total_final
FROM mg
WHERE "MG Eligible" = 0 AND "MG Amount" > 0
GROUP BY "Vendor Name"
ORDER BY mg_paid DESC;
""",
        "hint": "Top vendors by MG paid to non-eligible agents.",
    },
    {
        "id": "mg_attendance_below_threshold",
        "patterns": [
            [["attendance","working days","required days","pa"], ["below","short","fell short"]],
        ],
        "sql": """
SELECT
  "Vendor Name" AS vendor,
  COUNT(DISTINCT id_user) AS das_below_threshold,
  SUM(CASE WHEN "MG Amount" > 0 THEN 1 ELSE 0 END) AS rows_with_mg
FROM mg
WHERE COALESCE("Total Attendance", 0) < COALESCE("PA Needed", 0)
GROUP BY "Vendor Name"
ORDER BY das_below_threshold DESC;
""",
        "hint": "Agents below attendance requirement; which vendors see this most.",
    },
    {
        "id": "mg_top5_share",
        "patterns": [
            [["top","top 5","five","rank","share","percent"], ["mg"]],
        ],
        "sql": """
SELECT vendor, mg_paid,
       mg_paid / NULLIF(total_mg, 0) AS share_of_total
FROM (
  SELECT
    "Vendor Name" AS vendor,
    SUM("MG Amount") AS mg_paid,
    SUM(SUM("MG Amount")) OVER () AS total_mg
  FROM mg
  GROUP BY "Vendor Name"
)
ORDER BY mg_paid DESC
LIMIT 5;
""",
        "hint": "Top vendors by MG and their share of overall MG.",
    },
    {
        "id": "mg_non_eligible_avg_per_da",
        "patterns": [
            [["not eligible","non eligible","ineligible"], ["mg"], ["average","avg","per da","per agent"]],
        ],
        "sql": """
SELECT
  "Vendor Name" AS vendor,
  COUNT(DISTINCT id_user) AS das,
  SUM("MG Amount") AS mg_paid,
  SUM("MG Amount") / NULLIF(COUNT(DISTINCT id_user), 0) AS avg_mg_per_da
FROM mg
WHERE "MG Eligible" = 0 AND "MG Amount" > 0
GROUP BY "Vendor Name"
ORDER BY mg_paid DESC;
""",
        "hint": "Total and average MG per non-eligible agent, by vendor.",
    },
    {
        "id": "mg_monthly_spike_drop",
        "patterns": [
            [["spike","drop","increase","decrease","change"], ["month","monthly"]],
        ],
        "sql": """
WITH monthly AS (
  SELECT
    CAST("MG Month" AS DATE) AS m,
    SUM("MG Amount") AS mg_paid
  FROM mg
  GROUP BY 1
),
chg AS (
  SELECT
    m, mg_paid,
    LAG(mg_paid) OVER (ORDER BY m) AS prev_paid,
    (mg_paid - LAG(mg_paid) OVER (ORDER BY m)) / NULLIF(LAG(mg_paid) OVER (ORDER BY m), 0) AS pct_change
  FROM monthly
)
SELECT *
FROM chg
WHERE ABS(pct_change) >= 0.30
ORDER BY m;
""",
        "hint": "Months with >30% MG swings.",
    },
    {
        "id": "mg_top_vendors_in_swing_months",
        "patterns": [
            [["top vendors"], ["those months","swing months","spike","drop","increase","decrease"]],
        ],
        "sql": """
WITH monthly AS (
  SELECT
    CAST("MG Month" AS DATE) AS m,
    SUM("MG Amount") AS mg_paid
  FROM mg
  GROUP BY 1
),
swing AS (
  SELECT
    m,
    mg_paid,
    LAG(mg_paid) OVER (ORDER BY m) AS prev_paid,
    (mg_paid - LAG(mg_paid) OVER (ORDER BY m)) / NULLIF(LAG(mg_paid) OVER (ORDER BY m), 0) AS pct_change
  FROM monthly
),
by_month_vendor AS (
  SELECT
    CAST("MG Month" AS DATE) AS m,
    "Vendor Name" AS vendor,
    SUM("MG Amount") AS mg_paid
  FROM mg
  GROUP BY 1, 2
),
ranked AS (
  SELECT bmv.*,
         ROW_NUMBER() OVER (PARTITION BY bmv.m ORDER BY bmv.mg_paid DESC) AS rnk
  FROM by_month_vendor bmv
  JOIN swing s ON bmv.m = s.m
  WHERE ABS(s.pct_change) >= 0.30
)
SELECT m, vendor, mg_paid
FROM ranked
WHERE rnk <= 3
ORDER BY m, rnk;
""",
        "hint": "Top vendors in swing months.",
    },
    {
        "id": "mg_over_mg_outliers",
        "patterns": [
            [["final payout","final amount"], ["exceed","more than","over"], ["mg"]],
        ],
        "sql": """
SELECT
  id_user,
  "Vendor Name" AS vendor,
  SUM("DA Level Final Amount") AS total_final,
  SUM("MG Amount") AS mg_amount,
  (SUM("DA Level Final Amount") - SUM("MG Amount"))
     / NULLIF(SUM("MG Amount"), 0) AS over_mg_ratio
FROM mg
GROUP BY id_user, "Vendor Name"
HAVING SUM("MG Amount") > 0
   AND (SUM("DA Level Final Amount") - SUM("MG Amount"))
       / NULLIF(SUM("MG Amount"), 0) > 0.20
ORDER BY over_mg_ratio DESC
LIMIT 50;
""",
        "hint": "Agents whose final payout exceeds MG by >20%.",
    },
    {
        "id": "mg_early_tenure_repeated_mg",
        "patterns": [
            [["first 30 days","first month","within 30"], ["repeated mg","multiple mg","mg more than once","got mg"]],
        ],
        "sql": """
WITH rows AS (
  SELECT
    id_user,
    "Vendor Name" AS vendor,
    CAST("Joining Date" AS DATE) AS join_dt,
    CAST("MG Month" AS DATE)    AS mg_dt,
    "MG Amount" AS mg_amount
  FROM mg
  WHERE "MG Amount" > 0
),
early AS (
  SELECT *,
         DATEDIFF('day', join_dt, mg_dt) AS days_since_join
  FROM rows
  WHERE join_dt IS NOT NULL AND mg_dt IS NOT NULL
    AND DATEDIFF('day', join_dt, mg_dt) BETWEEN 0 AND 30
)
SELECT vendor, id_user,
       COUNT(*) AS early_mg_count,
       SUM(mg_amount) AS early_mg_paid
FROM early
GROUP BY vendor, id_user
HAVING COUNT(*) >= 2
ORDER BY early_mg_count DESC, early_mg_paid DESC
LIMIT 100;
""",
        "hint": "Repeated MG within first 30 days of joining.",
    },
    {
        "id": "mg_rate_outliers",
        "patterns": [
            [["rate"], ["outlier","unusual","anomaly","zscore","z score"]],
        ],
        "sql": """
WITH stats AS (
  SELECT AVG("Rate") AS mu, STDDEV_SAMP("Rate") AS sigma
  FROM mg
  WHERE "Rate" IS NOT NULL
),
z AS (
  SELECT
    "Vendor Name" AS vendor,
    "Rate",
    ("Rate" - stats.mu) / NULLIF(stats.sigma, 0) AS z
  FROM mg, stats
  WHERE "Rate" IS NOT NULL
)
SELECT
  vendor,
  SUM(CASE WHEN ABS(z) > 3 THEN 1 ELSE 0 END) AS outlier_rows,
  COUNT(*) AS total_rows,
  SUM(CASE WHEN ABS(z) > 3 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS outlier_rate
FROM z
GROUP BY vendor
ORDER BY outlier_rows DESC;
""",
        "hint": "Vendors with unusual Rate values (|z|>3).",
    },
    {
        "id": "mg_multi_vendor_same_month",
        "patterns": [
            [["same month","in the same month"], ["multiple vendors","more than one vendor","two vendors"]],
        ],
        "sql": """
WITH base AS (
  SELECT
    id_user,
    "Vendor Name" AS vendor,
    CAST("MG Month" AS DATE) AS month_dt,
    SUM("MG Amount") AS mg_amount
  FROM mg
  GROUP BY 1, 2, 3
),
overlap AS (
  SELECT
    id_user, month_dt,
    COUNT(DISTINCT vendor) AS vendors_in_month,
    SUM(mg_amount) AS mg_total
  FROM base
  GROUP BY 1, 2
  HAVING COUNT(DISTINCT vendor) > 1
)
SELECT *
FROM overlap
ORDER BY month_dt, id_user
LIMIT 200;
""",
        "hint": "Agents tied to multiple vendors within the same month.",
    },
    {
        "id": "mg_city_outliers",
        "patterns": [
            [["city","cities"], ["unusual","high","low","higher","lower"], ["eligibility","payout","mg rate","mg payment"]],
        ],
        "sql": """
WITH city AS (
  SELECT
    "City" AS city,
    AVG(CASE WHEN "MG Eligible" IS NULL
             THEN NULL
             ELSE CAST("MG Eligible" AS DOUBLE) END) AS eligibility_rate,
    AVG(CASE WHEN "MG Amount" > 0 THEN 1.0 ELSE 0.0 END) AS mg_payment_rate
  FROM mg
  GROUP BY "City"
),
overall AS (
  SELECT
    AVG(CASE WHEN "MG Eligible" IS NULL
             THEN NULL
             ELSE CAST("MG Eligible" AS DOUBLE) END) AS overall_eligibility_rate,
    AVG(CASE WHEN "MG Amount" > 0 THEN 1.0 ELSE 0.0 END) AS overall_mg_payment_rate
  FROM mg
)
SELECT
  c.city,
  c.eligibility_rate,
  c.mg_payment_rate,
  o.overall_eligibility_rate,
  o.overall_mg_payment_rate,
  c.eligibility_rate - o.overall_eligibility_rate AS eligibility_diff,
  c.mg_payment_rate - o.overall_mg_payment_rate AS mg_payment_diff
FROM city c CROSS JOIN overall o
WHERE ABS(c.eligibility_rate - o.overall_eligibility_rate) >= 0.10
   OR ABS(c.mg_payment_rate - o.overall_mg_payment_rate) >= 0.10
ORDER BY ABS(c.mg_payment_rate - o.overall_mg_payment_rate) DESC;
""",
        "hint": "Cities that deviate materially from overall eligibility or MG payment rates.",
    },
]

def detect_canned_mg(question: str) -> Optional[Dict]:
    """Return {'id','sql','hint'} if the question matches a canned card; else None."""
    for card in CARDS:
        for group in card["patterns"]:
            if _matches(question, group):
                return {"id": card["id"], "sql": card["sql"], "hint": card.get("hint","")}
    return None
