# demo_intents.py
from __future__ import annotations
from typing import Any, Dict

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

DEMO: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ---------------- DATA MODE (table preview only; NO SQL) ----------------
    "data": {
        _norm("Give me total purchase for each customer"): {
            "reply": "Executed SQL. Showing first 50 rows.",
            "sql": "",
            "preview": """customer_id, total_purchase
30048.0, 1,809.94
13219.0, 1,782.99
95674.0, 1,744.40
11361.0, 1,448.44
98530.0, 1,412.45
34572.0, 1,411.18
54985.0, 1,385.66
73335.0, 1,384.49
32056.0, 1,370.77
85693.0, 1,367.70
52491.0, 1,355.78
4852.0, 1,343.99
12793.0, 1,334.43
24843.0, 1,332.99
70463.0, 1,303.72
996.0, 1,300.94
29231.0, 1,298.89
13324.0, 1,293.85
72159.0, 1,285.90
87816.0, 1,275.94
61075.0, 1,270.23
80358.0, 1,259.88
99658.0, 1,258.32
68077.0, 1,257.95
17424.0, 1,256.61
37259.0, 1,250.75
70557.0, 1,247.96
39984.0, 1,246.53
87283.0, 1,245.57
88192.0, 1,245.20
15319.0, 1,240.80
22351.0, 1,240.23
32861.0, 1,237.69
32048.0, 1,235.39
47186.0, 1,232.44
46329.0, 1,229.66
64286.0, 1,226.77
96310.0, 1,225.19
72177.0, 1,224.18
87692.0, 1,222.76
40410.0, 1,219.30
78925.0, 1,219.17
81334.0, 1,219.02
81832.0, 1,218.41
75384.0, 1,217.61
39444.0, 1,214.31
60703.0, 1,214.28
27403.0, 1,197.98
54671.0, 1,192.93
40926.0, 1,191.89
26672.0, 1,191.34
27232.0, 1,173.67
16613.0, 1,172.95
96911.0, 1,170.95
30480.0, 1,170.43
79239.0, 1,170.00
73020.0, 1,169.90
66745.0, 1,168.99
69710.0, 1,159.11
97402.0, 1,158.75
57700.0, 1,158.14"""
        },

        _norm("Show total orders by day"): {
            "reply": "Executed SQL. Showing first 50 rows.",
            "sql": "",
            "preview": """date, total_orders, revenue
2019-01-09, 1.00, 83.76
2019-01-10, 1.00, 55.00
2019-01-13, 2.00, 437.48
2019-01-16, 2.00, 137.20
2019-01-20, 2.00, 121.55
2019-01-22, 1.00, 61.77
2019-01-24, 2.00, 180.26
2019-01-25, 2.00, 153.24
2019-01-27, 2.00, 182.50
2019-01-28, 1.00, 25.69
2019-01-29, 2.00, 148.96
2019-01-30, 1.00, 68.65
2019-01-31, 3.00, 173.56
2019-02-02, 2.00, 86.38
2019-02-03, 2.00, 173.46
2019-02-04, 1.00, 61.80
2019-02-05, 2.00, 300.64
2019-02-06, 2.00, 86.59
2019-02-07, 2.00, 132.51
2019-02-08, 2.00, 86.86
2019-02-09, 1.00, 93.37
2019-02-10, 2.00, 103.35
2019-02-11, 2.00, 180.75
2019-02-12, 1.00, 49.99
2019-02-13, 1.00, 59.73
2019-02-14, 2.00, 119.75
2019-02-15, 1.00, 68.15
2019-02-16, 2.00, 160.65
2019-02-17, 1.00, 68.17
2019-02-18, 2.00, 90.96
2019-02-19, 1.00, 49.99
2019-02-20, 2.00, 180.35
2019-02-21, 2.00, 79.98
2019-02-22, 2.00, 168.64
2019-02-23, 1.00, 49.99
2019-02-24, 2.00, 210.45
2019-02-25, 2.00, 148.16
2019-02-26, 2.00, 185.09
2019-02-27, 1.00, 49.99
2019-02-28, 1.00, 68.50
2019-03-01, 3.00, 126.95
2019-03-02, 2.00, 103.34
2019-03-03, 2.00, 119.48
2019-03-04, 2.00, 147.13
2019-03-05, 2.00, 99.33
2019-03-06, 2.00, 164.48
2019-03-07, 2.00, 104.03
2019-03-08, 1.00, 49.99
2019-03-09, 2.00, 119.68
2019-03-10, 1.00, 49.99"""
        },

        _norm("Top 5 sku by revenue"): {
            "reply": "Executed SQL. Showing first 5 rows.",
            "sql": "",
            "preview": """sku, revenue
23546.0, 11,988.00
24447.0, 10,989.00
24314.0, 9,750.00
23803.0, 8,150.00
23989.0, 8,127.00"""
        },

        _norm("Compare orders first half vs second half of month"): {
            "reply": "Executed SQL. Showing first 2 rows.",
            "sql": "",
            "preview": """period, total_orders, revenue
First half, 65,440.00, 5,444,105.42
Second half, 64,000.00, 5,331,184.89"""
        },

        _norm("Show minimum guarantee payout by vendor"): {
            "reply": "Executed SQL. Showing first 10 rows.",
            "sql": "",
            "preview": """vendor, mg_amount, payout, total_payout
Desert Riders, 7,187,855.00, 5,796,462.00, 12,984,317.00
Swift Hands, 7,085,978.00, 5,804,119.00, 12,890,097.00
Rapid Wheels, 7,068,874.00, 5,828,847.00, 12,897,721.00
First Company, 7,046,980.00, 5,793,164.00, 12,840,144.00
Falcon Express, 6,886,804.00, 5,566,694.00, 12,453,498.00
Tiger Delivery, 6,826,018.00, 5,753,096.00, 12,579,114.00
Alpha Logistics, 6,677,602.00, 5,792,685.00, 12,470,287.00
Station Express, 6,665,148.00, 5,672,368.00, 12,337,516.00
Camel Couriers, 6,528,785.00, 5,629,083.00, 12,157,868.00
Green Wheels, 6,441,440.00, 5,529,171.00, 11,970,611.00"""
        },
    },

    # ---------------- ANALYSIS MODE (numbers + short insights) ----------------
    "analysis": {
        _norm("What is total revenue and average order value?"): {
            "reply": """Answer: total revenue = 10,775,290.30, AOV = 86.35 (over 124,792 orders).
- Customer concentration: top 10% customers drive 33.78% of revenue; top 1% drive 6.38%.
- Loyalty effect: ~59.95% of revenue comes from returning customers."""
        },

        _norm("First vs second half of month — orders and revenue"): {
            "reply": """Answer: Orders — First half = 65,440, Second half = 64,000 (-2.20% change). Revenue — First half = 5,444,105.42, Second half = 5,331,184.89 (-2.07%). AOV — First half = 83.19, Second half = 83.30 (+0.13%).
- Insight: demand softens slightly in the second half, while value per order stays broadly stable."""
        },
        _norm("First vs second half of month - orders and revenue"): {
            "reply": """Answer: Orders — First half = 65,440, Second half = 64,000 (-2.20% change). Revenue — First half = 5,444,105.42, Second half = 5,331,184.89 (-2.07%). AOV — First half = 83.19, Second half = 83.30 (+0.13%).
- Insight: demand softens slightly in the second half, while value per order stays broadly stable."""
        },

        _norm("Top 5 SKUs by revenue and % share"): {
            "reply": """Answer: top-5 SKUs contribute 0.45% of revenue; top-20 = 1.42% across 29,044 SKUs.
- Top SKU 23546 revenue = 11,988.00. Revenue is highly long-tailed (no heavy concentration at SKU level)."""
        },

        _norm("Minimum guarantee utilization summary"): {
            "reply": """Answer: MG utilization = 83.56% (committed 68,415,484 vs paid 57,165,689). Median vendor utilization = 83.37%.
- Lowest utilization vendors:
  - Desert Riders: 80.64%
  - Falcon Express: 80.83%
  - Swift Hands: 81.91%
- Highest (still <90%):
  - Alpha Logistics: 86.75%
  - Camel Couriers: 86.22%
  - Green Wheels: 85.84%
- Distribution: 6 vendors <85% util, 4 vendors 85–90%, 0 vendors ≥90%."""
        },

        _norm("Inventory health summary"): {
            "reply": """Answer: 29,044 unique SKUs across 487,669 inventory items.
- Sell-through: 37.01% sold (180,505 items), 307,164 still unsold.
- Pricing: avg cost 28.72, avg retail 59.72, avg unit margin 31.00 (avg gross margin ≈ 51.06%).
- Brand mix: top 5 brands (Allegra K, Calvin Klein, Carhartt, Hanes, Volcom) represent 8.81% of inventory; total brands 2,754."""
        },
    },
}

def get_demo_response(mode: str, user_input: str) -> Dict[str, Any] | None:
    return DEMO.get((mode or "").lower(), {}).get(_norm(user_input))
