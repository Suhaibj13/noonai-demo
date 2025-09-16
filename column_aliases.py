# column_aliases.py
# Curated alternate names (aliases) for each dataset's columns.
# These are READ-ONLY lists. They don't change any behavior by themselves;
# main.py imports them and uses them to build the runtime alias map.

from typing import Dict, List

COLUMN_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "mg": {
        "id_user": ["user", "user_id", "userid", "da", "da_id", "rider_id", "courier_id"],
        "Name": ["da_name", "agent_name", "rider_name", "courier_name", "name"],
        "Vendor Name": ["vendor", "vendor_name", "fleet_company", "partner", "company"],
        "Rate": ["rate", "mg_rate"],
        "Joining Date": ["join_date", "joining_date", "start_date"],
        "Total Calendar Days": ["calendar_days", "total_calendar_days"],
        "Attendance": ["attendance_days", "days_attended", "attendance"],
        "Total Attendance": ["attendance_total", "total_attendance"],
        "Perfect Attendance": ["perfect_attendance", "pa"],
        "PA Needed": ["pa_needed", "attendance_needed"],
        "MG Eligible": ["eligible", "is_eligible", "mg_eligible", "eligible_mg"],
        "Total Delivered Month": ["delivered_days", "total_delivered_month"],
        "Monthly MG": ["monthly_mg", "mg_monthly"],
        "Eligible MG": ["eligible_mg"],
        "MG Month": ["mg_month"],
        "Payout": ["payout", "paid_amount"],
        "MG Amount": ["mg_amount", "amount_mg", "mg_payout"],
        "Total Payout": ["total_payout", "payout_total", "total_payment", "totalpayment"],
        "MOT": ["mot"],
        "FInal MG check": ["final_mg_check"],
        "FND+NDR+penalty Amount": ["fnd_ndr_penalty_amount", "penalty_amount"],
        "DA Level Final Amount": ["final_amount", "da_final_amount", "final_payout"],
        "City": ["city", "location_city"],
        "fleet": ["fleet", "vendor"],
    },
    "orders": {
        "id": ["row_id", "pk"],
        "order_id": ["order", "ord_id", "order_number"],
        "user_id": ["customer_id", "user", "uid"],
        "product_id": ["sku", "item_id", "product"],
        "inventory_item_id": ["inventory_id", "stock_item_id"],
        "status": ["order_status", "state"],
        "created_at": ["order_date", "created", "date"],
        "shipped_at": ["shipped", "shipping_date"],
        "delivered_at": ["delivered", "delivery_date"],
        "returned_at": ["returned", "return_date"],
        "sale_price": ["price", "amount", "order_amount"],
    },
    "inventory": {
        "id": ["row_id", "pk"],
        "product_id": ["sku", "item_id"],
        "created_at": ["received_at", "inbound_date", "created", "date"],
        "sold_at": ["sold_date", "sale_date"],
        "cost": ["unit_cost", "cogs"],
        "product_category": ["category"],
        "product_name": ["name"],
        "product_brand": ["brand"],
        "product_retail_price": ["retail_price", "price"],
        "product_department": ["department", "dept"],
        "product_sku": ["sku"],
        "product_distribution_center_id": ["dc_id", "warehouse_id", "distribution_center_id"],
    },
}
