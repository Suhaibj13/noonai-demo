# schema_registry.py
# Simple, extensible table -> {col:type} mapping used to craft schema text.
# Keep names EXACTLY as used in the front-end select and in BigQuery table names you expect.

SCHEMAS = {
    "order_items": {
        "id": "INT64",
        "order_id": "INT64",
        "user_id": "INT64",
        "status": "STRING",
        "created_at": "TIMESTAMP",
        "sale_price": "FLOAT64",
        "product_id": "INT64",
        "quantity": "INT64",
    },
    "inventory_items": {
        "id": "INT64",
        "product_id": "INT64",
        "warehouse_id": "INT64",
        "stock_on_hand": "INT64",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
    },
    "orders": {
        "order_id": "INT64",
        "user_id": "INT64",
        "status": "STRING",
        "created_at": "TIMESTAMP",
        "num_items": "INT64",
        "total_amount": "FLOAT64",
    },
    "products": {
        "product_id": "INT64",
        "brand": "STRING",
        "category": "STRING",
        "cost": "FLOAT64",
        "retail_price": "FLOAT64",
        "created_at": "TIMESTAMP",
    },
    "customers": {
        "user_id": "INT64",
        "first_name": "STRING",
        "last_name": "STRING",
        "email": "STRING",
        "created_at": "TIMESTAMP",
        "country": "STRING",
    },
}

def get_table_schema(table_name: str) -> dict:
    return SCHEMAS.get(table_name, {})

def schema_to_text(table: str, schema: dict, prefix: str = "") -> str:
    # Produce lines like: table.col_name: TYPE
    lines = []
    for col, typ in schema.items():
        qualified = f"{prefix}{col}" if prefix else col
        lines.append(f"{table}.{qualified}: {typ}")
    return "\n".join(lines)
