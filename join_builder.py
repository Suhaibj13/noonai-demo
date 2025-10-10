# join_builder.py
from typing import List, Tuple
from schema_registry import get_table_schema, schema_to_text

def build_from_and_schema(main_table: str, joins: List[dict]) -> Tuple[str, str]:
    """
    Returns (from_clause, joined_schema_text)
    - from_clause: e.g., "FROM order_items LEFT JOIN inventory_items USING (product_id)"
    - joined_schema_text: textual schema with table-prefixed lines. Joined cols are kept under their table name.
    """
    if not main_table:
        return "", ""

    # FROM + JOINS
    clauses = [f"FROM {main_table}"]
    main_schema = get_table_schema(main_table)
    all_schema_text = [schema_to_text(main_table, main_schema)]

    for j in (joins or []):
        jtype = (j.get("type") or "INNER").upper()
        jtable = j.get("table")
        jkey = j.get("key")
        if not jtable or not jkey:
            # skip malformed entries
            continue
        clauses.append(f"{jtype} JOIN {jtable} USING ({jkey})")
        j_schema = get_table_schema(jtable)
        # prefix columns for clarity in prompt (we keep textual prefix via table name)
        all_schema_text.append(schema_to_text(jtable, j_schema))

    from_clause = " ".join(clauses)
    joined_schema_text = "\n".join(s for s in all_schema_text if s)
    return from_clause, joined_schema_text
