# -*- coding: utf-8 -*-
"""
import_rules_overwrite.py (IDE runnable)

Overwrite-import rules from CSV into Postgres (Neon).

CSV expected columns (case-insensitive, spaces ok):
  - ID
  - Type
  - High Level Category
  - Criterion
  - Mock?    (or mock_decision)

DB table: rules
We will:
  1) TRUNCATE rules RESTART IDENTITY
  2) Insert rows (with explicit ID from CSV if DB has id column)

This is the simplest "replace everything" approach.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor


# ===================== CONFIG (EDIT THESE) =====================

DB_URL = "postgresql://neondb_owner:npg_I3eq9uZprPXL@ep-frosty-rice-adydzy86-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # postgresql://... ?sslmode=require

RULE_CSV = Path(r"E:\Files\Mock_Project\ML\ML-experiment\LLM-mock-study-v2\kb\dataset\mock_rules.csv")

# 填充 created_by（如果表里有该列）
CREATED_BY = "seed"

DRY_RUN = False  # True = 只打印，不写库

# ===============================================================


def get_conn():
    if not DB_URL or "postgresql://" not in DB_URL:
        raise RuntimeError("DB_URL is missing/invalid. Please set DB_URL to a proper postgresql:// URL.")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn


def db_fetchall(conn, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def normalize_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    m = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        k = normalize_colname(cand)
        if k in m:
            return m[k]
    return None


def main():
    if not RULE_CSV.exists():
        raise FileNotFoundError(f"Rule CSV not found: {RULE_CSV}")

    df = pd.read_csv(RULE_CSV)

    c_id = pick_col(df, "ID", "id")
    c_type = pick_col(df, "Type", "type")
    c_cat = pick_col(df, "High Level Category", "high_level_category", "category")
    c_crit = pick_col(df, "Criterion", "criterion")
    c_mock = pick_col(df, "Mock?", "mock?", "Mock", "mock_decision", "MockDecision")

    missing = [x for x, c in [("ID", c_id), ("Type", c_type), ("High Level Category", c_cat), ("Criterion", c_crit), ("Mock?", c_mock)] if c is None]
    if missing:
        raise RuntimeError(f"RULE_CSV missing columns: {missing}. Got: {list(df.columns)}")

    # Clean
    df[c_type] = df[c_type].astype(str).str.strip()
    df[c_cat] = df[c_cat].astype(str).str.strip()
    df[c_crit] = df[c_crit].astype(str).str.strip()
    df[c_mock] = df[c_mock].astype(str).str.strip()

    # id
    df[c_id] = pd.to_numeric(df[c_id], errors="coerce").astype("Int64")
    df = df[df[c_id].notna()].copy()
    df[c_id] = df[c_id].astype(int)

    print(f"[INFO] Loaded {len(df)} rules from CSV.")

    if DRY_RUN:
        print(df.head(10).to_string(index=False))
        print("[DRY_RUN] Not writing to DB.")
        return

    conn = get_conn()
    try:
        # detect actual columns in rules table
        cols = db_fetchall(
            conn,
            """
            select column_name
            from information_schema.columns
            where table_schema='public' and table_name='rules'
            order by ordinal_position
            """
        )
        rule_cols = [r["column_name"] for r in cols]
        rule_cols_set = set(rule_cols)

        # Decide insert columns
        use_created_by = "created_by" in rule_cols_set
        use_created_at = "created_at" in rule_cols_set  # usually default now(), so we can omit

        # We will truncate and insert with explicit id (keeps stable IDs)
        with conn.cursor() as cur:
            cur.execute("truncate table rules restart identity;")

            if use_created_by:
                rows = [
                    (int(r[c_id]), r[c_type], r[c_cat], r[c_crit], r[c_mock], CREATED_BY)
                    for _, r in df.iterrows()
                ]
                execute_values(
                    cur,
                    """
                    insert into rules(id, type, high_level_category, criterion, mock_decision, created_by)
                    values %s
                    """,
                    rows,
                    page_size=1000
                )
            else:
                rows = [
                    (int(r[c_id]), r[c_type], r[c_cat], r[c_crit], r[c_mock])
                    for _, r in df.iterrows()
                ]
                execute_values(
                    cur,
                    """
                    insert into rules(id, type, high_level_category, criterion, mock_decision)
                    values %s
                    """,
                    rows,
                    page_size=1000
                )

        conn.commit()
        print("[INFO] Rules imported (overwrite).")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()