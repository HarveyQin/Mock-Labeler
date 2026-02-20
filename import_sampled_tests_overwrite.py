# -*- coding: utf-8 -*-
"""
import_sampled_tests_overwrite.py (IDE runnable)

Overwrite sampled_tests from a CSV (new scope), using CSV's split column:
  split in {learning, eval_seen, eval_unseen}

Also:
- fills suite_basename if missing
- de-duplicates CSV by (project, suite_basename, method) to satisfy uq_sampled_case
- removes case_assignments rows that point to cases not in the new scope

NOTE:
This script does NOT truncate other tables. If you want "reset everything",
run TRUNCATE SQL in Neon console first (I'll give you a snippet below).
"""

from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# ===================== CONFIG (EDIT THESE) =====================

DB_URL = "postgresql://neondb_owner:npg_I3eq9uZprPXL@ep-frosty-rice-adydzy86-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # postgresql://... ?sslmode=require

CSV_PATH = Path(r"E:\Files\Mock_Project\ML\ML-experiment\LLM-mock-study-v2\kb\dataset\final_800_CUTknown_eval_split\final_800.csv")

BATCH_SIZE = 50        # 如果 CSV 没 batch_id，就每 50 个 case 自动生成 batch_id
DRY_RUN = False        # True = 只预览不写库

# ===============================================================


def basename_of_path(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\\", "/")
    return s.split("/")[-1]


def coerce_int(x):
    try:
        if pd.isna(x) or x == "":
            return None
        return int(x)
    except Exception:
        return None


def normalize_text(x):
    if x is None:
        return None
    if pd.isna(x):
        return None
    s = str(x)
    return s.strip() if s.strip() != "" else None


def get_conn():
    if not DB_URL or "postgresql://" not in DB_URL:
        raise RuntimeError("DB_URL is missing/invalid. Please set DB_URL to a proper postgresql:// URL.")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Accept both old names (Project/File/Method) and lowercase
    colmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
        return None

    c_project = pick("project", "Project")
    c_file = pick("file", "File")
    c_method = pick("method", "Method")
    if not (c_project and c_file and c_method):
        raise RuntimeError(f"CSV must include Project/File/Method. Got columns: {list(df.columns)}")

    # New CSV MUST have split
    c_split = pick("split", "Split")
    if not c_split:
        raise RuntimeError("CSV must include split column with values: learning / eval_seen / eval_unseen")

    c_testaware = pick("testaware", "TestAware")
    c_mockintensity = pick("mockintensity", "MockIntensity")
    c_dependencycount = pick("dependencycount", "DependencyCount")
    c_cctr = pick("cctr_bin", "CCTR_Bin", "CCTR_BIN")
    c_test_file_path = pick("test_file_path", "TestFileAbsPath", "TestFilePath")
    c_test_source = pick("test_source", "TestSource")
    c_suite_basename = pick("suite_basename", "SuiteBaseName")
    c_batch_id = pick("batch_id", "BatchId", "batch")

    # ----- dedupe by uq_sampled_case key: (project, suite_basename, method) -----
    df["_suite_basename"] = df[c_file].apply(lambda x: basename_of_path(x))
    df["_project"] = df[c_project].astype(str).str.strip()
    df["_method"] = df[c_method].astype(str).str.strip()

    dup_mask = df.duplicated(subset=["_project", "_suite_basename", "_method"], keep="first")
    dup_df = df.loc[dup_mask, ["_project", "_suite_basename", "_method"]].copy()

    if not dup_df.empty:
        print("[WARN] Duplicated cases found in CSV (will keep the first occurrence):")
        print(dup_df.head(50).to_string(index=False))
        print(f"[WARN] Total duplicated rows dropped: {len(dup_df)}")

    df = df.drop_duplicates(subset=["_project", "_suite_basename", "_method"], keep="first").reset_index(drop=True)
    print(f"[INFO] After dedupe, remaining rows = {len(df)}")

    # ----- build insert rows -----
    rows = []
    for i, r in df.iterrows():
        project = normalize_text(r[c_project])
        file_ = normalize_text(r[c_file])
        method = normalize_text(r[c_method])
        if not project or not file_ or not method:
            continue

        suite_base = normalize_text(r[c_suite_basename]) if c_suite_basename else None
        if not suite_base:
            suite_base = r["_suite_basename"]

        split = normalize_text(r[c_split])
        if split not in ("learning", "eval_seen", "eval_unseen"):
            raise RuntimeError(f"Bad split value at row {i}: {split}")

        batch_id = coerce_int(r[c_batch_id]) if c_batch_id else None
        if batch_id is None:
            batch_id = (i // BATCH_SIZE) + 1

        dep_count = coerce_int(r[c_dependencycount]) if c_dependencycount else None

        rows.append((
            project,
            file_,
            method,
            normalize_text(r[c_testaware]) if c_testaware else None,
            normalize_text(r[c_mockintensity]) if c_mockintensity else None,
            dep_count,
            normalize_text(r[c_cctr]) if c_cctr else None,
            normalize_text(r[c_test_file_path]) if c_test_file_path else None,
            normalize_text(r[c_test_source]) if c_test_source else None,
            suite_base,
            split,
            batch_id
        ))

    print(f"[INFO] Insertable rows: {len(rows)}")
    if DRY_RUN:
        print("[DRY_RUN] Not writing to DB.")
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Overwrite sampled_tests
            cur.execute("truncate table sampled_tests restart identity;")

            execute_values(
                cur,
                """
                insert into sampled_tests(
                  project, file, method,
                  testaware, mockintensity, dependencycount, cctr_bin,
                  test_file_path, test_source,
                  suite_basename, split, batch_id
                ) values %s
                """,
                rows,
                page_size=1000
            )

            # Defensive fill suite_basename if any blank
            cur.execute(
                r"""
                update sampled_tests
                set suite_basename = regexp_replace(replace(file, '\\', '/'), '^.*/', '')
                where suite_basename is null or suite_basename = '';
                """
            )

            # Remove case_assignments that are no longer in scope
            cur.execute(
                """
                delete from case_assignments ca
                where not exists (
                  select 1
                  from sampled_tests st
                  where st.project = ca.project
                    and st.suite_basename = ca.test_suite_basename
                    and st.method = ca.test_case
                );
                """
            )

        conn.commit()
        print("[INFO] Overwrite import done (sampled_tests).")
        print("[INFO] Cleaned case_assignments for dropped cases (if any).")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()