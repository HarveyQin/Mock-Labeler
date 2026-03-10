# -*- coding: utf-8 -*-
"""
overwrite_import_sampled_tests.py

IDE runnable, no CLI args.

What it does:
1) Read new sampled dataset CSV (800 cases) which includes:
   - project, file, method, split (learning/eval_seen/eval_unseen)
   - optional: testaware, mockintensity, dependencycount, cctr_bin, batch_id, test_file_path, test_source
2) Derive suite_basename = basename(file) (your CSV file column is already basename)
3) Fill test_source by searching local repo:
   project -> PROJECTS_BASE_DIR / PROJECT_FOLDER_OVERRIDES[project]
   then locate **/src/test/**/<suite_basename> (preferred), then **/test/**, then **/<suite_basename>
4) Upsert into sampled_tests on unique key (project, suite_basename, method)
5) Delete sampled_tests rows that are NOT in new scope (full overwrite)
6) Keep existing case_assignments, but remove assignments for cases dropped from new scope

It does NOT touch annotations_cv.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor


# ===================== CONFIG (EDIT THESE) =====================

DB_URL = "postgresql://neondb_owner:npg_I3eq9uZprPXL@ep-frosty-rice-adydzy86-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # postgresql://... ?sslmode=require
CSV_PATH = Path(r"E:\Files\Mock_Project\ML\ML-experiment\LLM-mock-study-v2\kb\dataset\final_800_CUTknown_eval_split\final_800.csv")

PROJECTS_BASE_DIR = Path(r"E:\Files\Mock_Project\ML\ML-experiment")

PROJECT_FOLDER_OVERRIDES: Dict[str, str] = {
    "cloudstack": "cloudstack-4.20.0.0",
    "commons-configuration": "commons-configuration-rel-commons-configuration-2.11.0",
    "camel": "camel-camel-4.10.8",
    "maven": "maven-maven-3.9.12",
    "crunch": "crunch-apache-crunch-1.0.0",
    "hadoop": "hadoop-rel-release-3.4.2",
    "hbase": "hbase-rel-2.6.4",
    "storm": "storm-2.8.3",
    "flink": "flink-release-2.2.0",
    "hive": "hive-rel-release-4.2.0",
    "dubbo": "dubbo-dubbo-3.3.6",
}

VALID_SPLITS = {"learning", "eval_seen", "eval_unseen"}

# Search patterns (ordered)
TEST_FILE_SEARCH_GLOBS = [
    "**/src/test/**/{name}",
    "**/src/test/**/{name}.java",
    "**/test/**/{name}",
    "**/test/**/{name}.java",
    "**/{name}",
    "**/{name}.java",
]

# ===============================================================


def connect():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn


def safe_read_text(p: Path) -> str:
    try:
        data = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        data = p.read_text(encoding="latin-1", errors="ignore")
    # strip NULs (Neon / psycopg2 will reject)
    return data.replace("\x00", "")


def basename_of_path(s: str) -> str:
    s = (s or "").replace("\\", "/")
    return s.split("/")[-1].strip()


def resolve_project_root(project: str) -> Optional[Path]:
    folder = PROJECT_FOLDER_OVERRIDES.get(project)
    if folder:
        p = (PROJECTS_BASE_DIR / folder)
        if p.exists() and p.is_dir():
            return p.resolve()

    # fallback: try exact match
    p2 = (PROJECTS_BASE_DIR / project)
    if p2.exists() and p2.is_dir():
        return p2.resolve()

    # fallback: prefix match
    cands = sorted(PROJECTS_BASE_DIR.glob(f"{project}-*"))
    if cands:
        return cands[0].resolve()

    return None


def choose_best_path(paths: List[Path]) -> Path:
    """
    Prefer:
      - contains /src/test/ or /test/
      - shorter path
    """
    def score(p: Path) -> Tuple[int, int]:
        s = p.as_posix().lower()
        test_bonus = 0
        if "/src/test/" in s:
            test_bonus -= 2
        elif "/test/" in s or "/tests/" in s:
            test_bonus -= 1
        return (test_bonus, len(s))

    return sorted(paths, key=score)[0]


def find_test_file_in_repo(project: str, suite_basename: str) -> Tuple[Optional[Path], str]:
    """
    Locate the test suite file in the local repo.
    Returns (path, note).
    """
    root = resolve_project_root(project)
    if root is None:
        return None, f"project_root not found (check PROJECT_FOLDER_OVERRIDES) for project={project}"

    name = suite_basename
    if not name.lower().endswith(".java"):
        name_java = name + ".java"
    else:
        name_java = name

    candidates: List[Path] = []
    for pattern in TEST_FILE_SEARCH_GLOBS:
        pat = pattern.format(name=name, name_java=name_java)
        # ensure .java is included
        pat = pat.replace("{name_java}", name_java)
        found = list(root.glob(pat))
        for f in found:
            if f.is_file() and f.name.lower() == name_java.lower():
                candidates.append(f)

        if candidates:
            best = choose_best_path(candidates)
            rel = None
            try:
                rel = best.relative_to(root).as_posix()
            except Exception:
                rel = best.as_posix()
            return best, f"found via glob='{pat}' rel='{rel}'"

    return None, f"not found in repo (searched by basename) project={project} suite={suite_basename}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # required-ish columns
    for col in ["Project", "File", "Method", "Split"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # standardize column names to lower
    df.columns = [c.strip().lower() for c in df.columns]

    # suite_basename
    df["suite_basename"] = df["file"].astype(str).apply(basename_of_path)

    # enforce split
    df["split"] = df["split"].astype(str).str.strip()
    bad = df[~df["split"].isin(VALID_SPLITS)]
    if not bad.empty:
        raise ValueError(f"Invalid split values found: {sorted(set(bad['split'].tolist()))}")

    # optional fields (create if missing)
    for opt in ["testaware", "mockintensity", "dependencycount", "cctr_bin", "batch_id", "test_file_path", "test_source"]:
        if opt not in df.columns:
            df[opt] = None

    # dependencycount numeric
    df["dependencycount"] = pd.to_numeric(df["dependencycount"], errors="coerce").fillna(0).astype(int)

    return df


def dedupe_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedupe by (project, suite_basename, method) keeping the first.
    """
    key_cols = ["project", "suite_basename", "method"]
    dup_mask = df.duplicated(subset=key_cols, keep="first")
    if dup_mask.any():
        dups = df.loc[dup_mask, key_cols]
        print("[WARN] Duplicated cases found in CSV (will keep the first occurrence):")
        print(dups.to_string(index=False))
        print(f"[WARN] Total duplicated rows dropped: {int(dup_mask.sum())}")
    return df.loc[~dup_mask].reset_index(drop=True)


def fetch_existing_case_keys(cur) -> List[Tuple[str, str, str]]:
    cur.execute("select project, suite_basename, method from sampled_tests")
    return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print(f"[INFO] Reading CSV: {CSV_PATH}")
    df_raw = pd.read_csv(CSV_PATH)
    df = normalize_columns(df_raw)
    df = dedupe_cases(df)

    print(f"[INFO] After dedupe, remaining rows = {len(df)}")

    # Fill test_source if empty
    filled = 0
    missed: List[Tuple[str, str, str, str]] = []  # (project, suite_basename, method, reason)

    # If CSV already has test_source, keep it; only fill when empty
    test_sources: List[str] = []
    test_paths: List[Optional[str]] = []

    for i, row in df.iterrows():
        project = str(row["project"]).strip()
        suite_basename = str(row["suite_basename"]).strip()
        method = str(row["method"]).strip()

        existing_src = row.get("test_source")
        if isinstance(existing_src, str) and existing_src.strip():
            src = existing_src.replace("\x00", "")
            test_sources.append(src)
            test_paths.append(str(row.get("test_file_path") or "") or None)
            filled += 1
            continue

        # locate file in repo
        p, note = find_test_file_in_repo(project, suite_basename)
        if p is None:
            test_sources.append("")
            test_paths.append(None)
            missed.append((project, suite_basename, method, note))
            continue

        src = safe_read_text(p)
        test_sources.append(src)
        # store relative path (best effort)
        root = resolve_project_root(project)
        rel = None
        if root:
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = p.as_posix()
        test_paths.append(rel or p.as_posix())
        filled += 1

        if (i + 1) % 50 == 0:
            print(f"[INFO] Filled test_source: {filled}/{i+1}")

    df["test_source"] = test_sources
    # only set test_file_path if empty (otherwise keep CSV)
    df["test_file_path"] = [
        (df.loc[idx, "test_file_path"] if isinstance(df.loc[idx, "test_file_path"], str) and df.loc[idx, "test_file_path"].strip()
         else (tp or None))
        for idx, tp in enumerate(test_paths)
    ]

    print(f"[INFO] CSV rows: {len(df)}, test_source filled: {filled}")
    if missed:
        print(f"[WARN] test_source NOT found for {len(missed)} cases (will store empty test_source). Showing up to 30:")
        for m in missed[:30]:
            print("   ", m)

    # Prepare upsert rows
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["project"]).strip(),
            str(r["file"]).strip(),
            str(r["method"]).strip(),
            str(r["suite_basename"]).strip(),
            str(r["split"]).strip(),
            int(r["batch_id"]) if pd.notna(r["batch_id"]) and str(r["batch_id"]).strip() else None,
            (str(r["testaware"]).strip() if pd.notna(r["testaware"]) else None),
            (str(r["mockintensity"]).strip() if pd.notna(r["mockintensity"]) else None),
            int(r["dependencycount"]) if pd.notna(r["dependencycount"]) else 0,
            (str(r["cctr_bin"]).strip() if pd.notna(r["cctr_bin"]) else None),
            (str(r["test_file_path"]).strip() if pd.notna(r["test_file_path"]) and str(r["test_file_path"]).strip() else None),
            (str(r["test_source"]) if isinstance(r["test_source"], str) else ""),
        ))

    new_keys = {(a, b, c) for (a, _, c, b, *_rest) in [(x[0], x[1], x[2], x[3], None) for x in rows]}  # will override below
    # rebuild correctly
    new_keys = {(x[0], x[3], x[2]) for x in rows}  # (project, suite_basename, method)

    conn = connect()
    try:
        with conn.cursor() as cur:
            # existing keys (for dropped-case cleanup)
            old_keys = set(fetch_existing_case_keys(cur))
            dropped = sorted(old_keys - new_keys)

            # 1) upsert sampled_tests
            execute_values(
                cur,
                """
                insert into sampled_tests(
                    project, file, method, suite_basename, split, batch_id,
                    testaware, mockintensity, dependencycount, cctr_bin,
                    test_file_path, test_source
                )
                values %s
                on conflict (project, suite_basename, method)
                do update set
                    file=excluded.file,
                    method=excluded.method,
                    split=excluded.split,
                    batch_id=excluded.batch_id,
                    testaware=excluded.testaware,
                    mockintensity=excluded.mockintensity,
                    dependencycount=excluded.dependencycount,
                    cctr_bin=excluded.cctr_bin,
                    test_file_path=excluded.test_file_path,
                    test_source=excluded.test_source
                """,
                rows,
                page_size=500,
            )

            # 2) delete sampled_tests rows not in new scope (full overwrite)
            # Using NOT EXISTS on the new keys list via temp table for safety/perf
            cur.execute("drop table if exists _new_scope_keys")
            cur.execute("""
                create temporary table _new_scope_keys(
                    project text not null,
                    suite_basename text not null,
                    method text not null
                ) on commit drop
            """)
            execute_values(
                cur,
                "insert into _new_scope_keys(project, suite_basename, method) values %s",
                [(k[0], k[1], k[2]) for k in new_keys],
                page_size=1000,
            )
            cur.execute("""
                delete from sampled_tests st
                where not exists (
                    select 1 from _new_scope_keys nk
                    where nk.project = st.project
                      and nk.suite_basename = st.suite_basename
                      and nk.method = st.method
                )
            """)

            # 3) clean case_assignments for dropped cases ONLY (keep the rest)
            if dropped:
                execute_values(
                    cur,
                    """
                    delete from case_assignments ca
                    using (values %s) as d(project, suite_basename, method)
                    where ca.project = d.project
                      and ca.test_suite_basename = d.suite_basename
                      and ca.test_case = d.method
                    """,
                    dropped,
                    page_size=1000,
                )
                print(f"[INFO] Cleaned case_assignments for dropped cases. dropped={len(dropped)}")
            else:
                print("[INFO] No dropped cases vs previous scope; case_assignments unchanged.")

        conn.commit()
        print("[INFO] Overwrite import done.")

    except Exception as e:
        conn.rollback()
        print("[ERROR] Import failed, rolled back.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()