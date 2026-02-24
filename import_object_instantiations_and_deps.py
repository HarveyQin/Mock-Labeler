# -*- coding: utf-8 -*-
"""
import_object_instantiations_and_deps.py (fixed UUID adaptation)

IDE runnable, no CLI args.

What it does:
1) For each project in PROJECTS:
   - Read object instantiations from XLSX: <project>_object_instantiations.xlsx
   - Filter to ONLY sampled_tests scope cases for that project
   - Compute occurrence_in_case for repeated class_name within same case
   - Upsert into object_instantiations (inst_id is stable uuid5 derived from key)
2) Optionally import test_dependencies from <project_root>/test_dependencies_map.csv
   - Only keep rows whose test file basename is in sampled scope for that project
   - Store dep paths as text (no source_files required)
3) Cleanup: delete object_instantiations out of sampled scope (optional)

Does NOT touch annotations_cv.
"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# ===================== CONFIG (EDIT THESE) =====================

DB_URL = "postgresql://neondb_owner:npg_I3eq9uZprPXL@ep-frosty-rice-adydzy86-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # postgresql://... ?sslmode=require

PROJECTS_BASE_DIR = Path(r"E:\Files\Mock_Project\ML\ML-experiment")

PROJECTS = ["camel", "hadoop", "flink", "hbase", "cloudstack", "hive", "commons-configuration", "dubbo", "storm", "crunch", "maven"]

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

OBJECT_LIST_DIR = Path(
    r"E:\Files\Mock_Project\ML\Meets\Mock Decision Dataset\Pilot Review\Object Instantiations Lists"
)

IMPORT_TEST_DEPENDENCIES = True
CLEAN_OUT_OF_SCOPE_ROWS = True

# ===============================================================

# Stable namespace for uuid5
NAMESPACE_UUID = uuid.UUID("b7a7c1c0-0d57-4b4c-9b16-8b3d1d19c5e3")


def connect():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn


def basename_of_path(s: str) -> str:
    s = (s or "").replace("\\", "/")
    return s.split("/")[-1].strip()


def resolve_project_root(project: str) -> Optional[Path]:
    folder = PROJECT_FOLDER_OVERRIDES.get(project)
    if folder:
        p = (PROJECTS_BASE_DIR / folder)
        if p.exists() and p.is_dir():
            return p.resolve()

    p2 = (PROJECTS_BASE_DIR / project)
    if p2.exists() and p2.is_dir():
        return p2.resolve()

    cands = sorted(PROJECTS_BASE_DIR.glob(f"{project}-*"))
    if cands:
        return cands[0].resolve()

    return None


def fetch_scope_cases_for_project(cur, project: str) -> Set[Tuple[str, str]]:
    """
    Return set of (suite_basename, method) for sampled_tests of a given project.
    """
    cur.execute(
        """
        select suite_basename, method
        from sampled_tests
        where project=%s
        """,
        (project,),
    )
    return {(r[0], r[1]) for r in cur.fetchall()}


def load_instantiations_xlsx(project: str) -> pd.DataFrame:
    xlsx_path = OBJECT_LIST_DIR / f"{project}_object_instantiations.xlsx"
    if not xlsx_path.exists():
        print(f"[WARN] XLSX not found for {project}: {xlsx_path}")
        return pd.DataFrame(columns=["Test Suite", "Test Case", "Class Name", "Mocked"])

    df = pd.read_excel(xlsx_path)

    for c in ["Test Suite", "Test Case", "Class Name"]:
        if c not in df.columns:
            raise ValueError(f"{xlsx_path} missing column: {c}")

    if "Mocked" not in df.columns:
        df["Mocked"] = None

    df["Test Suite"] = df["Test Suite"].astype(str).str.strip()
    df["Test Case"] = df["Test Case"].astype(str).str.strip()
    df["Class Name"] = df["Class Name"].astype(str).str.strip()
    df["Mocked"] = df["Mocked"].astype(str).str.strip()

    df["TestSuiteBaseName"] = df["Test Suite"].apply(basename_of_path)
    df["source_row_index"] = list(range(len(df)))  # stable ordering
    return df


def make_inst_id_str(project: str, suite_basename: str, test_case: str, class_name: str, occurrence: int) -> str:
    key = f"{project}|{suite_basename}|{test_case}|{class_name}|{occurrence}"
    return str(uuid.uuid5(NAMESPACE_UUID, key))


def upsert_object_instantiations(cur, rows: List[Tuple[Any, ...]]) -> None:
    """
    rows tuple:
      (inst_id_text, project, test_suite, suite_basename, test_case, class_name, occurrence, mocked, source_row_index)
    created_at is set to now() in SQL.
    """
    if not rows:
        return

    execute_values(
        cur,
        """
        insert into object_instantiations(
          inst_id, project, test_suite, test_suite_basename, test_case,
          class_name, occurrence_in_case, mocked, source_row_index, created_at
        )
        values %s
        on conflict (inst_id)
        do update set
          project=excluded.project,
          test_suite=excluded.test_suite,
          test_suite_basename=excluded.test_suite_basename,
          test_case=excluded.test_case,
          class_name=excluded.class_name,
          occurrence_in_case=excluded.occurrence_in_case,
          mocked=excluded.mocked,
          source_row_index=excluded.source_row_index
        """,
        # IMPORTANT: cast inst_id to uuid by passing "inst_id::uuid" in template
        rows,
        template="(%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,now())",
        page_size=1000,
    )


def import_test_dependencies(cur, project: str, scope_suite_basenames: Set[str]) -> int:
    root = resolve_project_root(project)
    if root is None:
        print(f"[WARN] project_root not found for {project}; skipping dependencies")
        return 0

    dep_csv = root / "test_dependencies_map.csv"
    if not dep_csv.exists():
        print(f"[WARN] test_dependencies_map.csv not found for {project}: {dep_csv}")
        return 0

    out_rows: List[Tuple[str, str, str]] = []
    with dep_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if "Test File Path" not in reader.fieldnames or "Dependency Files" not in reader.fieldnames:
            print(f"[WARN] bad dependency CSV columns for {project}: {reader.fieldnames}")
            return 0

        for row in reader:
            tf_raw = (row.get("Test File Path") or "").strip()
            if not tf_raw:
                continue
            tf_base = basename_of_path(tf_raw)
            if tf_base not in scope_suite_basenames:
                continue

            tf_path = Path(tf_raw)
            if not tf_path.is_absolute():
                tf_abs = (root / tf_path).resolve()
            else:
                tf_abs = tf_path

            try:
                tf_rel = tf_abs.relative_to(root).as_posix()
            except Exception:
                tf_rel = tf_abs.as_posix()

            deps = (row.get("Dependency Files") or "").strip()
            if not deps:
                continue

            for d in deps.split(";"):
                d = d.strip()
                if not d:
                    continue
                dp = Path(d)
                if not dp.is_absolute():
                    dp_abs = (root / dp).resolve()
                else:
                    dp_abs = dp

                try:
                    dp_rel = dp_abs.relative_to(root).as_posix()
                except Exception:
                    dp_rel = dp_abs.as_posix()

                out_rows.append((project, tf_rel, dp_rel))

    if not out_rows:
        return 0

    # Overwrite per-project to avoid duplicates
    cur.execute("delete from test_dependencies where project=%s", (project,))
    execute_values(
        cur,
        """
        insert into test_dependencies(project, test_path, dep_path)
        values %s
        """,
        out_rows,
        page_size=5000,
    )
    return len(out_rows)


def main():
    conn = connect()
    try:
        with conn.cursor() as cur:
            total_inst = 0
            total_dep = 0

            for project in PROJECTS:
                print(f"\n[INFO] ==== PROJECT: {project} ====")
                scope = fetch_scope_cases_for_project(cur, project)
                if not scope:
                    print(f"[WARN] No sampled_tests scope cases for project={project}. Skipping.")
                    continue

                scope_basenames = {sb for (sb, _m) in scope}

                df = load_instantiations_xlsx(project)
                if df.empty:
                    print(f"[WARN] Empty or missing XLSX for {project}.")
                else:
                    df2 = df[df.apply(lambda r: (r["TestSuiteBaseName"], r["Test Case"]) in scope, axis=1)].copy()
                    print(f"[INFO] XLSX rows: {len(df)}, in-scope rows: {len(df2)}")

                    if not df2.empty:
                        df2.sort_values(["TestSuiteBaseName", "Test Case", "Class Name", "source_row_index"], inplace=True)
                        df2["occurrence_in_case"] = (
                            df2.groupby(["TestSuiteBaseName", "Test Case", "Class Name"]).cumcount() + 1
                        )

                        rows: List[Tuple[Any, ...]] = []
                        for _, r in df2.iterrows():
                            suite_basename = str(r["TestSuiteBaseName"])
                            test_case = str(r["Test Case"])
                            class_name = str(r["Class Name"])
                            occ = int(r["occurrence_in_case"])
                            inst_id_txt = make_inst_id_str(project, suite_basename, test_case, class_name, occ)

                            rows.append((
                                inst_id_txt,
                                project,
                                str(r["Test Suite"]),
                                suite_basename,
                                test_case,
                                class_name,
                                occ,
                                (str(r["Mocked"]) if pd.notna(r["Mocked"]) else None),
                                int(r["source_row_index"]),
                            ))

                        upsert_object_instantiations(cur, rows)
                        total_inst += len(rows)
                        print(f"[INFO] Upserted object_instantiations for {project}: {len(rows)}")
                    else:
                        print(f"[INFO] No in-scope instantiations for {project}.")

                if IMPORT_TEST_DEPENDENCIES:
                    dep_n = import_test_dependencies(cur, project, scope_basenames)
                    total_dep += dep_n
                    print(f"[INFO] Imported test_dependencies for {project}: {dep_n}")

            if CLEAN_OUT_OF_SCOPE_ROWS:
                cur.execute("""
                    delete from object_instantiations oi
                    where not exists (
                      select 1 from sampled_tests st
                      where st.project = oi.project
                        and st.suite_basename = oi.test_suite_basename
                        and st.method = oi.test_case
                    )
                """)

            conn.commit()
            print("\n[INFO] DONE.")
            print(f"[INFO] Total object_instantiations upserted: {total_inst}")
            print(f"[INFO] Total test_dependencies imported: {total_dep}")

    except Exception:
        conn.rollback()
        print("[ERROR] Import failed; rolled back.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()