# -*- coding: utf-8 -*-
"""
import_projects_incremental.py (IDE runnable)

Imports per-project supporting tables into Neon:
- test_dependencies (from <project_root>/test_dependencies_map.csv)
- source_files (read content for test+deps files)
- object_instantiations (from <project>_object_instantiations.xlsx)

Re-runnable safely (UPSERT / ON CONFLICT).
Cleans NUL bytes in file content.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# ===================== CONFIG (EDIT THESE) =====================

DB_URL = "postgresql://neondb_owner:npg_I3eq9uZprPXL@ep-frosty-rice-adydzy86-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # postgresql://... ?sslmode=require

PROJECTS: List[str] = ["camel", "hadoop", "flink", "hbase", "cloudstack", "hive",
                       "commons-configuration", "dubbo", "storm", "crunch", "maven"]

PROJECTS_BASE_DIR = Path(r"E:\Files\Mock_Project\ML\ML-experiment")
OBJECT_LIST_DIR = Path(r"E:\Files\Mock_Project\ML\Meets\Mock Decision Dataset\Pilot Review\Object Instantiations Lists")

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

INST_NAMESPACE = uuid.UUID("3b4f2fb7-3c34-4d3a-8c62-7c4cb1c40a3d")

DRY_RUN = False

# ===============================================================


def norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


def basename_of_path(s: str) -> str:
    if s is None:
        return ""
    return norm_path(s).split("/")[-1]


def read_text_safely(path: Path) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        data = path.read_text(encoding="latin-1", errors="ignore")
    return data.replace("\x00", "")


def resolve_project_root(project: str, projects_base: Path) -> Optional[Path]:
    if project in PROJECT_FOLDER_OVERRIDES:
        cand = projects_base / PROJECT_FOLDER_OVERRIDES[project]
        return cand if cand.exists() else None

    exact = projects_base / project
    if exact.exists():
        return exact

    cands = sorted(projects_base.glob(f"{project}-*"))
    return cands[0] if cands else None


def get_conn():
    if not DB_URL or "postgresql://" not in DB_URL:
        raise RuntimeError("DB_URL is missing/invalid. Please set DB_URL to a proper postgresql:// URL.")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn


def ensure_unique_indexes(conn):
    stmts = [
        "create unique index if not exists test_dependencies_uniq on test_dependencies(project, test_path, dep_path);",
        "create unique index if not exists source_files_uniq on source_files(project, path);",
    ]
    with conn.cursor() as cur:
        for s in stmts:
            cur.execute(s)


def upsert_test_dependencies(cur, deps_rows: List[Tuple[str, str, str]]):
    if not deps_rows:
        return
    execute_values(
        cur,
        """
        insert into test_dependencies(project, test_path, dep_path)
        values %s
        on conflict (project, test_path, dep_path) do nothing
        """,
        deps_rows,
        page_size=5000
    )


def upsert_source_files(cur, file_rows: List[Tuple[str, str, str]]):
    if not file_rows:
        return
    execute_values(
        cur,
        """
        insert into source_files(project, path, content)
        values %s
        on conflict (project, path) do update
          set content = excluded.content
        """,
        file_rows,
        page_size=1000
    )


def upsert_object_instantiations(cur, rows: List[Tuple[str, str, str, str, str, str, int, str, int]]):
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
        on conflict (inst_id) do update
          set mocked = excluded.mocked,
              source_row_index = excluded.source_row_index
        """,
        rows,
        template="(%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,now())",
        page_size=2000
    )


def load_dependency_map(project_root: Path) -> List[Tuple[str, str]]:
    dep_csv = project_root / "test_dependencies_map.csv"
    if not dep_csv.exists():
        return []
    df = pd.read_csv(dep_csv)
    if "Test File Path" not in df.columns or "Dependency Files" not in df.columns:
        return []
    out = []
    for _, r in df.iterrows():
        tf = str(r["Test File Path"]).strip()
        deps = str(r["Dependency Files"]).strip()
        if tf:
            out.append((tf, deps))
    return out


def resolve_file_path(project_root: Path, raw_path: str) -> Tuple[str, Optional[Path]]:
    raw = str(raw_path).strip()
    if not raw:
        return "", None
    p = Path(raw)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    db_path = p.as_posix()
    return db_path, (p if p.exists() else None)


def import_one_project(conn, project: str):
    project_root = resolve_project_root(project, PROJECTS_BASE_DIR)
    if not project_root:
        print(f"[WARN] Project root not found for {project} under {PROJECTS_BASE_DIR}")
        return

    print(f"\n[INFO] Project {project} root = {project_root}")

    dep_pairs = load_dependency_map(project_root)
    dep_rows: List[Tuple[str, str, str]] = []
    needed_files: Set[str] = set()

    for tf_raw, deps_raw in dep_pairs:
        test_db_path, _ = resolve_file_path(project_root, tf_raw)
        if not test_db_path:
            continue
        needed_files.add(test_db_path)

        deps_list = []
        if deps_raw and deps_raw.lower() != "nan":
            for x in str(deps_raw).split(";"):
                x = x.strip()
                if not x:
                    continue
                dep_db_path, _ = resolve_file_path(project_root, x)
                if dep_db_path:
                    deps_list.append(dep_db_path)
                    needed_files.add(dep_db_path)

        for dep_db_path in deps_list:
            dep_rows.append((project, test_db_path, dep_db_path))

    if DRY_RUN:
        print(f"[DRY_RUN] Would upsert test_dependencies rows: {len(dep_rows)}")
    else:
        with conn.cursor() as cur:
            upsert_test_dependencies(cur, dep_rows)
        print(f"[INFO] Upserted test_dependencies for {project}: {len(dep_rows)} rows")

    file_rows: List[Tuple[str, str, str]] = []
    for db_path in sorted(needed_files):
        _, fs_path = resolve_file_path(project_root, db_path)
        if fs_path and fs_path.exists():
            file_rows.append((project, db_path, read_text_safely(fs_path)))

    if DRY_RUN:
        print(f"[DRY_RUN] Would upsert source_files: {len(file_rows)} files")
    else:
        with conn.cursor() as cur:
            upsert_source_files(cur, file_rows)
        print(f"[INFO] Upserted source_files for {project}: {len(file_rows)} files")

    xlsx = OBJECT_LIST_DIR / f"{project}_object_instantiations.xlsx"
    if not xlsx.exists():
        print(f"[WARN] Object instantiations XLSX not found for {project}: {xlsx}")
        return

    df = pd.read_excel(xlsx)
    required = ["Test Suite", "Test Case", "Class Name"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"{xlsx} missing column: {c}. Got: {list(df.columns)}")

    for c in ["Test Suite", "Test Case", "Class Name", "Mocked"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["TestSuiteBaseName"] = df["Test Suite"].apply(lambda x: basename_of_path(x))
    df["_grp"] = df["TestSuiteBaseName"] + "||" + df["Test Case"] + "||" + df["Class Name"]
    df["occurrence_in_case"] = df.groupby("_grp").cumcount() + 1
    df["source_row_index"] = df.index.astype(int)

    oi_rows: List[Tuple[str, str, str, str, str, str, int, str, int]] = []
    for _, r in df.iterrows():
        test_suite = str(r["Test Suite"]).strip()
        test_suite_basename = str(r["TestSuiteBaseName"]).strip()
        test_case = str(r["Test Case"]).strip()
        class_name = str(r["Class Name"]).strip()
        mocked = str(r.get("Mocked", "")).strip() if "Mocked" in df.columns else None
        occ = int(r["occurrence_in_case"])
        sri = int(r["source_row_index"])

        # Stable ID
        key = f"{project}|{test_suite_basename}|{test_case}|{class_name}|{occ}|{sri}"
        inst_id = str(uuid.uuid5(INST_NAMESPACE, key))
        oi_rows.append((inst_id, project, test_suite, test_suite_basename, test_case, class_name, occ, mocked, sri))

    if DRY_RUN:
        print(f"[DRY_RUN] Would upsert object_instantiations: {len(oi_rows)} rows")
    else:
        with conn.cursor() as cur:
            upsert_object_instantiations(cur, oi_rows)
        print(f"[INFO] Upserted object_instantiations for {project}: {len(oi_rows)} rows")


def main():
    if DRY_RUN:
        print("[DRY_RUN] No DB writes.\n")
        for p in PROJECTS:
            root = resolve_project_root(p, PROJECTS_BASE_DIR)
            xlsx = OBJECT_LIST_DIR / f"{p}_object_instantiations.xlsx"
            print(f"  - {p}: root={root} xlsx_exists={xlsx.exists()}")
        return

    conn = get_conn()
    try:
        ensure_unique_indexes(conn)
        conn.commit()

        for p in PROJECTS:
            import_one_project(conn, p)
            conn.commit()

        print("\n[INFO] All projects imported.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()