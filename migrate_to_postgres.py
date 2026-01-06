# migrate_to_postgres.py
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ====== 你本地的路径（沿用你的原配置）======
PROJECTS_BASE_DIR = Path(r"E:\Files\Mock_Project\ML\ML-experiment")
SAMPLED_CSV = Path("data/sampled_tests.csv")
OBJECT_LIST_DIR = Path("data\Object Instantiations Lists")
RULE_CSV = Path("data/mock_rules.csv")

DB_URL = ""

def norm_path(p: str) -> str:
    return str(Path(p).as_posix())

def safe_read_text(path: Path) -> str:
    """
    Read as text; if file seems binary or contains NUL, sanitize.
    """
    try:
        data = path.read_bytes()
    except Exception:
        return ""

    # Quick binary heuristic: NUL byte present => treat as binary-ish
    if b"\x00" in data:
        # remove NULs and decode best-effort
        print(f"[WARN] NUL bytes found, sanitizing: {path}")
        data = data.replace(b"\x00", b"")

    # Try utf-8 then latin-1
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return data.decode("latin-1", errors="ignore")
        except Exception:
            return ""

def resolve_project_root(project_name: str) -> Optional[Path]:
    exact = PROJECTS_BASE_DIR / project_name
    if exact.exists() and exact.is_dir():
        return exact
    candidates = sorted(PROJECTS_BASE_DIR.glob(f"{project_name}-*"))
    return candidates[0] if candidates else None

def load_dependencies_map(project_root: Path) -> pd.DataFrame:
    dep_csv = project_root / "test_dependencies_map.csv"
    if not dep_csv.exists():
        return pd.DataFrame(columns=["Test File Path", "Dependency Files"])
    return pd.read_csv(dep_csv)

TEXT_EXTS = {".java", ".kt", ".scala", ".groovy", ".xml", ".properties", ".yml", ".yaml", ".json", ".txt", ".md"}
MAX_BYTES = 2000000  # 2MB，按需调大/调小

def upsert_source_files(cur, project: str, paths: list[str]):
    rows = []
    for p in paths:
        fp = Path(p)

        # Skip non-existent
        if not fp.exists() or not fp.is_file():
            continue

        # Skip very large files
        try:
            size = fp.stat().st_size
        except Exception:
            continue
        if size > MAX_BYTES:
            continue

        # Skip likely binary by extension (keep only text-y ones)
        if fp.suffix.lower() not in TEXT_EXTS:
            continue

        content = safe_read_text(fp)

        # Ensure no NUL (double safety)
        if "\x00" in content:
            content = content.replace("\x00", "")

        normed = norm_path(str(fp.resolve()))
        rows.append((project, normed, content))

    if not rows:
        return

    execute_values(
        cur,
        """
        insert into source_files(project, path, content)
        values %s
        on conflict (project, path) do update set content = excluded.content
        """,
        rows,
        page_size=200
    )

def main():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # 1) rules
    if RULE_CSV.exists():
        df_rules = pd.read_csv(RULE_CSV)
        df_rules = df_rules.fillna("")
        rows = []
        for _, r in df_rules.iterrows():
            rows.append((str(r.get("Type","")), str(r.get("High Level Category","")), str(r.get("Criterion","")), str(r.get("Mock?",""))))
        if rows:
            execute_values(
                cur,
                """
                insert into rules(type, high_level_category, criterion, mock_decision)
                values %s
                """,
                rows,
                page_size=200
            )
        conn.commit()
        print("Imported rules.")

    # 2) sampled_tests
    df = pd.read_csv(SAMPLED_CSV)
    df = df.fillna("")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("Project","")).strip(),
            str(r.get("File","")).strip(),
            str(r.get("Method","")).strip(),
            str(r.get("TestAware","")).strip(),
            str(r.get("MockIntensity","")).strip(),
            int(pd.to_numeric(r.get("DependencyCount", 0), errors="coerce") or 0),
            str(r.get("CCTR_Bin","")).strip(),
            None,  # test_file_path 先空，后面根据依赖表/路径匹配再补也行
            None   # test_source 先空，后面再写
        ))
    execute_values(
        cur,
        """
        insert into sampled_tests(project, file, method, testaware, mockintensity, dependencycount, cctr_bin, test_file_path, test_source)
        values %s
        """,
        rows,
        page_size=500
    )
    conn.commit()
    print("Imported sampled_tests (basic columns).")

    # 3) per-project: dependencies + source files + object instantiations
    projects = sorted({str(x).strip() for x in df["Project"].tolist() if str(x).strip()})
    for project in projects:
        root = resolve_project_root(project)
        if not root:
            print(f"[WARN] project root not found for {project}")
            continue

        # 3.1 dependencies map
        dep_df = load_dependencies_map(root)
        dep_rows = []
        src_paths = set()

        for _, row in dep_df.iterrows():
            tf_raw = str(row.get("Test File Path","")).strip()
            deps_raw = str(row.get("Dependency Files","")).strip()
            if not tf_raw:
                continue

            tf = Path(tf_raw)
            if not tf.is_absolute():
                tf = (root / tf).resolve()
            else:
                tf = tf.resolve()

            test_path = norm_path(str(tf))
            src_paths.add(str(tf))

            if deps_raw:
                for p in deps_raw.split(";"):
                    p = p.strip()
                    if not p:
                        continue
                    dp = Path(p)
                    if not dp.is_absolute():
                        dp = (root / dp).resolve()
                    else:
                        dp = dp.resolve()
                    dep_path = norm_path(str(dp))
                    src_paths.add(str(dp))
                    dep_rows.append((project, test_path, dep_path))

        if dep_rows:
            execute_values(
                cur,
                """
                insert into test_dependencies(project, test_path, dep_path)
                values %s
                """,
                dep_rows,
                page_size=1000
            )
            conn.commit()
            print(f"Imported dependencies for {project}: {len(dep_rows)} rows.")

        # 3.2 source files (test + deps) content
        if src_paths:
            upsert_source_files(cur, project, sorted(src_paths))
            conn.commit()
            print(f"Upserted source_files for {project}: {len(src_paths)} files.")

        # 3.3 object instantiations xlsx
        xlsx_path = OBJECT_LIST_DIR / f"{project}_object_instantiations.xlsx"
        if not xlsx_path.exists():
            print(f"[WARN] no xlsx for {project}")
            continue

        odf = pd.read_excel(xlsx_path).fillna("")
        for col in ["Test Suite", "Test Case", "Class Name", "Mocked"]:
            if col not in odf.columns:
                odf[col] = ""
        odf["TestSuiteBaseName"] = odf["Test Suite"].apply(lambda x: Path(str(x)).name)

        # occurrence_in_case：同一 (suite_basename, test_case, class_name) 按出现顺序计数
        occ_counter = {}
        obj_rows = []
        for i, r in odf.iterrows():
            ts = str(r["Test Suite"]).strip()
            base = str(r["TestSuiteBaseName"]).strip()
            tc = str(r["Test Case"]).strip()
            cn = str(r["Class Name"]).strip()
            mocked = str(r["Mocked"]).strip()
            if not base or not tc or not cn:
                continue
            k = (base, tc, cn)
            occ_counter[k] = occ_counter.get(k, 0) + 1
            occ = occ_counter[k]
            obj_rows.append((project, ts, base, tc, cn, occ, mocked, int(i)))

        if obj_rows:
            execute_values(
                cur,
                """
                insert into object_instantiations(
                    project, test_suite, test_suite_basename, test_case, class_name,
                    occurrence_in_case, mocked, source_row_index
                )
                values %s
                """,
                obj_rows,
                page_size=1000
            )
            conn.commit()
            print(f"Imported object_instantiations for {project}: {len(obj_rows)} rows.")

    cur.close()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
