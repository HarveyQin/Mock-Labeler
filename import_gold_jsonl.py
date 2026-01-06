# import_gold_jsonl.py
import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

DB_URL = "postgresql://neondb_owner:npg_I3eq9uZprPXL@ep-frosty-rice-adydzy86-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # postgresql://... ?sslmode=require
JSONL_PATH = Path("outputs/object_instantiation_annotations.jsonl")  # 改成你的真实路径

GOLD_REVIEWER_ID = "gold"  # gold 标注的 reviewer_id
UPDATED_BY_FALLBACK = "gold"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Robust JSONL loader:
    - Each line should be a JSON object.
    - If a line contains multiple JSON objects concatenated, try to split.
    - If still fails, skip and report.
    """
    rows: List[Dict[str, Any]] = []
    bad_lines = 0
    multi_obj_lines = 0

    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue

        # 1) Normal case
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                # if it's not a dict, skip (unexpected for your file)
                bad_lines += 1
            continue
        except json.JSONDecodeError:
            pass

        # 2) Try to split concatenated JSON objects in the same line
        #    e.g. {...}{...} or {...} {...}
        # Use JSONDecoder raw_decode to parse sequentially.
        decoder = json.JSONDecoder()
        idx = 0
        got_any = False
        while idx < len(s):
            # skip whitespace
            while idx < len(s) and s[idx].isspace():
                idx += 1
            if idx >= len(s):
                break
            try:
                obj, end = decoder.raw_decode(s, idx)
                got_any = True
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad_lines += 1
                idx = end
            except json.JSONDecodeError:
                # cannot parse further
                break

        if got_any:
            multi_obj_lines += 1
            continue

        # 3) Give up: report and skip
        bad_lines += 1
        snippet = s[:200].replace("\t", "\\t")
        print(f"[WARN] Bad JSON at line {lineno}: {snippet}")

    print(f"[INFO] Loaded {len(rows)} JSON objects.")
    print(f"[INFO] Multi-object lines fixed: {multi_obj_lines}")
    print(f"[INFO] Bad lines skipped: {bad_lines}")
    return rows


def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(JSONL_PATH)

    ann = load_jsonl(JSONL_PATH)
    # group by matching key (project, suite_basename, test_case, class_name)
    g: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for r in ann:
        project = str(r.get("project", "")).strip()
        suite_basename = str(r.get("test_suite_basename", "")).strip()
        test_case = str(r.get("test_case", "")).strip()
        class_name = str(r.get("class_name", "")).strip()
        if not (project and suite_basename and test_case and class_name):
            continue
        g[(project, suite_basename, test_case, class_name)].append(r)

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False

    inserted = 0
    skipped_no_db_match = 0
    skipped_overflow = 0

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            batch = []

            # Iterate groups; for each group, fetch inst_ids ordered by occurrence
            for k, items in g.items():
                project, suite_basename, test_case, class_name = k

                cur.execute(
                    """
                    select inst_id::text as inst_id
                    from object_instantiations
                    where project=%s and test_suite_basename=%s and test_case=%s and class_name=%s
                    order by occurrence_in_case, source_row_index nulls last, inst_id
                    """,
                    (project, suite_basename, test_case, class_name),
                )
                inst_rows = cur.fetchall()
                inst_ids = [row["inst_id"] for row in inst_rows]

                if not inst_ids:
                    skipped_no_db_match += len(items)
                    continue

                # map each jsonl record to inst_id by order
                for i, r in enumerate(items):
                    if i >= len(inst_ids):
                        skipped_overflow += 1
                        continue

                    inst_id = inst_ids[i]
                    decision = str(r.get("decision", "uncertain")).strip() or "uncertain"
                    notes = str(r.get("notes", "") or "")
                    rule_ids = r.get("rule_ids", []) or []
                    rule_labels = r.get("rule_labels", []) or []

                    # updated_by: prefer from jsonl, else fallback
                    updated_by = str(r.get("updated_by", "") or UPDATED_BY_FALLBACK)

                    # timestamp: jsonl ts is like 2026-...Z; store as timestamptz by letting Postgres parse
                    ts = r.get("ts")  # may be None

                    batch.append((
                        inst_id, GOLD_REVIEWER_ID, decision, notes,
                        json.dumps(rule_ids), json.dumps(rule_labels),
                        ts, updated_by
                    ))

                    # flush in chunks
                    if len(batch) >= 2000:
                        execute_values(
                            cur,
                            """
                            insert into annotations_cv(inst_id, reviewer_id, decision, notes, rule_ids, rule_labels, updated_at, updated_by)
                            values %s
                            on conflict (inst_id, reviewer_id)
                            do update set
                              decision=excluded.decision,
                              notes=excluded.notes,
                              rule_ids=excluded.rule_ids,
                              rule_labels=excluded.rule_labels,
                              updated_at=coalesce(excluded.updated_at, now()),
                              updated_by=excluded.updated_by
                            """,
                            batch,
                            page_size=500,
                        )
                        inserted += len(batch)
                        batch.clear()

            # final flush
            if batch:
                execute_values(
                    cur,
                    """
                    insert into annotations_cv(inst_id, reviewer_id, decision, notes, rule_ids, rule_labels, updated_at, updated_by)
                    values %s
                    on conflict (inst_id, reviewer_id)
                    do update set
                      decision=excluded.decision,
                      notes=excluded.notes,
                      rule_ids=excluded.rule_ids,
                      rule_labels=excluded.rule_labels,
                      updated_at=coalesce(excluded.updated_at, now()),
                      updated_by=excluded.updated_by
                    """,
                    batch,
                    page_size=500,
                )
                inserted += len(batch)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("DONE")
    print("Upserted into annotations_cv:", inserted)
    print("Skipped (no DB inst match):", skipped_no_db_match)
    print("Skipped (more jsonl than DB occurrences):", skipped_overflow)


if __name__ == "__main__":
    main()
