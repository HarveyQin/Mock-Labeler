# -*- coding: utf-8 -*-
"""
inst_labeler_app_pg.py

Streamlit app for labeling whether each object instantiation SHOULD be mocked.
All data is read/written from Postgres (Neon).

Matching rule (guaranteed by user):
- object_instantiations matched by (project, test_suite_basename == basename(sampled_tests.file), test_case == sampled_tests.method)

Key design:
- object_instantiations has inst_id (UUID PK) + occurrence_in_case (to preserve duplicates)
- annotations is keyed by inst_id (UPSERT), safe for multi-user collaboration

Secrets:
- st.secrets["DB_URL"] = "postgresql://.../neondb?sslmode=require"
"""

import json
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor

VALID_DECISIONS = ["mock", "no-mock", "uncertain", "skip"]


# ===================== DB =====================

@st.cache_resource
def get_conn():
    db_url = st.secrets.get("DB_URL")
    if not db_url:
        raise RuntimeError(
            "Missing DB_URL in st.secrets. Add it in .streamlit/secrets.toml or Community Cloud Secrets."
        )
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def db_fetchall(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def db_fetchone(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def db_execute(sql: str, params: tuple = ()) -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)


# ===================== Helpers =====================

def basename_of_path(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\\", "/")
    return s.split("/")[-1]


def build_where_clause(
    sel_project: str,
    sel_bin: str,
    sel_aware: str,
    dep_range: Tuple[int, int],
    q: str
) -> Tuple[str, List[Any]]:
    clauses = []
    params: List[Any] = []

    if sel_project != "ALL":
        clauses.append("st.project = %s")
        params.append(sel_project)

    if sel_bin != "ALL":
        clauses.append("st.cctr_bin = %s")
        params.append(sel_bin)

    if sel_aware != "ALL":
        clauses.append("st.testaware = %s")
        params.append(sel_aware)

    clauses.append("coalesce(st.dependencycount, 0) between %s and %s")
    params.extend([int(dep_range[0]), int(dep_range[1])])

    if q:
        qq = f"%{q.lower()}%"
        clauses.append("(lower(st.file) like %s or lower(st.method) like %s)")
        params.extend([qq, qq])

    where = " where " + " and ".join(clauses) if clauses else ""
    return where, params


# ===================== Filter options =====================

@st.cache_data(show_spinner=False)
def get_filter_options() -> Dict[str, Any]:
    projects = [r["project"] for r in db_fetchall("select distinct project from sampled_tests order by project")]
    bins = [r["cctr_bin"] for r in db_fetchall(
        "select distinct cctr_bin from sampled_tests where cctr_bin is not null and cctr_bin <> '' order by cctr_bin"
    )]
    aware = [r["testaware"] for r in db_fetchall(
        "select distinct testaware from sampled_tests where testaware is not null and testaware <> '' order by testaware"
    )]
    dep = db_fetchone("select min(coalesce(dependencycount,0)) as mn, max(coalesce(dependencycount,0)) as mx from sampled_tests") or {"mn": 0, "mx": 0}
    return {
        "projects": projects,
        "bins": bins,
        "aware": aware,
        "dep_min": int(dep["mn"] or 0),
        "dep_max": int(dep["mx"] or 0),
    }


def get_cases_count(where: str, params: List[Any]) -> int:
    row = db_fetchone(f"select count(*) as n from sampled_tests st {where}", tuple(params))
    return int(row["n"])


def get_case_at_index(where: str, params: List[Any], offset: int) -> Optional[Dict[str, Any]]:
    rows = db_fetchall(
        f"""
        select st.*
        from sampled_tests st
        {where}
        order by st.id
        offset %s limit 1
        """,
        tuple(params + [offset]),
    )
    return rows[0] if rows else None


# ===================== Progress =====================

def get_progress_under_filters(where: str, params: List[Any]) -> Dict[str, int]:
    """
    total_insts: instantiation rows for all sampled_tests under filters
    labeled_insts: those with annotations.decision != 'uncertain'
    """
    sql_total = f"""
    with stf as (
      select st.project, st.method,
             regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename
      from sampled_tests st
      {where}
    )
    select count(*) as n
    from object_instantiations oi
    join stf
      on stf.project = oi.project
     and stf.method = oi.test_case
     and stf.suite_basename = oi.test_suite_basename
    """
    total = int((db_fetchone(sql_total, tuple(params)) or {"n": 0})["n"] or 0)

    sql_labeled = f"""
    with stf as (
      select st.project, st.method,
             regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename
      from sampled_tests st
      {where}
    )
    select count(*) as n
    from object_instantiations oi
    join stf
      on stf.project = oi.project
     and stf.method = oi.test_case
     and stf.suite_basename = oi.test_suite_basename
    join annotations a
      on a.inst_id = oi.inst_id
    where a.decision is not null and a.decision <> 'uncertain'
    """
    labeled = int((db_fetchone(sql_labeled, tuple(params)) or {"n": 0})["n"] or 0)
    return {"total_insts": total, "labeled_insts": labeled}


# ===================== Domain reads/writes =====================

@st.cache_data(show_spinner=False)
def load_rules() -> List[Dict[str, Any]]:
    return db_fetchall(
        """
        select id, type, high_level_category, criterion, mock_decision
        from rules
        order by id
        """
    )


def insert_rule(rule_type: str, high_level_cat: str, criterion: str, mock_decision: str, created_by: str) -> int:
    row = db_fetchone(
        """
        insert into rules(type, high_level_category, criterion, mock_decision, created_by)
        values (%s,%s,%s,%s,%s)
        returning id
        """,
        (rule_type, high_level_cat, criterion, mock_decision, created_by),
    )
    return int(row["id"])


def get_object_instantiations(project: str, suite_basename: str, test_case: str) -> List[Dict[str, Any]]:
    return db_fetchall(
        """
        select inst_id::text as inst_id,
               test_suite,
               test_suite_basename,
               test_case,
               class_name,
               occurrence_in_case,
               mocked,
               source_row_index
        from object_instantiations
        where project=%s and test_suite_basename=%s and test_case=%s
        order by class_name, occurrence_in_case, source_row_index nulls last, inst_id
        """,
        (project, suite_basename, test_case),
    )


def load_annotations_for_inst_ids(inst_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not inst_ids:
        return {}
    rows = db_fetchall(
        """
        select inst_id::text as inst_id,
               decision, notes, rule_ids, rule_labels, updated_at, updated_by
        from annotations
        where inst_id = any(%s::uuid[])
        """,
        (inst_ids,),
    )
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[r["inst_id"]] = r
    return out


def upsert_annotation(inst_id: str, decision: str, notes: str, rule_ids: List[int], rule_labels: List[str], updated_by: str):
    db_execute(
        """
        insert into annotations(inst_id, decision, notes, rule_ids, rule_labels, updated_at, updated_by)
        values (%s, %s, %s, %s::jsonb, %s::jsonb, now(), %s)
        on conflict (inst_id)
        do update set
            decision = excluded.decision,
            notes = excluded.notes,
            rule_ids = excluded.rule_ids,
            rule_labels = excluded.rule_labels,
            updated_at = now(),
            updated_by = excluded.updated_by
        """,
        (
            inst_id,
            decision,
            notes,
            json.dumps(rule_ids),
            json.dumps(rule_labels),
            updated_by,
        ),
    )


def get_source_content(project: str, path: str) -> str:
    row = db_fetchone(
        "select content from source_files where project=%s and path=%s",
        (project, path),
    )
    return (row or {}).get("content") or ""


def get_dependencies(project: str, test_path: Optional[str]) -> List[str]:
    if not test_path:
        return []
    rows = db_fetchall(
        """
        select dep_path
        from test_dependencies
        where project=%s and test_path=%s
        order by dep_path
        """,
        (project, test_path),
    )
    return [r["dep_path"] for r in rows]


def resolve_test_path(project: str, sampled_test_row: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Prefer sampled_tests.test_file_path; otherwise infer from test_dependencies using basename.
    Since you stated suite_basename+method will not collide, basename inference is acceptable.
    """
    tp = sampled_test_row.get("test_file_path")
    if tp:
        return str(tp), "from sampled_tests.test_file_path"

    base = basename_of_path(sampled_test_row.get("file") or "").lower()
    if not base:
        return None, "no file basename"

    rows = db_fetchall(
        """
        select distinct test_path
        from test_dependencies
        where project=%s and lower(test_path) like %s
        order by test_path
        limit 10
        """,
        (project, f"%/{base}"),
    )
    if len(rows) == 1:
        return rows[0]["test_path"], "inferred from test_dependencies (unique basename match)"
    if len(rows) > 1:
        return rows[0]["test_path"], f"inferred from test_dependencies (ambiguous basename match: {len(rows)}; using first)"
    return None, "not found in test_dependencies"


# ===================== UI =====================

st.set_page_config(page_title="Object Instantiation Mock Labeler (Neon/Postgres)", layout="wide")
st.title("🧪 Object Instantiation Mock Labeler (Neon/Postgres)")

# Identity (works immediately). Swap to OIDC later if needed.
with st.sidebar:
    st.header("User")
    user_name = st.text_input(
        "Your name / email (for updated_by)",
        value=st.session_state.get("user_name", "")
    ).strip()
    st.session_state["user_name"] = user_name or "unknown"

opts = get_filter_options()

st.sidebar.header("Filters")
sel_project = st.sidebar.selectbox("Project", ["ALL"] + opts["projects"], index=0)
sel_bin = st.sidebar.selectbox("CCTR_Bin", ["ALL"] + opts["bins"], index=0)
sel_aware = st.sidebar.selectbox("TestAware", ["ALL"] + opts["aware"], index=0)
dep_range = st.sidebar.slider(
    "DependencyCount range",
    min_value=opts["dep_min"],
    max_value=opts["dep_max"],
    value=(opts["dep_min"], opts["dep_max"])
)
q = st.sidebar.text_input("Search in File/Method")

where, params = build_where_clause(sel_project, sel_bin, sel_aware, dep_range, q)

# Progress
st.sidebar.header("Progress")
try:
    prog = get_progress_under_filters(where, params)
    tot = prog["total_insts"]
    lab = prog["labeled_insts"]
    ratio = (lab / tot) if tot else 0.0
    st.sidebar.progress(ratio)
    st.sidebar.caption(f"Under filters: labeled {lab} / {tot} instantiations")
except Exception as e:
    st.sidebar.warning(f"Progress unavailable: {e}")

# Pagination
if "case_idx" not in st.session_state:
    st.session_state.case_idx = 0

total_cases = get_cases_count(where, params)

def move_case(delta: int):
    if total_cases <= 0:
        st.session_state.case_idx = 0
        return
    st.session_state.case_idx = max(0, min(total_cases - 1, st.session_state.case_idx + delta))

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 6, 2])
with col_nav1:
    if st.button("⏮ Prev Case"):
        move_case(-1)
with col_nav2:
    if st.button("Next Case ⏭"):
        move_case(+1)
with col_nav3:
    st.write(f"Case {st.session_state.case_idx + 1} / {total_cases}")
with col_nav4:
    if total_cases > 0:
        jump = st.number_input("Jump to index", min_value=1, max_value=total_cases, value=st.session_state.case_idx + 1, step=1)
        if st.button("Go"):
            st.session_state.case_idx = int(jump) - 1

if total_cases <= 0:
    st.info("No test cases under current filter.")
    st.stop()

row = get_case_at_index(where, params, st.session_state.case_idx)
if not row:
    st.warning("Cannot load case row.")
    st.stop()

project_name = row["project"]
file_col = row["file"]
method_name = row["method"]
suite_basename = basename_of_path(file_col)

st.subheader("📄 Test Case")
st.markdown(f"**Project:** `{project_name}`  |  **File:** `{file_col}`  |  **Method:** `{method_name}`")
st.caption(
    f"TestAware: {row.get('testaware')} | "
    f"MockIntensity: {row.get('mockintensity')} | "
    f"DependencyCount: {int(row.get('dependencycount') or 0)} | "
    f"CCTR_Bin: {row.get('cctr_bin')}"
)

# Resolve test_path and load test source
test_path, test_path_note = resolve_test_path(project_name, row)
test_src = row.get("test_source") or (get_source_content(project_name, test_path) if test_path else "")

st.subheader("🧾 Test Source")
st.caption(f"Resolved test_path: {test_path}  ({test_path_note})")
if not test_src:
    st.warning("⚠️ No test source content found in DB for this case.")
else:
    st.code(test_src, language="java")

# Dependencies
st.subheader("🔗 Dependency Sources (from DB:test_dependencies + source_files)")
deps = get_dependencies(project_name, test_path)
if not deps:
    st.caption("No dependencies found for this test_path.")
else:
    # optional: allow limit control
    max_deps = st.number_input("Max dependency files to show", min_value=5, max_value=200, value=40, step=5)
    for dp in deps[: int(max_deps)]:
        with st.expander(dp):
            st.code(get_source_content(project_name, dp), language="java")

# Instantiations
st.subheader("🧩 Object Instantiations (from DB:object_instantiations)")
insts = get_object_instantiations(project_name, suite_basename, method_name)
if not insts:
    st.info("No instantiations found for this test case in DB (check suite_basename + method).")
    st.stop()

st.caption(f"Found {len(insts)} instantiation(s). Duplicates preserved via occurrence_in_case and inst_id.")

inst_ids = [r["inst_id"] for r in insts]
ann_map = load_annotations_for_inst_ids(inst_ids)

# Rules
rules = load_rules()
rule_options: List[Tuple[str, int]] = []
label_to_id: Dict[str, int] = {}
for r in rules:
    rid = int(r["id"])
    label = f"{rid:02d} [{r.get('type','')}] {r.get('criterion','')} ({r.get('mock_decision','')})"
    rule_options.append((label, rid))
    label_to_id[label] = rid

# Per-inst UI meta for case-level save
case_inst_ui: List[Dict[str, Any]] = []

for idx, inst in enumerate(insts):
    inst_id = inst["inst_id"]
    class_name = inst["class_name"]
    occurrence = int(inst["occurrence_in_case"] or 0)
    existing_mocked = inst.get("mocked", "")
    testsuite_name = inst.get("test_suite", suite_basename)

    ui_suffix = f"{inst_id}_{idx}"

    prev = ann_map.get(inst_id, {})
    default_dec = prev.get("decision", "uncertain")
    default_note = (prev.get("notes") or "")

    prev_rule_ids = prev.get("rule_ids") or []
    if isinstance(prev_rule_ids, str):
        try:
            prev_rule_ids = json.loads(prev_rule_ids)
        except Exception:
            prev_rule_ids = []
    prev_rule_ids = [int(x) for x in prev_rule_ids if str(x).isdigit()]
    default_labels = [lab for (lab, rid) in rule_options if rid in prev_rule_ids]

    with st.container(border=True):
        st.markdown(
            f"**[{idx+1}] Class:** `{class_name}`  |  "
            f"**Occurrence:** `#{occurrence}`  |  "
            f"**Existing Mocked:** `{existing_mocked}`"
        )
        st.caption(f"inst_id: {inst_id} | Test Suite: {testsuite_name}")

        st.markdown("**Applicable Rules**")
        st.multiselect(
            "Select rules applied to this instantiation",
            options=[lab for (lab, _) in rule_options],
            default=default_labels,
            key=f"rules_{ui_suffix}"
        )

        # Add rule (auto-refresh + auto-select)
        with st.expander("➕ Add new rule"):
            new_rule_type = st.selectbox("Type", ["Base Rule", "Extended Rule", "Other"], key=f"new_type_{ui_suffix}")
            new_rule_cat = st.text_input("High Level Category", value="Class Characteristics", key=f"new_cat_{ui_suffix}")
            new_rule_criterion = st.text_area("Criterion (when this rule applies)", key=f"new_criterion_{ui_suffix}")
            new_rule_mock = st.text_input("Mock? (Yes / No / etc.)", value="Yes", key=f"new_mock_{ui_suffix}")

            if st.button("💾 Add rule & select it", key=f"add_rule_{ui_suffix}"):
                if new_rule_criterion.strip():
                    created_by = st.session_state["user_name"]
                    new_id = insert_rule(
                        new_rule_type.strip(),
                        new_rule_cat.strip(),
                        new_rule_criterion.strip(),
                        new_rule_mock.strip(),
                        created_by,
                    )
                    # select immediately
                    new_label = f"{new_id:02d} [{new_rule_type.strip()}] {new_rule_criterion.strip()} ({new_rule_mock.strip()})"
                    multi_key = f"rules_{ui_suffix}"
                    cur = st.session_state.get(multi_key, []) or []
                    if new_label not in cur:
                        st.session_state[multi_key] = cur + [new_label]

                    # refresh cached rules/progress immediately
                    st.cache_data.clear()
                    st.success(f"Rule added: ID={new_id}")
                    st.rerun()
                else:
                    st.warning("Criterion is required.")

        st.radio(
            f"Decision for `{class_name}` (#{occurrence})",
            options=VALID_DECISIONS,
            horizontal=True,
            index=VALID_DECISIONS.index(default_dec) if default_dec in VALID_DECISIONS else 2,
            key=f"dec_{ui_suffix}"
        )
        st.text_area(
            f"Notes for `{class_name}` (#{occurrence})",
            value=default_note,
            height=80,
            key=f"note_{ui_suffix}"
        )

        case_inst_ui.append({
            "inst_id": inst_id,
            "rules_state_key": f"rules_{ui_suffix}",
            "dec_state_key": f"dec_{ui_suffix}",
            "note_state_key": f"note_{ui_suffix}",
        })

# Case Actions
st.subheader("✅ Case Actions")

# Case progress (strict: decision != uncertain)
labeled_insts = 0
for meta in case_inst_ui:
    prev = ann_map.get(meta["inst_id"])
    if prev and prev.get("decision") and prev.get("decision") != "uncertain":
        labeled_insts += 1

st.progress(labeled_insts / len(case_inst_ui) if case_inst_ui else 0.0)
st.caption(f"This case: labeled {labeled_insts} / {len(case_inst_ui)}")

btn_c1, btn_c2 = st.columns([1, 1])
with btn_c1:
    save_all = st.button(
        "💾 Save ALL instantiations",
        key=f"save_all_{project_name}_{suite_basename}_{method_name}_{st.session_state.case_idx}"
    )
with btn_c2:
    save_all_next = st.button(
        "💾 Save ALL & Next Case",
        key=f"save_all_next_{project_name}_{suite_basename}_{method_name}_{st.session_state.case_idx}"
    )

def build_recs_for_case() -> List[Dict[str, Any]]:
    recs = []
    for meta in case_inst_ui:
        selected_labels = st.session_state.get(meta["rules_state_key"], []) or []
        selected_rule_ids = [label_to_id.get(lab) for lab in selected_labels if lab in label_to_id]
        selected_rule_ids = [int(rid) for rid in selected_rule_ids if isinstance(rid, int)]

        decision = st.session_state.get(meta["dec_state_key"], "uncertain")
        notes = st.session_state.get(meta["note_state_key"], "")

        recs.append({
            "inst_id": meta["inst_id"],
            "decision": decision,
            "notes": notes,
            "rule_ids": selected_rule_ids,
            "rule_labels": selected_labels,
        })
    return recs

if save_all or save_all_next:
    updated_by = st.session_state["user_name"]
    recs = build_recs_for_case()

    for r in recs:
        upsert_annotation(
            inst_id=r["inst_id"],
            decision=r["decision"],
            notes=r["notes"],
            rule_ids=r["rule_ids"],
            rule_labels=r["rule_labels"],
            updated_by=updated_by,
        )

    st.cache_data.clear()
    st.success(f"Saved {len(recs)} instantiation(s).")

    if save_all_next and st.session_state.case_idx < total_cases - 1:
        st.session_state.case_idx += 1
    st.rerun()
