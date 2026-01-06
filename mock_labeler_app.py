# -*- coding: utf-8 -*-
"""
mock_labeler_app.py

Cross-validation-ready Streamlit app:
- Scope of review: ONLY object_instantiations that belong to sampled_tests
  (matched by project + suite_basename(file) + method(test_case))
- Multi-reviewer annotations: annotations_cv(inst_id, reviewer_id)
- Assignments: assignments(inst_id, reviewer_id, bucket)

Modes:
- Review: reviewers label ONLY their assigned instantiations (including gold if reviewer_id='gold')
- Compare: compare reviewer vs gold and export disagreements
- Admin: create/reset assignments and view counts

Secrets:
- st.secrets["DB_URL"] = "postgresql://.../neondb?sslmode=require"
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import re
import html
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

VALID_DECISIONS = ["mock", "no-mock", "uncertain", "skip"]
GOLD_REVIEWER_ID = "gold"


# ===================== DB =====================

@st.cache_resource
def get_conn():
    db_url = st.secrets.get("DB_URL")
    if not db_url:
        raise RuntimeError("Missing DB_URL in st.secrets.")
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


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

import re
import html

def java_class_name_from_path(p: str) -> str:
    """
    Show only class name for a java file path.
    Example:
      .../src/main/java/org/foo/BarBaz.java -> BarBaz
    Fallback: basename without extension.
    """
    if not p:
        return ""
    p = str(p).replace("\\", "/")
    base = p.split("/")[-1]
    if base.lower().endswith(".java"):
        base = base[:-5]
    return base


def extract_java_method(src: str, method_name: str, max_chars: int = 200000) -> str:
    """
    Robust extraction of a Java method declaration by name.

    Strategy:
    - find occurrences of 'methodName('
    - filter out likely invocations (preceded by '.' or part of identifier)
    - parse matching ')' for the parameter list
    - skip whitespace + optional 'throws ...'
    - require next significant token to be '{'
    - then brace-match to capture full method body

    Not a full Java parser, but works well for most test suites.
    """
    if not src or not method_name:
        return ""

    s = src[:max_chars] if len(src) > max_chars else src
    needle = method_name + "("

    def is_ident_char(ch: str) -> bool:
        return ch.isalnum() or ch in ["_", "$"]

    def skip_ws_and_comments(i: int) -> int:
        """Skip whitespace and //... or /*...*/ comments."""
        n = len(s)
        while i < n:
            if s[i].isspace():
                i += 1
                continue
            # line comment
            if s.startswith("//", i):
                j = s.find("\n", i + 2)
                return n if j == -1 else j + 1
            # block comment
            if s.startswith("/*", i):
                j = s.find("*/", i + 2)
                if j == -1:
                    return n
                i = j + 2
                continue
            break
        return i

    def find_matching_paren(open_paren_idx: int) -> int:
        """Given index of '(', find matching ')' considering strings/chars."""
        i = open_paren_idx
        n = len(s)
        depth = 0
        in_str = False
        in_chr = False
        escape = False
        while i < n:
            ch = s[i]
            if escape:
                escape = False
                i += 1
                continue
            if (in_str or in_chr) and ch == "\\":
                escape = True
                i += 1
                continue
            if ch == '"' and not in_chr:
                in_str = not in_str
                i += 1
                continue
            if ch == "'" and not in_str:
                in_chr = not in_chr
                i += 1
                continue

            if not in_str and not in_chr:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    def find_open_brace(after_idx: int) -> int:
        """
        Find the '{' that starts the method body after the parameter list.
        Allows: whitespace/comments + optional 'throws ...' (possibly multi-token) before '{'
        """
        i = skip_ws_and_comments(after_idx)

        # optional "throws ..."
        if s.startswith("throws", i) and (i == 0 or not is_ident_char(s[i - 1])) and (i + 6 >= len(s) or not is_ident_char(s[i + 6])):
            i += 6
            # scan until we hit '{' or ';' (abstract/interface method) or end
            n = len(s)
            in_str = False
            in_chr = False
            escape = False
            while i < n:
                i = skip_ws_and_comments(i)
                if i >= n:
                    break
                ch = s[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if (in_str or in_chr) and ch == "\\":
                    escape = True
                    i += 1
                    continue
                if ch == '"' and not in_chr:
                    in_str = not in_str
                    i += 1
                    continue
                if ch == "'" and not in_str:
                    in_chr = not in_chr
                    i += 1
                    continue
                if not in_str and not in_chr:
                    if ch == "{":
                        return i
                    if ch == ";":
                        return -1
                i += 1

        i = skip_ws_and_comments(i)
        return i if i < len(s) and s[i] == "{" else -1

    def find_matching_brace(open_brace_idx: int) -> int:
        """Given index of '{', find matching '}' considering strings/chars."""
        i = open_brace_idx
        n = len(s)
        depth = 0
        in_str = False
        in_chr = False
        escape = False
        while i < n:
            ch = s[i]
            if escape:
                escape = False
                i += 1
                continue
            if (in_str or in_chr) and ch == "\\":
                escape = True
                i += 1
                continue
            if ch == '"' and not in_chr:
                in_str = not in_str
                i += 1
                continue
            if ch == "'" and not in_str:
                in_chr = not in_chr
                i += 1
                continue

            if not in_str and not in_chr:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    # iterate all occurrences
    start_search = 0
    while True:
        idx = s.find(needle, start_search)
        if idx == -1:
            break

        name_start = idx
        name_end = idx + len(method_name)

        # Filter out invocations like obj.methodName( or Class.methodName(
        prev = s[name_start - 1] if name_start - 1 >= 0 else ""
        if prev == ".":
            start_search = idx + 1
            continue

        # Ensure method name is not part of a larger identifier
        if prev and is_ident_char(prev):
            start_search = idx + 1
            continue

        # Parse params
        open_paren = name_end
        close_paren = find_matching_paren(open_paren)
        if close_paren == -1:
            start_search = idx + 1
            continue

        # Find body '{'
        open_brace = find_open_brace(close_paren + 1)
        if open_brace == -1:
            start_search = idx + 1
            continue

        close_brace = find_matching_brace(open_brace)
        if close_brace == -1:
            start_search = idx + 1
            continue

        # choose a reasonable start: beginning of the line that contains the declaration
        line_start = s.rfind("\n", 0, name_start)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1

        return s[line_start:close_brace + 1].strip()

        # next
        start_search = idx + 1

    return ""

def add_markers_around_method(full_src: str, method_src: str, method_name: str) -> str:
    """
    Insert visible comment markers around the first occurrence of method_src in full_src.
    Keeps st.code() syntax highlighting.
    """
    if not full_src:
        return ""
    if not method_src:
        return full_src

    start_marker = f"// >>>>>>> CURRENT REVIEW TEST CASE: {method_name} >>>>>>>"
    end_marker   = f"// <<<<<<< END CURRENT REVIEW TEST CASE: {method_name} <<<<<<<"

    idx = full_src.find(method_src)
    if idx < 0:
        return full_src

    before = full_src[:idx].rstrip("\n")
    after = full_src[idx + len(method_src):].lstrip("\n")

    return f"{before}\n{start_marker}\n{method_src}\n{end_marker}\n{after}"


def highlight_method_in_source_html(full_src: str, method_src: str) -> str:
    """
    Render full source as HTML <pre><code> with method_src highlighted.
    """
    if not full_src:
        return "<pre><code></code></pre>"

    esc_full = html.escape(full_src)

    if method_src and method_src in full_src:
        esc_method = html.escape(method_src)
        # highlight first occurrence
        esc_full = esc_full.replace(
            esc_method,
            f'<span class="hl">{esc_method}</span>',
            1
        )

    return f"<pre class='code'><code>{esc_full}</code></pre>"


# ===================== Sampled scope (instantiations under sampled_tests) =====================

@st.cache_data(show_spinner=False)
def get_sampled_scope_inst_ids() -> List[str]:
    """
    Return all inst_id (uuid as text) that belong to sampled_tests scope.
    Deterministic ordering for assignment slicing.
    """
    rows = db_fetchall(
        """
        with stf as (
          select
            st.project,
            regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename,
            st.method
          from sampled_tests st
        )
        select oi.inst_id::text as inst_id
        from object_instantiations oi
        join stf
          on stf.project = oi.project
         and stf.suite_basename = oi.test_suite_basename
         and stf.method = oi.test_case
        order by oi.project, oi.test_suite_basename, oi.test_case, oi.class_name, oi.occurrence_in_case, oi.inst_id
        """
    )
    return [r["inst_id"] for r in rows]


@st.cache_data(show_spinner=False)
def get_sampled_cases_for_reviewer(reviewer_id: str) -> List[Dict[str, Any]]:
    """
    List sampled_tests cases that have at least one instantiation assigned to reviewer.
    """
    rows = db_fetchall(
        """
        with st as (
          select
            st.id,
            st.project,
            st.file,
            st.method,
            regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename,
            st.testaware,
            st.mockintensity,
            st.dependencycount,
            st.cctr_bin,
            st.test_file_path,
            st.test_source
          from sampled_tests st
        ),
        assigned as (
          select distinct
            oi.project, oi.test_suite_basename, oi.test_case
          from assignments a
          join object_instantiations oi on oi.inst_id = a.inst_id
          where a.reviewer_id = %s
        )
        select st.*
        from st
        join assigned
          on assigned.project = st.project
         and assigned.test_suite_basename = st.suite_basename
         and assigned.test_case = st.method
        order by st.id
        """,
        (reviewer_id,),
    )
    return rows

@st.cache_data(show_spinner=False)
def get_sampled_cases_all() -> List[Dict[str, Any]]:
    """
    All sampled_tests cases (full sampled scope).
    """
    return db_fetchall(
        """
        select
          st.id,
          st.project,
          st.file,
          st.method,
          regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename,
          st.testaware,
          st.mockintensity,
          st.dependencycount,
          st.cctr_bin,
          st.test_file_path,
          st.test_source
        from sampled_tests st
        order by st.id
        """
    )

def get_instantiations_for_case(project: str, suite_basename: str, method: str) -> List[Dict[str, Any]]:
    return db_fetchall(
        """
        select
          inst_id::text as inst_id,
          project,
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
        (project, suite_basename, method),
    )


def filter_insts_to_assigned(insts: List[Dict[str, Any]], reviewer_id: str) -> List[Dict[str, Any]]:
    if not insts:
        return []
    inst_ids = [x["inst_id"] for x in insts]
    rows = db_fetchall(
        """
        select inst_id::text as inst_id
        from assignments
        where reviewer_id=%s and inst_id = any(%s::uuid[])
        """,
        (reviewer_id, inst_ids),
    )
    allowed = set(r["inst_id"] for r in rows)
    return [x for x in insts if x["inst_id"] in allowed]


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


def load_annotations_cv(inst_ids: List[str], reviewer_id: str) -> Dict[str, Dict[str, Any]]:
    if not inst_ids:
        return {}
    rows = db_fetchall(
        """
        select
          inst_id::text as inst_id,
          reviewer_id,
          decision,
          notes,
          rule_ids,
          rule_labels,
          updated_at,
          updated_by
        from annotations_cv
        where reviewer_id=%s and inst_id = any(%s::uuid[])
        """,
        (reviewer_id, inst_ids),
    )
    out = {}
    for r in rows:
        out[r["inst_id"]] = r
    return out


def upsert_annotation_cv(
    inst_id: str,
    reviewer_id: str,
    decision: str,
    notes: str,
    rule_ids: List[int],
    rule_labels: List[str],
    updated_by: str
) -> None:
    db_execute(
        """
        insert into annotations_cv(inst_id, reviewer_id, decision, notes, rule_ids, rule_labels, updated_at, updated_by)
        values (%s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, now(), %s)
        on conflict (inst_id, reviewer_id)
        do update set
          decision=excluded.decision,
          notes=excluded.notes,
          rule_ids=excluded.rule_ids,
          rule_labels=excluded.rule_labels,
          updated_at=now(),
          updated_by=excluded.updated_by
        """,
        (inst_id, reviewer_id, decision, notes, json.dumps(rule_ids), json.dumps(rule_labels), updated_by),
    )


def get_source_content(project: str, path: str) -> str:
    if not path:
        return ""
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
    Prefer sampled_tests.test_file_path; else infer by basename from test_dependencies.
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


# ===================== Assignments =====================

def create_assignments_by_counts(
    reviewers_and_counts: List[Tuple[str, int]],
    allow_partial: bool = True
) -> Dict[str, Any]:
    """
    Manually assign N instantiations to each reviewer in order (stable slicing).
    reviewers_and_counts: [(reviewer_id, n), ...]
    - If sum(n) < total and allow_partial=True: leftover will remain unassigned.
    - If sum(n) > total: error.
    Overwrites existing assignments for these reviewers within sampled scope inst_ids.
    """
    reviewers_and_counts = [(r.strip(), int(n)) for r, n in reviewers_and_counts if r and r.strip()]
    if len(reviewers_and_counts) != 4:
        raise ValueError("Please provide exactly 4 reviewers (each with a count).")

    if any(n < 0 for _, n in reviewers_and_counts):
        raise ValueError("Counts must be >= 0.")

    inst_ids = get_sampled_scope_inst_ids()
    total_scope = len(inst_ids)
    total_need = sum(n for _, n in reviewers_and_counts)

    if total_need > total_scope:
        raise ValueError(f"Requested {total_need} but sampled scope has only {total_scope} instantiations.")

    # delete existing assignments for these reviewers within sampled scope
    reviewer_ids = [r for r, _ in reviewers_and_counts]
    db_execute(
        """
        delete from assignments
        where reviewer_id = any(%s)
          and inst_id = any(%s::uuid[])
        """,
        (reviewer_ids, inst_ids),
    )

    # slice deterministically
    rows = []
    cur = 0
    for idx, (reviewer, n) in enumerate(reviewers_and_counts):
        bucket = idx + 1
        take = inst_ids[cur: cur + n]
        for inst_id in take:
            rows.append((inst_id, reviewer, bucket))
        cur += n

    if rows:
        conn = get_conn()
        with conn.cursor() as cur2:
            execute_values(
                cur2,
                """
                insert into assignments(inst_id, reviewer_id, bucket)
                values %s
                on conflict (inst_id, reviewer_id) do update
                  set bucket = excluded.bucket,
                      assigned_at = now()
                """,
                rows,
                page_size=1000,
            )

    counts = {r: n for r, n in reviewers_and_counts}
    leftover = total_scope - total_need
    return {
        "scope_total": total_scope,
        "assigned_total": total_need,
        "leftover_unassigned": leftover,
        "per_reviewer_requested": counts,
    }


@st.cache_data(show_spinner=False)
def get_assignment_counts() -> List[Dict[str, Any]]:
    return db_fetchall(
        """
        select reviewer_id, bucket, count(*) as n
        from assignments
        group by reviewer_id, bucket
        order by bucket, reviewer_id
        """
    )


def get_reviewer_progress(reviewer_id: str) -> Tuple[int, int]:
    total_row = db_fetchone("select count(*) as n from assignments where reviewer_id=%s", (reviewer_id,))
    total = int((total_row or {}).get("n") or 0)

    labeled_row = db_fetchone(
        """
        select count(*) as n
        from assignments a
        join annotations_cv ac
          on ac.inst_id = a.inst_id and ac.reviewer_id = a.reviewer_id
        where a.reviewer_id=%s and ac.decision <> 'uncertain'
        """,
        (reviewer_id,),
    )
    labeled = int((labeled_row or {}).get("n") or 0)
    return labeled, total


# ===================== Compare =====================

def compare_gold_vs_reviewer(reviewer_id: str) -> pd.DataFrame:
    rows = db_fetchall(
        """
        with stf as (
          select
            st.project,
            regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename,
            st.method
          from sampled_tests st
        ),
        scope as (
          select
            oi.inst_id,
            oi.project,
            oi.test_suite_basename,
            oi.test_case,
            oi.class_name,
            oi.occurrence_in_case
          from object_instantiations oi
          join stf
            on stf.project = oi.project
           and stf.suite_basename = oi.test_suite_basename
           and stf.method = oi.test_case
        ),
        assigned as (
          select s.*
          from scope s
          join assignments a on a.inst_id = s.inst_id
          where a.reviewer_id = %s
        ),
        gold as (
          select inst_id, decision as gold_decision
          from annotations_cv
          where reviewer_id = %s
        ),
        rev as (
          select inst_id, decision as reviewer_decision
          from annotations_cv
          where reviewer_id = %s
        )
        select
          a.inst_id::text as inst_id,
          a.project,
          a.test_suite_basename as suite_basename,
          a.test_case,
          a.class_name,
          a.occurrence_in_case as occurrence,
          g.gold_decision,
          r.reviewer_decision
        from assigned a
        left join gold g on g.inst_id = a.inst_id
        left join rev  r on r.inst_id = a.inst_id
        order by a.project, a.test_suite_basename, a.test_case, a.class_name, a.occurrence_in_case, a.inst_id
        """,
        (reviewer_id, GOLD_REVIEWER_ID, reviewer_id),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["agree"] = (df["gold_decision"].fillna("") == df["reviewer_decision"].fillna(""))
    return df


def confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df["gold_decision"].fillna("MISSING")
    r = df["reviewer_decision"].fillna("MISSING")
    cats = VALID_DECISIONS + ["MISSING"]
    mat = (
        pd.crosstab(g, r, rownames=["gold"], colnames=["reviewer"], dropna=False)
        .reindex(index=cats, columns=cats, fill_value=0)
    )
    return mat


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ===================== UI =====================

st.set_page_config(page_title="Mock Labeler (Cross-validation)", layout="wide")
st.title("🧪 Mock Labeler (Cross-validation)")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Review", "Compare", "Admin"], index=0)

    st.header("Identity")
    reviewer_id = st.text_input(
        "reviewer_id",
        value=st.session_state.get("reviewer_id", "")
    ).strip()
    st.session_state["reviewer_id"] = reviewer_id
    # st.caption("Tip: input 'gold' to view/edit gold labels in Review mode.")


# ---------- Admin ----------
if mode == "Admin":
    st.subheader("🛠 Admin: Assignments (Manual counts)")

    inst_ids = get_sampled_scope_inst_ids()
    scope_total = len(inst_ids)
    st.caption(f"Sampled scope instantiations: {scope_total}")

    st.write("Provide **exactly 4** reviewers and how many instantiations each should review. "
             "Assignments are created by deterministic slicing of the sampled-scope inst_id list.")

    c1, c2 = st.columns(2)
    with c1:
        r1 = st.text_input("Reviewer #1 (bucket 1)", value=st.session_state.get("r1", "")).strip()
        n1 = st.number_input("Count #1", min_value=0, max_value=scope_total, value=int(st.session_state.get("n1", 0) or 0), step=1)
        r2 = st.text_input("Reviewer #2 (bucket 2)", value=st.session_state.get("r2", "")).strip()
        n2 = st.number_input("Count #2", min_value=0, max_value=scope_total, value=int(st.session_state.get("n2", 0) or 0), step=1)
    with c2:
        r3 = st.text_input("Reviewer #3 (bucket 3)", value=st.session_state.get("r3", "")).strip()
        n3 = st.number_input("Count #3", min_value=0, max_value=scope_total, value=int(st.session_state.get("n3", 0) or 0), step=1)
        r4 = st.text_input("Reviewer #4 (bucket 4)", value=st.session_state.get("r4", "")).strip()
        n4 = st.number_input("Count #4", min_value=0, max_value=scope_total, value=int(st.session_state.get("n4", 0) or 0), step=1)

    st.session_state.update({"r1": r1, "r2": r2, "r3": r3, "r4": r4, "n1": int(n1), "n2": int(n2), "n3": int(n3), "n4": int(n4)})

    total_need = int(n1 + n2 + n3 + n4)
    st.info(f"Requested total = {total_need} / scope total = {scope_total} (leftover = {scope_total - total_need})")

    if st.button("✅ Create / Overwrite assignments (manual counts)"):
        try:
            res = create_assignments_by_counts(
                [(r1, int(n1)), (r2, int(n2)), (r3, int(n3)), (r4, int(n4))],
                allow_partial=True
            )
            st.cache_data.clear()
            st.success("Assignments created/overwritten.")
            st.json(res)
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Current assignment counts")
    ac = get_assignment_counts()
    if ac:
        st.dataframe(pd.DataFrame(ac), use_container_width=True)
    else:
        st.info("No assignments yet.")

    st.stop()


# ---------- Review ----------
if mode == "Review":
    if not reviewer_id:
        st.warning("Please input your reviewer_id in the sidebar.")
        st.stop()

    st.sidebar.header("Your progress")

    if reviewer_id == GOLD_REVIEWER_ID:
        # total = all instantiations under sampled scope
        total_row = db_fetchone(
            """
            with stf as (
              select
                st.project,
                regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename,
                st.method
              from sampled_tests st
            )
            select count(*) as n
            from object_instantiations oi
            join stf
              on stf.project = oi.project
             and stf.suite_basename = oi.test_suite_basename
             and stf.method = oi.test_case
            """
        )
        total = int((total_row or {}).get("n") or 0)

        labeled_row = db_fetchone(
            """
            with stf as (
              select
                st.project,
                regexp_replace(replace(st.file, '\\\\', '/'), '^.*/', '') as suite_basename,
                st.method
              from sampled_tests st
            ),
            scope as (
              select oi.inst_id
              from object_instantiations oi
              join stf
                on stf.project = oi.project
               and stf.suite_basename = oi.test_suite_basename
               and stf.method = oi.test_case
            )
            select count(*) as n
            from annotations_cv ac
            join scope s on s.inst_id = ac.inst_id
            where ac.reviewer_id = %s and ac.decision <> 'uncertain'
            """,
            (GOLD_REVIEWER_ID,),
        )
        labeled = int((labeled_row or {}).get("n") or 0)
    else:
        labeled, total = get_reviewer_progress(reviewer_id)

    st.sidebar.progress((labeled / total) if total else 0.0)
    st.sidebar.caption(f"{labeled} / {total} labeled (decision != uncertain)")

    # gold: can browse full sampled scope without assignments
    if reviewer_id == GOLD_REVIEWER_ID:
        cases = get_sampled_cases_all()
    else:
        cases = get_sampled_cases_for_reviewer(reviewer_id)

    if not cases:
        st.info("No cases available for this reviewer_id. For non-gold reviewers, ask admin to create assignments.")
        st.stop()

    if "case_idx" not in st.session_state:
        st.session_state.case_idx = 0

    def move_case(delta: int):
        n = len(cases)
        if n <= 0:
            st.session_state.case_idx = 0
        else:
            st.session_state.case_idx = max(0, min(n - 1, st.session_state.case_idx + delta))

    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 6, 2])
    with col_nav1:
        if st.button("⏮ Prev Case"):
            move_case(-1)
    with col_nav2:
        if st.button("Next Case ⏭"):
            move_case(+1)
    with col_nav3:
        st.write(f"Assigned Case {st.session_state.case_idx + 1} / {len(cases)}")
    with col_nav4:
        jump = st.number_input("Jump", min_value=1, max_value=len(cases), value=st.session_state.case_idx + 1, step=1)
        if st.button("Go"):
            st.session_state.case_idx = int(jump) - 1

    row = cases[st.session_state.case_idx]
    project_name = row["project"]
    file_col = row["file"]
    method_name = row["method"]
    suite_basename = basename_of_path(file_col)

    st.subheader("📄 Test Case (Assigned)")
    st.markdown(f"**Reviewer:** `{reviewer_id}`")
    st.markdown(f"**Project:** `{project_name}`  |  **File:** `{file_col}`  |  **Method:** `{method_name}`")
    st.caption(
        f"TestAware: {row.get('testaware')} | "
        f"MockIntensity: {row.get('mockintensity')} | "
        f"DependencyCount: {safe_int(row.get('dependencycount'))} | "
        f"CCTR_Bin: {row.get('cctr_bin')}"
    )

    test_path, test_path_note = resolve_test_path(project_name, row)
    test_src = row.get("test_source") or (get_source_content(project_name, test_path) if test_path else "")

    st.subheader("🧾 Test Source")

    test_path, test_path_note = resolve_test_path(project_name, row)
    test_src_full = row.get("test_source") or (get_source_content(project_name, test_path) if test_path else "")

    # st.caption(f"Resolved test_path: {test_path}  ({test_path_note})")

    if not test_src_full:
        st.warning("No test source content found.")
    else:
        method_src = extract_java_method(test_src_full, method_name)

        toggle_key = f"show_full_test_{reviewer_id}_{project_name}_{suite_basename}_{method_name}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = False

        # If we failed to extract method, fallback = show full suite directly
        if not method_src:
            st.warning(f"Could not extract method `{method_name}`. Showing full test suite.")
            st.session_state[toggle_key] = True

        # Default view: show current method only (if extracted)
        if method_src and not st.session_state[toggle_key]:
            st.caption(f"Showing current test case only: `{method_name}`")
            st.code(method_src, language="java")

        # Toggle button
        btn_label = "Show full test suite" if not st.session_state[toggle_key] else "Hide full test suite"
        if st.button(btn_label, key=f"btn_{toggle_key}"):
            st.session_state[toggle_key] = not st.session_state[toggle_key]

        # Full suite view (syntax-highlighted)
        if st.session_state[toggle_key]:
            st.caption("Full test suite (method is marked with comment banners)")
            marked_src = add_markers_around_method(test_src_full, method_src, method_name)
            st.code(marked_src, language="java")

    st.subheader("🧩 Your Assigned Instantiations in This Case")
    insts_all = get_instantiations_for_case(project_name, suite_basename, method_name)

    # gold: see all instantiations in sampled scope case
    if reviewer_id == GOLD_REVIEWER_ID:
        insts = insts_all
    else:
        insts = filter_insts_to_assigned(insts_all, reviewer_id)

    if not insts:
        st.info("No instantiations available in this case for you. Click Next Case.")
        st.stop()

    inst_ids = [x["inst_id"] for x in insts]
    ann_map = load_annotations_cv(inst_ids, reviewer_id)

    rules = load_rules()
    rule_options: List[Tuple[str, int]] = []
    label_to_id: Dict[str, int] = {}
    for r in rules:
        rid = int(r["id"])
        label = f"{rid:02d} [{r.get('type','')}] {r.get('criterion','')} ({r.get('mock_decision','')})"
        rule_options.append((label, rid))
        label_to_id[label] = rid

    case_inst_ui: List[Dict[str, Any]] = []

    for idx, inst in enumerate(insts):
        inst_id = inst["inst_id"]
        class_name = inst["class_name"]
        occurrence = safe_int(inst.get("occurrence_in_case"), 0)
        existing_mocked = inst.get("mocked", "")
        ui_suffix = f"{inst_id}_{idx}"

        prev = ann_map.get(inst_id, {})
        default_dec = prev.get("decision", "uncertain")
        default_note = prev.get("notes", "") or ""

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
                # f"**Existing Mocked:** `{existing_mocked}`"
            )
            st.caption(f"inst_id: {inst_id}")

            st.multiselect(
                "Applicable rules",
                options=[lab for (lab, _) in rule_options],
                default=default_labels,
                key=f"rules_{ui_suffix}",
            )

            with st.expander("➕ Add new rule"):
                new_rule_type = st.selectbox("Type", ["Base Rule", "Extended Rule", "Other"], key=f"new_type_{ui_suffix}")
                new_rule_cat = st.text_input("High Level Category", value="Class Characteristics", key=f"new_cat_{ui_suffix}")
                new_rule_criterion = st.text_area("Criterion", key=f"new_criterion_{ui_suffix}")
                new_rule_mock = st.text_input("Mock? (Yes/No/etc.)", value="Yes", key=f"new_mock_{ui_suffix}")

                if st.button("💾 Add rule & select it", key=f"add_rule_{ui_suffix}"):
                    if new_rule_criterion.strip():
                        new_id = insert_rule(
                            new_rule_type.strip(),
                            new_rule_cat.strip(),
                            new_rule_criterion.strip(),
                            new_rule_mock.strip(),
                            created_by=reviewer_id,
                        )
                        new_label = f"{new_id:02d} [{new_rule_type.strip()}] {new_rule_criterion.strip()} ({new_rule_mock.strip()})"
                        mk = f"rules_{ui_suffix}"
                        cur = st.session_state.get(mk, []) or []
                        if new_label not in cur:
                            st.session_state[mk] = cur + [new_label]
                        st.cache_data.clear()
                        st.success(f"Rule added: ID={new_id}")
                        st.rerun()
                    else:
                        st.warning("Criterion is required.")

            st.radio(
                "Decision",
                options=VALID_DECISIONS,
                horizontal=True,
                index=VALID_DECISIONS.index(default_dec) if default_dec in VALID_DECISIONS else 2,
                key=f"dec_{ui_suffix}",
            )
            st.text_area("Notes", value=default_note, height=80, key=f"note_{ui_suffix}")

            case_inst_ui.append(
                {
                    "inst_id": inst_id,
                    "rules_key": f"rules_{ui_suffix}",
                    "dec_key": f"dec_{ui_suffix}",
                    "note_key": f"note_{ui_suffix}",
                }
            )

    st.subheader("✅ Case Actions")

    labeled_case = sum(
        1 for iid in inst_ids
        if (ann_map.get(iid) and ann_map[iid].get("decision") != "uncertain")
    )
    st.progress(labeled_case / len(inst_ids) if inst_ids else 0.0)
    st.caption(f"This case (your assigned insts): labeled {labeled_case} / {len(inst_ids)}")

    b1, b2 = st.columns([1, 1])
    with b1:
        save_all = st.button("💾 Save ALL", key=f"save_all_{reviewer_id}_{project_name}_{suite_basename}_{method_name}_{st.session_state.case_idx}")
    with b2:
        save_next = st.button("💾 Save ALL & Next Case", key=f"save_next_{reviewer_id}_{project_name}_{suite_basename}_{method_name}_{st.session_state.case_idx}")

    def collect_recs() -> List[Dict[str, Any]]:
        recs = []
        for meta in case_inst_ui:
            selected_labels = st.session_state.get(meta["rules_key"], []) or []
            selected_rule_ids = [label_to_id.get(lab) for lab in selected_labels if lab in label_to_id]
            selected_rule_ids = [int(x) for x in selected_rule_ids if isinstance(x, int)]

            decision = st.session_state.get(meta["dec_key"], "uncertain")
            notes = st.session_state.get(meta["note_key"], "")

            recs.append(
                {
                    "inst_id": meta["inst_id"],
                    "decision": decision,
                    "notes": notes,
                    "rule_ids": selected_rule_ids,
                    "rule_labels": selected_labels,
                }
            )
        return recs

    if save_all or save_next:
        recs = collect_recs()
        for r in recs:
            upsert_annotation_cv(
                inst_id=r["inst_id"],
                reviewer_id=reviewer_id,
                decision=r["decision"],
                notes=r["notes"],
                rule_ids=r["rule_ids"],
                rule_labels=r["rule_labels"],
                updated_by=reviewer_id,
            )
        st.cache_data.clear()
        st.success(f"Saved {len(recs)} instantiation(s).")

        if save_next and st.session_state.case_idx < len(cases) - 1:
            st.session_state.case_idx += 1
        st.rerun()

    st.stop()


# ---------- Compare ----------
if mode == "Compare":
    st.subheader("📊 Compare (gold vs reviewer)")

    revs = db_fetchall("select distinct reviewer_id from assignments order by reviewer_id")
    revs = [r["reviewer_id"] for r in revs if r["reviewer_id"] != GOLD_REVIEWER_ID]

    sel = st.selectbox("Select reviewer", options=revs, index=0 if revs else None)
    if not sel:
        st.info("No reviewers found in assignments yet. Create assignments in Admin mode.")
        st.stop()

    df = compare_gold_vs_reviewer(sel)
    if df.empty:
        st.warning("No comparable rows found. Check that assignments exist and gold labels are imported.")
        st.stop()

    total = len(df)
    have_reviewer = int(df["reviewer_decision"].notna().sum())
    have_gold = int(df["gold_decision"].notna().sum())
    agree = int(df["agree"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total assigned insts", total)
    c2.metric("Reviewer labeled", have_reviewer)
    c3.metric("Gold available", have_gold)
    c4.metric("Agreements", agree)

    denom = max(have_reviewer, 1)
    st.caption(f"Agreement rate (among reviewer-labeled rows): {agree/denom:.3f}")

    st.subheader("Confusion matrix (gold vs reviewer)")
    cm = confusion_matrix(df)
    st.dataframe(cm, use_container_width=True)

    st.subheader("Disagreements")
    only_disagree = df[
        (df["reviewer_decision"].notna()) &
        (df["gold_decision"].notna()) &
        (df["agree"] == False)
    ].copy()
    st.caption(f"Disagreements (both labeled): {len(only_disagree)}")
    st.dataframe(only_disagree.head(500), use_container_width=True)

    st.subheader("Export")
    st.download_button(
        "⬇️ Download full comparison CSV",
        data=to_csv_download(df),
        file_name=f"compare_gold_vs_{sel}.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download disagreements CSV",
        data=to_csv_download(only_disagree),
        file_name=f"disagreements_gold_vs_{sel}.csv",
        mime="text/csv",
    )

    st.stop()
