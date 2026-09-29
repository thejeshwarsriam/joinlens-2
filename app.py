"""JoinLens - Streamlit UI. Run with:  streamlit run app.py"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from joinlens import analyzer, connectors, llm, report, samples

st.set_page_config(page_title="JoinLens", page_icon="🔍", layout="wide")

DIALECTS = {
    "PostgreSQL / Netezza": "postgres",
    "Databricks / Spark": "databricks",
    "Snowflake": "snowflake",
    "BigQuery": "bigquery",
    "Redshift": "redshift",
    "MySQL": "mysql",
    "SQL Server": "tsql",
    "Oracle": "oracle",
    "Trino / Presto": "trino",
    "DuckDB": "duckdb",
}


# ------------------------------------------------------------------ data loading
def load_uploads(files) -> tuple[dict[str, pd.DataFrame], list[str]]:
    tables, errors = {}, []
    for f in files:
        name = samples.sanitize_name(f.name.rsplit(".", 1)[0])
        try:
            if f.name.lower().endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f, sep=None, engine="python")
            tables[name] = df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.name}: {e}")
    return tables, errors


def flow_dot(a: analyzer.Analysis) -> str:
    def esc(t: str) -> str:
        return t.replace('"', "'")

    L = ['digraph G {', 'rankdir=LR;', 'bgcolor="transparent";',
         'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
         'edge [fontname="Helvetica", fontsize=10];',
         f'r0 [label="{esc(a.driving_source)}\\n{a.driving_rows:,} rows", fillcolor="#dbeafe"];']
    for s in a.steps:
        bad = s.fanout
        L.append(f't{s.index} [label="{esc(s.source)}\\n'
                 f'{s.right_stats.n_rows:,} rows"' if s.right_stats else
                 f't{s.index} [label="{esc(s.source)}"')
        L[-1] += ', fillcolor="#f1f5f9"];'
        fill = "#fecaca" if bad else "#bbf7d0"
        L.append(f'r{s.index + 1} [label="after join {s.index + 1}\\n{s.rows_after:,} rows '
                 f'({s.factor:.2f}x)", fillcolor="{fill}"];')
        color = "#dc2626" if bad else "#16a34a"
        L.append(f'r{s.index} -> r{s.index + 1} [label="{s.join_type}", color="{color}", '
                 f'penwidth={2.5 if bad else 1}];')
        key = esc(", ".join(s.right_keys)) if s.right_keys else "condition"
        L.append(f't{s.index} -> r{s.index + 1} [label="{key}\\n{esc(s.cardinality)}", '
                 f'color="{color}", style=dashed];')
    L.append("}")
    return "\n".join(L)



CONN_FIELDS = {
    "netezza": ["host", "port", "database", "schema", "user", "password"],
    "databricks": ["host", "http_path", "access_token", "database"],
    "oracle": ["host", "port", "database", "schema", "user", "password"],
    "sqlserver": ["host", "port", "database", "schema", "user", "password"],
}
FIELD_LABELS = {
    "host": "Host", "port": "Port", "database": "Database / service name",
    "schema": "Schema (optional)", "user": "User", "password": "Password",
    "http_path": "HTTP path (SQL warehouse or cluster)",
    "access_token": "Databricks personal access token (PAT)",
}


def live_connection_ui() -> dict:
    """Sidebar form for a live source. Credentials live only in st.session_state for this
    session - never written to disk. Nothing here is sent anywhere except directly to the
    host you enter, using the official driver library for that source."""
    st.sidebar.warning("Credentials are kept in memory for this session only and are never "
                       "saved to disk. Close the tab to clear them.")
    kind = st.sidebar.selectbox(
        "Source", list(connectors.KINDS), format_func=lambda k: connectors.LABEL_OF[k])
    if kind in ("oracle", "sqlserver"):
        st.sidebar.caption("Facets sits on top of this database - connect to the database "
                           "your Facets instance actually runs on.")
    cfg_kwargs = {"kind": kind}
    for field in CONN_FIELDS[kind]:
        is_secret = field in ("password", "access_token")
        val = st.sidebar.text_input(FIELD_LABELS[field], type="password" if is_secret else "default",
                                    key=f"live_{kind}_{field}")
        if field == "port" and val:
            val = int(val) if val.isdigit() else val
        cfg_kwargs[field] = val
    if kind == "sqlserver":
        cfg_kwargs["odbc_driver"] = st.sidebar.text_input(
            "ODBC driver name", value="ODBC Driver 18 for SQL Server", key="live_odbc_driver")

    if st.sidebar.button("Test connection"):
        try:
            conn = connectors.make_connector(connectors.SourceConfig(**cfg_kwargs))
            conn.test()
            st.session_state["live_conn_ok"] = True
            st.sidebar.success("Connected.")
        except connectors.ConnectorError as e:
            st.session_state["live_conn_ok"] = False
            st.sidebar.error(str(e))

    sample_n = st.sidebar.number_input(
        "Sample rows per table", min_value=1000, max_value=1_000_000, value=50_000, step=1000,
        help="Tables your query reads are sampled automatically when you click Analyze joins.")
    # samples are reused until the connection details or the sample size change
    cache_key = hashlib.md5(str((sorted(cfg_kwargs.items()), int(sample_n))).encode()).hexdigest()
    if st.session_state.get("live_cache_key") != cache_key:
        st.session_state["live_cache_key"] = cache_key
        st.session_state["live_tables"] = {}
        st.session_state["live_row_counts"] = {}
        st.session_state["live_fetch_errors"] = []
    st.session_state["live_cfg"] = cfg_kwargs
    st.session_state["live_sample_n"] = int(sample_n)

    with st.sidebar.expander("Fetch tables manually (optional)"):
        table_names = st.text_area(
            "Tables to pull (one per line, e.g. schema.orders)", key="live_table_names",
            help="Only needed if a table isn't picked up from the query automatically.")
        if st.button("Fetch sample") and table_names.strip():
            names = [t.strip() for t in table_names.splitlines() if t.strip()]
            with st.spinner("Pulling samples..."):
                for err in fetch_live_samples(names, force=True):
                    st.error(err)
    return st.session_state["live_tables"]


def live_ready() -> bool:
    cfg = st.session_state.get("live_cfg") or {}
    return bool(cfg.get("host"))


def fetch_live_samples(names: list[str], force: bool = False) -> list[str]:
    """Sample each live table not already cached into st.session_state["live_tables"].
    Returns one error message per table that couldn't be fetched."""
    cache = st.session_state["live_tables"]
    todo = [t for t in names if force or live_local_name(t) not in cache]
    if not todo:
        return []
    try:
        conn = connectors.make_connector(connectors.SourceConfig(**st.session_state["live_cfg"]))
    except connectors.ConnectorError as e:
        return [str(e)]
    errors = []
    try:
        for t in todo:
            try:
                df = conn.sample(t, n=st.session_state["live_sample_n"])
            except connectors.ConnectorError as e:
                errors.append(f"Could not fetch {t}: {e}")
                continue
            cache[live_local_name(t)] = df
            try:
                st.session_state["live_row_counts"][live_local_name(t)] = conn.row_count(t)
            except connectors.ConnectorError:
                pass
    finally:
        conn.close()
    return errors


def live_local_name(table: str) -> str:
    return ".".join(samples.sanitize_name(part) for part in table.split(".")[-3:])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔍 JoinLens")
st.sidebar.caption("See where a join fans out, why, and how to fix it.")
dialect_label = st.sidebar.selectbox("SQL dialect of your query", list(DIALECTS))
dialect = DIALECTS[dialect_label]
source = st.sidebar.radio("Data", ["Demo data", "Upload files", "Live connection"])

if source == "Demo data":
    tables = samples.make_demo_tables()
elif source == "Upload files":
    files = st.sidebar.file_uploader("CSV / TSV / Parquet (one file per table)",
                                     type=["csv", "tsv", "txt", "parquet"],
                                     accept_multiple_files=True)
    tables, errs = load_uploads(files or [])
    for e in errs:
        st.sidebar.error(e)
else:
    tables = live_connection_ui()

with st.sidebar.expander(f"Tables ({len(tables)})", expanded=source == "Upload files"):
    for name, df in tables.items():
        total = st.session_state.get("live_row_counts", {}).get(name) if source == "Live connection" else None
        st.markdown(f"**{name}** - {len(df):,} rows"
                    + (f" (sample of {total:,})" if total and total > len(df) else ""))
        st.caption(", ".join(map(str, df.columns)))

st.sidebar.info("All analysis runs locally in an in-memory DuckDB. Your data isn't sent anywhere.")

# ------------------------------------------------------------------ main
st.title("JoinLens")
st.write("Paste a query. JoinLens measures the row count after **every** join, finds the join "
         "that multiplies rows, explains the cause from your actual data, and shows fixes it has "
         "already verified.")

if source == "Demo data":
    preset = st.selectbox("Example query", list(samples.PRESET_QUERIES))
    default_sql, key = samples.PRESET_QUERIES[preset], f"sql_{preset}"
else:
    default_sql, key = "SELECT ...\nFROM a\nJOIN b ON a.id = b.a_id", "sql_upload"
sql = st.text_area("SQL (a single SELECT; CTEs are fine)", value=default_sql, height=230, key=key)

include_ctes = st.checkbox(
    "Also analyze joins inside CTEs", value=True,
    help="Each CTE body is measured on its own (with the CTEs defined before it). This is "
         "usually where row multiplication starts in long queries.")



def data_sig(tbls: dict[str, pd.DataFrame]) -> str:
    shape = str({k: (len(v), tuple(v.columns)) for k, v in tbls.items()})
    return hashlib.md5((sql + dialect + str(include_ctes) + shape).encode()).hexdigest()


is_live = source == "Live connection"
sig = data_sig(tables)
if st.button("Analyze joins", type="primary", disabled=not (tables or is_live and live_ready())):
    try:
        if is_live:
            names = analyzer.source_tables(sql, dialect)
            before = len(tables)
            with st.spinner(f"Sampling the {len(names)} table(s) your query reads..."):
                st.session_state["live_fetch_errors"] = fetch_live_samples(names)
            tables = st.session_state["live_tables"]
            if not tables:
                raise analyzer.AnalysisError("None of the tables in the query could be fetched.")
        with st.spinner("Profiling each join..."):
            con = analyzer.connect(tables)
            sig = data_sig(tables)
            st.session_state["analysis"] = (
                sig, analyzer.analyze_query(con, sql, dialect, include_ctes=include_ctes), sql)
        if is_live and len(tables) != before:
            st.rerun()  # refresh the sidebar table list with the new samples
    except analyzer.AnalysisError as e:
        st.session_state["analysis"] = None
        st.error(str(e))
if is_live:
    for err in st.session_state.get("live_fetch_errors", []):
        st.error(err)
if not tables and not is_live:
    st.warning("Upload at least one table (or switch to demo data) to begin.")
elif not tables:
    st.info("Enter the connection details in the sidebar. The tables your query reads are "
            "sampled automatically when you click Analyze joins.")


def show_analysis(a: analyzer.Analysis, part_sql: str, report_md: str) -> None:
    fans = a.fanout_steps

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Driving rows", f"{a.driving_rows:,}")
    c2.metric("Rows after all joins", f"{a.final_rows:,}",
              f"{a.overall_factor:.2f}x", delta_color="inverse" if fans else "off")
    c3.metric("Joins", len(a.steps))
    c4.metric("Joins that multiply rows", len(fans), delta_color="off")

    if fans:
        names = ", ".join(f"#{s.index + 1} ({s.source})" for s in fans)
        st.error(f"Fan-out detected at join {names}. Aggregates over this result will be inflated.")
    else:
        st.success("No join multiplies rows - every join keeps or reduces the row count.")

    if source == "Live connection" and st.session_state.get("live_cfg"):
        if st.button("Validate final row count against full live data (not the sample)"):
            try:
                conn = connectors.make_connector(connectors.SourceConfig(**st.session_state["live_cfg"]))
                with st.spinner("Running the full query on the live source..."):
                    true_rows = conn.query_row_count(part_sql)
                conn.close()
                st.info(f"Full live data: **{true_rows:,} rows** after all joins "
                       f"(sample-based estimate was {a.final_rows:,}).")
            except connectors.ConnectorError as e:
                st.error(str(e))

    t_flow, t_steps, t_agg, t_ai, t_rep = st.tabs(
        ["Flow", "Join by join", "Aggregate impact", "AI insights", "Report"])

    with t_flow:
        st.graphviz_chart(flow_dot(a), use_container_width=True)
        counts = pd.DataFrame({"rows": [a.driving_rows] + [s.rows_after for s in a.steps]},
                              index=["FROM"] + [f"join {s.index + 1}: {s.alias}" for s in a.steps])
        st.bar_chart(counts)

    with t_steps:
        for s in a.steps:
            icon = "🔴" if s.fanout else "🟢"
            title = (f"{icon} Join {s.index + 1}: {s.join_type} {s.source} - "
                     f"{s.rows_before:,} → {s.rows_after:,} rows ({s.factor:.2f}x)")
            with st.expander(title, expanded=s.fanout):
                st.code(f"{s.join_type} JOIN {s.source}\n  ON {s.on_sql}", language="sql")
                m1, m2, m3 = st.columns(3)
                m1.metric("Cardinality", s.cardinality)
                if s.right_stats:
                    m2.metric("Rows per key (worst)", f"{s.right_stats.max_dup:,}")
                    m3.metric("Repeated keys", f"{s.right_stats.dup_keys:,}")
                st.markdown(s.explanation)
                for w in s.warnings:
                    st.warning(w)
                if s.right_stats is not None and not s.right_stats.top_dups.empty:
                    st.markdown("**Most repeated join keys**")
                    st.dataframe(s.right_stats.top_dups, hide_index=True)
                if s.sample is not None:
                    st.markdown("**Rows produced for the most repeated key** "
                                "(the same left row appears once per match)")
                    st.dataframe(s.sample, hide_index=True)
                if s.fixes:
                    st.markdown("#### Fixes (verified against your data)")
                    for f in s.fixes:
                        st.markdown(f"**{f.title}**")
                        st.caption(f.why)
                        st.code(f.sql, language="sql")
                        if f.verified_rows is not None:
                            ok = f.verified_factor is not None and f.verified_factor <= 1.0 + 1e-9
                            (st.success if ok else st.warning)(
                                f"After this fix: {f.verified_rows:,} rows "
                                f"({f.verified_factor:.2f}x of the rows entering the join)")

    with t_agg:
        if not a.agg_risks:
            st.info("No aggregate in the SELECT list is affected.")
        else:
            st.caption("Aggregates whose value changes because rows are repeated. "
                       "Values are computed without WHERE / GROUP BY.")
            df = pd.DataFrame([{
                "expression": r.expression, "joins that inflate it": ", ".join(map(str, r.affected_joins)),
                "driving table alone": r.base_value, "after all joins": r.joined_value,
                "change %": None if r.inflation_pct is None else round(r.inflation_pct, 1),
                "why": r.note} for r in a.agg_risks])
            st.dataframe(df, hide_index=True)

    with t_ai:
        st.caption("Sends only the computed facts above (row counts, cardinality, verified "
                  "fixes) to a Databricks Model Serving endpoint in your own workspace. Raw "
                  "table rows are never sent. Requires an endpoint your team has already "
                  "deployed - this doesn't create one.")
        c1, c2 = st.columns(2)
        ws = c1.text_input("Databricks workspace URL", key="llm_ws",
                           placeholder="https://adb-....azuredatabricks.net")
        ep = c1.text_input("Serving endpoint name", key="llm_ep",
                           placeholder="databricks-meta-llama-3-3-70b-instruct")
        pat = c2.text_input("Databricks PAT", type="password", key="llm_pat")
        question = c2.text_input("Optional question for the model", key="llm_q",
                                 placeholder="e.g. which fix should I apply first?")
        if st.button("Generate insight", disabled=not (ws and ep and pat)):
            cfg = llm.DatabricksLLMConfig(workspace_url=ws, endpoint_name=ep, token=pat)
            try:
                with st.spinner("Asking the Databricks endpoint..."):
                    text = llm.explain(cfg, a, question=question)
                st.markdown(text)
            except llm.LLMError as e:
                st.error(str(e))
        with st.expander("Facts that would be sent"):
            st.json(llm.facts_from_analysis(a))

    with t_rep:
        st.download_button("Download report (.md)", report_md, "joinlens_report.md", "text/markdown")
        st.markdown(report_md)


state = st.session_state.get("analysis")
if state and state[0] == sig:
    qa: analyzer.QueryAnalysis = state[1]
    parts = qa.analyzed

    if len(qa.parts) > 1:
        st.subheader("Query sections")
        st.dataframe(pd.DataFrame([{
            "section": p.label,
            "joins": len(p.analysis.steps) if p.analysis else None,
            "joins that multiply rows": len(p.analysis.fanout_steps) if p.analysis else None,
            "rows in → out": (f"{p.analysis.driving_rows:,} → {p.analysis.final_rows:,}"
                              if p.analysis else "-"),
            "note": p.error.split("\n")[0]} for p in qa.parts]), hide_index=True)

    def part_label(i: int) -> str:
        p = parts[i]
        n = len(p.analysis.fanout_steps)
        return f"{'🔴' if n else '🟢'} {p.label}" + (f" - {n} fan-out{'s' if n > 1 else ''}" if n else "")

    if len(parts) > 1:
        default = next((i for i, p in enumerate(parts) if p.analysis.fanout_steps), 0)
        part = parts[st.selectbox("Section to inspect", range(len(parts)), index=default,
                                  format_func=part_label, key=f"part_{sig}")]
    else:
        part = parts[0]
    if len(qa.parts) > 1:
        with st.expander(f"SQL analyzed for: {part.label}"):
            st.code(part.sql, language="sql")

    show_analysis(part.analysis, part.sql, report.to_markdown_query(qa, state[2]))
