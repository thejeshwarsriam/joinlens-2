# JoinLens 🔍

**See exactly where a SQL join fans out, why it happens, and how to fix it.**

Row multiplication ("fan-out") is the silent killer of dashboard numbers: a join on a
non-unique key repeats rows, and every `SUM` / `COUNT` above it is inflated. JoinLens
takes a query plus your data, measures the row count after **every** join, and tells you:

- **Where** - which join multiplies rows (`1,000 → 1,568 → 4,811`), drawn as a flow diagram
- **Why** - key uniqueness on both sides, worst repeated keys, exact-duplicate rows,
  one-to-many vs many-to-many, compounding fan-outs
- **What it breaks** - which aggregates are inflated, quantified (`SUM(o.amount) +381%`)
- **How to fix it** - ready-to-paste SQL in *your* dialect, each fix **re-run against your
  data** to prove the row count is restored:
  - join on the full grain (finds the missing key column automatically)
  - drop the join / replace with `EXISTS` when nothing is selected from it
  - pre-aggregate the many-side to one row per key
  - keep one row per key (`ROW_NUMBER`)
  - remove exact duplicate rows

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the demo data, pick an example query, click **Analyze joins**. To use your own data,
choose *Upload files* and drop one CSV / TSV / Parquet per table (file name = table name).

## How it works

1. `sqlglot` parses your query (PostgreSQL/Netezza, Databricks, Snowflake, BigQuery, ...).
2. For each join step it rebuilds `COUNT(*)` and profiling queries from the AST (WHERE /
   GROUP BY are stripped: fan-out is measured before filtering) and transpiles them to DuckDB.
3. Tables are DataFrames registered in an in-memory DuckDB. **Nothing leaves your machine**, and
   `enable_external_access` is switched off so analysed SQL cannot read local files.

Your original SQL text is never executed directly - only the generated queries are.

```
app.py                 Streamlit UI
joinlens/analyzer.py   engine (no Streamlit dependency - reusable as a library / CLI)
joinlens/report.py     Markdown export
joinlens/samples.py    demo tables + example queries
tests/                 pytest suite (engine + headless UI smoke test)
```

Use the engine on its own:

```python
from joinlens import analyzer
con = analyzer.connect({"orders": orders_df, "payments": payments_df})
result = analyzer.analyze(con, "SELECT ... FROM orders o JOIN payments p ON ...", "postgres")
for step in result.fanout_steps:
    print(step.explanation, [f.title for f in step.fixes])
```

## Live connections (Netezza, Databricks, Facets)

Install only the drivers you need:

```bash
pip install -r requirements-live.txt
```

Pick **Live connection** in the sidebar, choose a source, fill in the fields, click
**Test connection**, then **Fetch sample**. JoinLens pulls a row-count-accurate *sample*
of each table you list (default 50,000 rows) into a local, in-memory DuckDB and runs the
same join engine on it as on uploaded files - nothing about the analysis logic changes.
Once you have a result, a **"Validate final row count against full live data"** button
re-runs your exact query's row count directly on the source (no sampling) so you can
confirm the sample-based numbers hold on the whole table before trusting them.

| Source | Fields needed | Notes |
|---|---|---|
| Netezza | host, port (default 5480), database, schema, user, password | Uses `nzpy`, pure Python |
| Databricks | host (workspace hostname), HTTP path, access token, database | HTTP path and token come from **SQL Warehouses → Connection details** in your workspace; the warehouse/cluster must be running |
| Facets on Oracle | host, port (default 1521), service name, schema, user, password | Facets is the claims application - you're connecting to whichever RDBMS actually hosts it |
| Facets on SQL Server | host, port (default 1433), database, schema, user, password, ODBC driver name | Needs the Microsoft ODBC Driver for SQL Server installed on the machine running JoinLens, separately from `pip install pyodbc` |

**Security notes**
- Credentials are kept only in Streamlit's `st.session_state` for the running session and
  are never written to disk, logged, or included in the exported report.
- This has **not** been reviewed for a shared or internet-facing deployment. Run it
  locally, or put real secrets handling (env vars / a secrets manager, per-user auth) in
  front of it before sharing.
- For PHI-adjacent data: only row counts and small aggregate numbers are computed by the
  "validate against live data" pushdown query. The **sample** step does pull real rows
  locally to power the detailed diagnostics (duplicate-key samples, fix verification) -
  keep sample sizes and who can run this in mind accordingly.

## AI insights (Databricks Model Serving)

In the **AI insights** tab, enter your Databricks workspace URL, an existing serving
endpoint name (a foundation model like Llama, or one your team deployed), and a personal
access token (PAT). JoinLens sends only the already-computed facts - row counts,
cardinality, verified fix results - as JSON; **no raw table rows are ever sent**. The call
goes straight from your machine to your own Databricks workspace over the standard
OpenAI-compatible `/serving-endpoints/<name>/invocations` API - nothing passes through
Anthropic or any other third party. The "Facts that would be sent" expander shows the
exact payload before you send anything.

This needs a serving endpoint your workspace already has (Databricks doesn't let you spin
one up through this app).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Limitations (v0.1)

- Top-level `SELECT` with joins (CTEs and subqueries as join sources work; `UNION`, DDL, DML don't)
- The detailed join diagnostics (cardinality, duplicate-key samples, verified fixes) run
  against a **sample**, not the full live table - the "validate against live data" button
  checks the final row count only, not the intermediate per-join breakdown, against full data
- Fix templates guess aggregates (numeric → `SUM`, else `MAX`) - review against your business rule
- Only equality join keys are profiled automatically; range / `OR` joins still get measured
  row counts but no key analysis
- Live connectors are built to each driver's documented API but **have not been tested
  against a real Netezza, Databricks, or Facets instance** - only with mocked drivers.
  Expect to debug connection-string / auth quirks on first real use, especially for
  SQL Server's ODBC driver naming
- No row-loss detection yet (a LEFT JOIN silently turned INNER by a WHERE clause), no
  mutation testing, no EXPLAIN-plan ingestion, no business-glossary/grain layer

## Roadmap

- Full pushdown profiling (per-join counts run directly on the warehouse, not a local sample)
- Row-loss detection, NULL-key blindness, type-coercion and skew detection
- Mutation testing (systematically swap LEFT/INNER, drop ON conditions, show the row-count delta)
- Business glossary + grain registry, so the AI insights tab can explain results in the
  organization's own terms instead of raw column names
- dbt model mode: analyze every join in a project's compiled SQL
- CLI + GitHub Action that fails a PR when a join newly fans out
- Tableau data-source join analysis (parse `.twb` relationships / joins)
