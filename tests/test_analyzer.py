import pandas as pd
import pytest

from joinlens import analyzer, report, samples


@pytest.fixture(scope="module")
def con():
    return analyzer.connect(samples.make_demo_tables())


def run(con, name):
    return analyzer.analyze(con, samples.PRESET_QUERIES[name], "postgres")


def test_detects_two_fanouts_and_counts_rows(con):
    a = run(con, "Revenue by region (fans out twice)")
    assert a.driving_rows == 1000
    assert [s.fanout for s in a.steps] == [False, True, True]
    assert a.final_rows == a.steps[-1].rows_after > a.driving_rows
    assert a.steps[1].cardinality == "one-to-many"
    assert a.steps[1].right_stats.max_dup == 3


def test_fixes_restore_row_count(con):
    a = run(con, "Revenue by region (fans out twice)")
    pre = next(f for f in a.steps[1].fixes if f.title.startswith("Pre-aggregate"))
    assert pre.verified_rows == a.steps[1].rows_before
    assert "SUM(paid_amount)" in pre.sql.replace('"', "")


def test_missing_grain_column_is_found(con):
    a = run(con, "Wrong grain: join on customer_id (many-to-many)")
    s = a.steps[0]
    assert s.cardinality == "many-to-many"
    grain = next(f for f in s.fixes if f.title.startswith("Join on the full grain"))
    assert "order_id" in grain.title
    assert grain.verified_rows == 800  # exactly one shipment per shipped order


def test_aggregate_inflation_is_quantified(con):
    a = run(con, "Wrong grain: join on customer_id (many-to-many)")
    total = next(r for r in a.agg_risks if r.kind == "sum")
    assert total.inflation_pct > 100


def test_clean_query_has_no_fanout(con):
    a = run(con, "Clean query (no fan-out)")
    assert not a.fanout_steps and not a.agg_risks


def test_cross_join_and_errors(con):
    a = analyzer.analyze(con, "SELECT 1 FROM customers c CROSS JOIN shipments s", "postgres")
    assert a.steps[0].fanout and a.steps[0].join_type == "CROSS"
    with pytest.raises(analyzer.AnalysisError):
        analyzer.analyze(con, "SELECT * FROM orders", "postgres")
    with pytest.raises(analyzer.AnalysisError):
        analyzer.analyze(con, "SELECT * FROM nope n JOIN orders o ON n.id = o.id", "postgres")


def test_cte_and_subquery_sources(con):
    sql = """WITH p AS (SELECT order_id, paid_amount FROM payments)
             SELECT o.order_id FROM orders o JOIN p ON o.order_id = p.order_id"""
    a = analyzer.analyze(con, sql, "postgres")
    assert a.steps[0].fanout


def test_external_file_access_is_blocked():
    c = analyzer.connect({"t": pd.DataFrame({"id": [1]})})
    with pytest.raises(Exception):
        c.execute("SELECT * FROM read_csv('/etc/passwd')").fetchall()


def test_qualified_live_table_names_are_queryable():
    c = analyzer.connect({
        "sales.orders": pd.DataFrame({"id": [1], "customer_id": [10]}),
        "sales.customers": pd.DataFrame({"id": [10], "name": ["Ada"]}),
    })
    a = analyzer.analyze(
        c,
        "SELECT o.id FROM sales.orders o JOIN sales.customers c "
        "ON o.customer_id = c.id",
        "postgres",
    )
    assert a.driving_rows == 1
    assert a.final_rows == 1


def test_report_renders(con):
    a = run(con, "Revenue by region (fans out twice)")
    md = report.to_markdown(a, "SELECT 1")
    assert "Fix: Pre-aggregate" in md and "Aggregates at risk" in md


UNION_SQL = """
WITH po AS (
  SELECT o.order_id, o.customer_id, o.order_date, p.paid_amount
  FROM orders o JOIN payments p ON o.order_id = p.order_id
), plain AS (SELECT * FROM customers)
SELECT c.region, SUM(po.paid_amount) AS paid
FROM po JOIN plain c ON po.customer_id = c.customer_id
GROUP BY 1
UNION ALL
SELECT c.region, SUM(i.qty)
FROM orders o JOIN order_items i ON o.order_id = i.order_id
  JOIN customers c ON c.customer_id = o.customer_id
GROUP BY 1
"""


def test_union_branches_and_ctes_are_analyzed_separately(con):
    qa = analyzer.analyze_query(con, UNION_SQL, "postgres")
    labels = [p.label for p in qa.parts]
    assert labels == ["Main query - UNION branch 1 of 2", "Main query - UNION branch 2 of 2",
                      "CTE po"]  # `plain` has no joins, so it is skipped
    b1, b2, cte = (p.analysis for p in qa.parts)
    assert not b1.fanout_steps
    assert [s.alias for s in b2.fanout_steps] == ["i"]
    assert [s.alias for s in cte.fanout_steps] == ["p"]
    assert qa.parts[1].sql.startswith("WITH po AS")  # each part runs standalone


def test_ctes_can_be_skipped(con):
    qa = analyzer.analyze_query(con, UNION_SQL, "postgres", include_ctes=False)
    assert [p.label for p in qa.parts] == ["Main query - UNION branch 1 of 2",
                                            "Main query - UNION branch 2 of 2"]


def test_single_select_keeps_original_sql(con):
    sql = samples.PRESET_QUERIES["Revenue by region (fans out twice)"]
    qa = analyzer.analyze_query(con, sql, "postgres")
    assert qa.parts[0].label == "Main query" and qa.parts[0].sql == sql.strip()
    assert len(qa.parts[0].analysis.fanout_steps) == 2


def test_analyze_query_errors(con):
    with pytest.raises(analyzer.AnalysisError, match="No JOINs"):
        analyzer.analyze_query(con, "SELECT * FROM orders UNION ALL SELECT * FROM orders", "postgres")
    with pytest.raises(analyzer.AnalysisError, match="DDL"):
        analyzer.analyze_query(con, "DELETE FROM orders", "postgres")


def test_netezza_next_month_is_emulated_in_duckdb():
    c = analyzer.connect({
        "terms": pd.DataFrame({"id": [1, 2], "term_dt": pd.to_datetime(["2024-01-31", "2024-12-15"])}),
        "enrolls": pd.DataFrame({"id": [1, 2], "eff_dt": pd.to_datetime(["2024-02-01", "2025-01-01"])}),
    })
    sql = ("SELECT t.id FROM terms t JOIN enrolls e "
           "ON e.id = t.id AND e.eff_dt = NEXT_MONTH(t.term_dt)")
    a = analyzer.analyze(c, sql, "postgres")
    assert a.final_rows == 2
    assert "NEXT_MONTH" in a.steps[0].on_sql  # shown to the user in their own dialect


def test_query_report_has_a_section_per_part(con):
    qa = analyzer.analyze_query(con, UNION_SQL, "postgres")
    md = report.to_markdown_query(qa, UNION_SQL)
    assert "## Sections" in md and "## CTE po" in md and "## Main query - UNION branch 2 of 2" in md


def test_source_tables_skips_ctes_and_dedupes():
    sql = """WITH pm AS (SELECT * FROM db.admin.product_map)
             SELECT * FROM pm JOIN DB.ADMIN.PRODUCT_MAP x ON pm.id = x.id
             LEFT JOIN (SELECT * FROM cust.admin.stg) b ON b.id = x.id
             UNION ALL SELECT * FROM orders o JOIN pm ON o.id = pm.id"""
    assert sorted(t.lower() for t in analyzer.source_tables(sql)) == [
        "cust.admin.stg", "db.admin.product_map", "orders"]
