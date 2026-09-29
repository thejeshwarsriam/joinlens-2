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


def test_report_renders(con):
    a = run(con, "Revenue by region (fans out twice)")
    md = report.to_markdown(a, "SELECT 1")
    assert "Fix: Pre-aggregate" in md and "Aggregates at risk" in md
