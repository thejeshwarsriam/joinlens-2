from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_and_analyzes_every_preset():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    presets = at.selectbox[1].options  # [dialect, example query]
    for p in presets:
        at.selectbox[1].select(p).run()
        at.button[0].click().run()
        assert not at.exception, p
        assert not at.error or "Fan-out detected" in at.error[0].value, p
    assert at.tabs  # results rendered
    assert "AI insights" in [t.label for t in at.tabs]


def test_live_connection_source_renders_without_crashing():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Live connection").run()
    assert not at.exception
    from joinlens import connectors
    assert at.sidebar.selectbox[1].options == [connectors.LABEL_OF[k] for k in connectors.KINDS]
    at.sidebar.button[0].click().run()  # "Test connection" with empty fields
    assert not at.exception  # should surface a ConnectorError in the UI, not crash


def test_union_query_shows_one_section_per_part():
    from tests.test_analyzer import UNION_SQL
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value(UNION_SQL).run()
    at.button[0].click().run()
    assert not at.exception
    section = next(s for s in at.selectbox if s.key and s.key.startswith("part_"))
    assert len(section.options) == 3
    assert "UNION branch 2 of 2" in section.options[section.index]  # first part with a fan-out
    assert any("Fan-out detected" in e.value for e in at.error)


def test_live_source_fetches_query_tables_automatically(monkeypatch):
    from joinlens import connectors, samples
    demo = samples.make_demo_tables()
    sampled: list[str] = []

    class FakeConnector:
        def sample(self, table, n=50_000):
            sampled.append(table)
            return demo[table.split(".")[-1]].head(n)

        def row_count(self, table):
            return len(demo[table.split(".")[-1]]) * 10

        def close(self):
            pass

    monkeypatch.setattr(connectors, "make_connector", lambda cfg: FakeConnector())
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Live connection").run()
    analyze = next(b for b in at.button if b.label == "Analyze joins")
    assert analyze.disabled  # no connection details yet
    at.sidebar.text_input[0].set_value("warehouse.example.com").run()
    at.text_area[0].set_value(
        "SELECT o.order_id FROM sales.orders o JOIN sales.payments p ON o.order_id = p.order_id").run()
    next(b for b in at.button if b.label == "Analyze joins").click().run()
    assert not at.exception
    assert sorted(sampled) == ["sales.orders", "sales.payments"]
    assert any("Fan-out detected" in e.value for e in at.error)
    assert any("(sample of" in m.value for m in at.sidebar.markdown)

    next(b for b in at.button if b.label == "Analyze joins").click().run()
    assert len(sampled) == 2  # cached samples are reused
