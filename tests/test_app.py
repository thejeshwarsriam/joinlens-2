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
