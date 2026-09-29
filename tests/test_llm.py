import json
from unittest.mock import MagicMock, patch

import pytest

from joinlens import analyzer, llm, samples


@pytest.fixture(scope="module")
def analysis():
    con = analyzer.connect(samples.make_demo_tables())
    return analyzer.analyze(con, samples.PRESET_QUERIES["Revenue by region (fans out twice)"],
                            "postgres")


def cfg(**kw):
    base = dict(workspace_url="https://adb-1.databricks.net", endpoint_name="my-llm",
               token="dapi-secret-token")
    base.update(kw)
    return llm.DatabricksLLMConfig(**base)


def test_facts_contain_no_raw_rows_only_computed_numbers(analysis):
    facts = llm.facts_from_analysis(analysis)
    blob = json.dumps(facts)
    assert "133.92" not in blob  # a real payment amount from the sample data
    assert facts["driving_rows"] == 1000
    assert any(j["fanout"] for j in facts["joins"])
    assert facts["aggregates_at_risk"]


def test_explain_posts_facts_and_returns_content(analysis):
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"choices": [{"message": {"content": "Looks fanned out."}}]}
    fake_resp.raise_for_status.return_value = None
    with patch("joinlens.llm.requests.post", return_value=fake_resp) as post:
        result = llm.explain(cfg(), analysis, question="is this safe?")
    assert result == "Looks fanned out."
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "https://adb-1.databricks.net/serving-endpoints/my-llm/invocations"
    assert kwargs["headers"]["Authorization"] == "Bearer dapi-secret-token"
    sent = json.loads(json.dumps(kwargs["json"]))  # round-trip, just to inspect shape
    assert sent["messages"][0]["role"] == "system"
    assert "is this safe?" in sent["messages"][1]["content"]


def test_missing_config_fields_raise_without_network(analysis):
    with pytest.raises(llm.LLMError, match="required"):
        llm.explain(llm.DatabricksLLMConfig("", "", ""), analysis)


def test_401_gives_actionable_message(analysis):
    fake_resp = MagicMock(status_code=401)
    with patch("joinlens.llm.requests.post", return_value=fake_resp):
        with pytest.raises(llm.LLMError, match="rejected the token"):
            llm.explain(cfg(), analysis)


def test_404_names_the_missing_endpoint(analysis):
    fake_resp = MagicMock(status_code=404)
    with patch("joinlens.llm.requests.post", return_value=fake_resp):
        with pytest.raises(llm.LLMError, match="my-llm"):
            llm.explain(cfg(), analysis)


def test_network_failure_wrapped(analysis):
    import requests
    with patch("joinlens.llm.requests.post", side_effect=requests.ConnectionError("dns fail")):
        with pytest.raises(llm.LLMError, match="Could not reach"):
            llm.explain(cfg(), analysis)


def test_malformed_response_shape(analysis):
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"unexpected": "shape"}
    fake_resp.raise_for_status.return_value = None
    with patch("joinlens.llm.requests.post", return_value=fake_resp):
        with pytest.raises(llm.LLMError, match="Unexpected response shape"):
            llm.explain(cfg(), analysis)
