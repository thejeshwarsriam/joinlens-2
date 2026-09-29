"""Grounded LLM insights via a Databricks Model Serving endpoint.

Only the structured FACTS already computed by joinlens.analyzer - exact row counts,
cardinality, verified fix results - are sent to the endpoint. Raw table rows never leave
the analyzer. The call goes directly from this app to your own Databricks workspace using
your personal access token; nothing passes through Anthropic or any other third party.

This calls the OpenAI-compatible chat endpoint every Databricks serving endpoint exposes
at /serving-endpoints/<name>/invocations. The endpoint (a foundation model like Llama or
DBRX, or a model your team has deployed) must already exist in your workspace - this
module doesn't create one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from .analyzer import Analysis

TIMEOUT_S = 60

SYSTEM_PROMPT = (
    "You are a data-quality assistant for a business analyst. You are given a JSON object "
    "of facts already computed by a deterministic SQL join analyzer: exact row counts, "
    "join cardinality, and fixes that have already been re-run and verified against the "
    "data. Write a short, plain-English explanation: what happened, why it happened, and "
    "which fix to apply and why. Only use the numbers and facts given in the JSON - never "
    "invent a row count, a percentage, or a cause that isn't in the data. If nothing in the "
    "JSON shows a fan-out, say plainly that the query looks safe. Keep it under 200 words."
)


class LLMError(Exception):
    """The Databricks endpoint could not be reached, or returned something unexpected."""


@dataclass
class DatabricksLLMConfig:
    workspace_url: str      # e.g. https://adb-1234567890123456.7.azuredatabricks.net
    endpoint_name: str      # serving endpoint name, e.g. databricks-meta-llama-3-3-70b-instruct
    token: str              # Databricks personal access token (PAT)
    max_tokens: int = 600
    temperature: float = 0.1


def facts_from_analysis(a: Analysis) -> dict:
    """The only data sent to the LLM: computed facts, never raw rows."""
    return {
        "driving_source": a.driving_source,
        "driving_rows": a.driving_rows,
        "final_rows": a.final_rows,
        "overall_factor": round(a.overall_factor, 3) if a.driving_rows else None,
        "joins": [
            {
                "step": s.index + 1,
                "type": s.join_type,
                "table": s.source,
                "rows_before": s.rows_before,
                "rows_after": s.rows_after,
                "factor": round(s.factor, 3),
                "cardinality": s.cardinality,
                "fanout": s.fanout,
                "join_keys": s.right_keys,
                "max_repeats_of_a_key": s.right_stats.max_dup if s.right_stats else None,
                "top_fix": s.fixes[0].title if s.fixes else None,
                "top_fix_verified_rows": s.fixes[0].verified_rows if s.fixes else None,
            }
            for s in a.steps
        ],
        "aggregates_at_risk": [
            {"expression": r.expression, "kind": r.kind,
             "inflation_pct": None if r.inflation_pct is None else round(r.inflation_pct, 1),
             "note": r.note}
            for r in a.agg_risks
        ],
    }


def _endpoint_url(cfg: DatabricksLLMConfig) -> str:
    return f"{cfg.workspace_url.rstrip('/')}/serving-endpoints/{cfg.endpoint_name}/invocations"


def test_connection(cfg: DatabricksLLMConfig) -> None:
    """Cheap call to confirm the workspace URL, endpoint name and PAT all work."""
    _chat(cfg, "Reply with the single word: ok")


def explain(cfg: DatabricksLLMConfig, analysis: Analysis, question: str = "") -> str:
    """Ask the endpoint to explain an Analysis in plain English, grounded in its facts."""
    facts = facts_from_analysis(analysis)
    content = "Facts:\n" + json.dumps(facts, indent=2)
    if question:
        content += f"\n\nSpecific question from the analyst: {question}"
    return _chat(cfg, content, system=SYSTEM_PROMPT)


def _chat(cfg: DatabricksLLMConfig, user_content: str, system: str | None = None) -> str:
    if not (cfg.workspace_url and cfg.endpoint_name and cfg.token):
        raise LLMError("Workspace URL, endpoint name, and PAT are all required.")
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user_content}]
    payload = {"messages": messages, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature}
    try:
        resp = requests.post(
            _endpoint_url(cfg),
            headers={"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"},
            json=payload, timeout=TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise LLMError(f"Could not reach the Databricks endpoint: {e}") from e
    if resp.status_code == 401:
        raise LLMError("Databricks rejected the token (401) - check the PAT is valid and unexpired.")
    if resp.status_code == 404:
        raise LLMError(f"No serving endpoint named '{cfg.endpoint_name}' was found (404).")
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise LLMError(f"Databricks endpoint returned an error: {e}") from e
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape from the endpoint: {data}") from e
