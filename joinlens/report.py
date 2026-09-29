"""Markdown export of an Analysis."""
from __future__ import annotations

from .analyzer import Analysis, QueryAnalysis


def to_markdown(a: Analysis, sql: str) -> str:
    out = ["# JoinLens report", "", "## Query", "", "```sql", sql.strip(), "```", ""]
    return "\n".join(out + _analysis_md(a, "##"))


def to_markdown_query(qa: QueryAnalysis, sql: str) -> str:
    """Report for every analyzed section (UNION branches, CTEs) of one query."""
    if len(qa.parts) == 1 and qa.parts[0].analysis is not None:
        return to_markdown(qa.parts[0].analysis, sql)
    out = ["# JoinLens report", "", "## Query", "", "```sql", sql.strip(), "```", "",
           "## Sections", "",
           "| Section | Joins | Joins that multiply rows | Rows in -> out |",
           "|---|---:|---:|---|"]
    for p in qa.parts:
        a = p.analysis
        out.append(f"| {p.label} | {len(a.steps)} | {len(a.fanout_steps)} | "
                   f"{a.driving_rows:,} -> {a.final_rows:,} |" if a else
                   f"| {p.label} | - | - | {p.error.splitlines()[0] if p.error else '-'} |")
    out.append("")
    for p in qa.analyzed:
        out += [f"## {p.label}", ""] + _analysis_md(p.analysis, "###")
    return "\n".join(out)


def _analysis_md(a: Analysis, h: str) -> list[str]:
    sub = h + "#"
    out = [
        f"{h} Summary", "",
        f"- Driving table: `{a.driving_source}` ({a.driving_rows:,} rows)",
        f"- Rows after all joins: **{a.final_rows:,}** ({a.overall_factor:.2f}x)",
        f"- Joins that multiply rows: **{len(a.fanout_steps)}** of {len(a.steps)}", "",
        "| # | Join | Rows before | Rows after | Factor | Cardinality |",
        "|---|------|------------:|-----------:|-------:|-------------|",
    ]
    for s in a.steps:
        flag = " (fan-out)" if s.fanout else ""
        out.append(f"| {s.index + 1} | {s.join_type} {s.source} | {s.rows_before:,} | "
                   f"{s.rows_after:,} | {s.factor:.2f}x{flag} | {s.cardinality} |")
    out.append("")
    for s in a.fanout_steps:
        out += [f"{h} Join {s.index + 1}: {s.join_type} JOIN {s.source}", "",
                f"`ON {s.on_sql}`", "", s.explanation, ""]
        for f in s.fixes:
            out += [f"{sub} Fix: {f.title}", "", f.why, "", "```sql", f.sql, "```"]
            if f.verified_rows is not None:
                out.append(f"\nVerified: {f.verified_rows:,} rows after this join "
                           f"({f.verified_factor:.2f}x).")
            out.append("")
    if a.agg_risks:
        out += [f"{h} Aggregates at risk", "",
                "| Expression | Joins that inflate it | Driving-table value | After joins | Change |",
                "|---|---|---:|---:|---:|"]
        for r in a.agg_risks:
            base = f"{r.base_value:,.2f}" if r.base_value is not None else "-"
            joined = f"{r.joined_value:,.2f}" if r.joined_value is not None else "-"
            chg = f"{r.inflation_pct:+.1f}%" if r.inflation_pct is not None else "-"
            out.append(f"| `{r.expression}` | {', '.join(map(str, r.affected_joins))} | "
                       f"{base} | {joined} | {chg} |")
        out.append("")
    return out
