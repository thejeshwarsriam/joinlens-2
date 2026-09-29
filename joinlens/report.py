"""Markdown export of an Analysis."""
from __future__ import annotations

from .analyzer import Analysis


def to_markdown(a: Analysis, sql: str) -> str:
    out = ["# JoinLens report", "", "## Query", "", "```sql", sql.strip(), "```", ""]
    out += [
        "## Summary", "",
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
        out += [f"## Join {s.index + 1}: {s.join_type} JOIN {s.source}", "",
                f"`ON {s.on_sql}`", "", s.explanation, ""]
        for f in s.fixes:
            out += [f"### Fix: {f.title}", "", f.why, "", "```sql", f.sql, "```"]
            if f.verified_rows is not None:
                out.append(f"\nVerified: {f.verified_rows:,} rows after this join "
                           f"({f.verified_factor:.2f}x).")
            out.append("")
    if a.agg_risks:
        out += ["## Aggregates at risk", "",
                "| Expression | Joins that inflate it | Driving-table value | After joins | Change |",
                "|---|---|---:|---:|---:|"]
        for r in a.agg_risks:
            base = f"{r.base_value:,.2f}" if r.base_value is not None else "-"
            joined = f"{r.joined_value:,.2f}" if r.joined_value is not None else "-"
            chg = f"{r.inflation_pct:+.1f}%" if r.inflation_pct is not None else "-"
            out.append(f"| `{r.expression}` | {', '.join(map(str, r.affected_joins))} | "
                       f"{base} | {joined} | {chg} |")
        out.append("")
    return "\n".join(out)
