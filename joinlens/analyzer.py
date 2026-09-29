"""JoinLens engine: measure every JOIN step of a SELECT, explain fan-out, propose fixes.

The engine never runs the user's SQL text directly. It parses it with sqlglot, rebuilds
small COUNT / profiling queries from the AST, transpiles them to DuckDB and runs those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

DUCK = "duckdb"
SAFE_AGGS = {"min", "max", "anyvalue"}
ORDER_HINTS = ("updated", "modified", "created", "date", "time", "_ts", "version", "seq")


class AnalysisError(Exception):
    """Input the analyzer cannot handle (bad SQL, no joins, failing query...)."""


# --------------------------------------------------------------------------- models
@dataclass
class KeyStats:
    key_cols: list[str]
    n_rows: int
    n_keys: int
    max_dup: int
    dup_keys: int
    null_rows: int
    exact_dup_rows: int
    top_dups: pd.DataFrame


@dataclass
class Fix:
    title: str
    why: str
    sql: str
    verified_rows: Optional[int] = None
    verified_factor: Optional[float] = None


@dataclass
class JoinStep:
    index: int
    join_type: str
    source: str
    alias: str
    on_sql: str
    rows_before: int
    rows_after: int
    factor: float
    fanout: bool
    equi: bool
    cardinality: str
    left_keys: list[str] = field(default_factory=list)
    right_keys: list[str] = field(default_factory=list)
    right_stats: Optional[KeyStats] = None
    left_max_dup: Optional[int] = None
    explanation: str = ""
    sample: Optional[pd.DataFrame] = None
    fixes: list[Fix] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AggRisk:
    expression: str
    kind: str
    columns: list[str]
    affected_joins: list[int]
    base_value: Optional[float] = None
    joined_value: Optional[float] = None
    inflation_pct: Optional[float] = None
    note: str = ""


@dataclass
class Analysis:
    dialect: str
    driving_source: str
    driving_rows: int
    final_rows: int
    steps: list[JoinStep]
    agg_risks: list[AggRisk]
    warnings: list[str] = field(default_factory=list)

    @property
    def fanout_steps(self) -> list[JoinStep]:
        return [s for s in self.steps if s.fanout]

    @property
    def overall_factor(self) -> float:
        return self.final_rows / self.driving_rows if self.driving_rows else float("nan")


# --------------------------------------------------------------------------- helpers
def connect(tables: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the given DataFrames registered as tables."""
    con = duckdb.connect(":memory:")
    attached_catalogs: set[str] = set()
    for index, (name, df) in enumerate(tables.items()):
        parts = name.split(".")
        if len(parts) in (2, 3):
            if len(parts) == 3 and parts[0] not in attached_catalogs:
                con.execute(f"ATTACH ':memory:' AS {_ident(parts[0])}")
                attached_catalogs.add(parts[0])
            stage = f"__joinlens_sample_{index}"
            con.register(stage, df)
            if len(parts) == 2:
                con.execute(f"CREATE SCHEMA IF NOT EXISTS {_ident(parts[0])}")
            else:
                con.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {_ident(parts[0])}.{_ident(parts[1])}")
            target = ".".join(_ident(part) for part in parts)
            con.execute(f"CREATE VIEW {target} AS SELECT * FROM {_ident(stage)}")
        else:
            con.register(name, df)
    try:  # analysed SQL must never touch the local file system
        con.execute("SET enable_external_access = false")
    except duckdb.Error:
        pass
    return con


def _arg(node: exp.Expression, *names: str):
    """Read an AST arg under any of the names used by different sqlglot versions."""
    for n in names:
        v = node.args.get(n)
        if v is not None:
            return v
    return None


def _ident(name: str) -> str:
    return exp.to_identifier(name).sql(dialect=DUCK)


def _run(con, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).df()
    except duckdb.Error as e:
        raise AnalysisError(f"DuckDB could not run a generated query: {e}\n\n{sql}") from e


def _count(con, sql: str) -> int:
    return int(_run(con, sql).iloc[0, 0])


def _with_prefix(with_sql: str, ctes: str) -> str:
    return f"{with_sql}, {ctes}" if with_sql else f"WITH {ctes}"


def _dedupe_cols(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        n = seen.get(c, 0)
        cols.append(c if n == 0 else f"{c}_{n}")
        seen[c] = n + 1
    df = df.copy()
    df.columns = cols
    return df


def _count_star() -> exp.Expression:
    return exp.Count(this=exp.Star())


def _strip(select: exp.Select) -> exp.Select:
    """Copy of the SELECT without filters / grouping (fan-out is measured pre-filter)."""
    q = select.copy()
    for key in ("where", "group", "having", "order", "limit", "offset", "qualify",
                "distinct", "windows"):
        q.set(key, None)
    return q


def _step_query(select: exp.Select, k: int, projections: list) -> exp.Select:
    q = _strip(select)
    joins = [j.copy() for j in (select.args.get("joins") or [])[:k]]
    q.set("joins", joins or None)
    q.set("expressions", projections)
    return q


def _source_sql(node: exp.Expression) -> Optional[str]:
    if isinstance(node, exp.Table):
        t = node.copy()
        t.set("alias", None)
        return f"SELECT * FROM {t.sql(dialect=DUCK)}"
    if isinstance(node, exp.Subquery):
        return node.this.sql(dialect=DUCK)
    return None


def _from_aliased(node: exp.Expression, alias: str) -> str:
    if isinstance(node, exp.Table):
        t = node.copy()
        t.set("alias", None)
        return f"{t.sql(dialect=DUCK)} AS {alias}"
    return f"({node.this.sql(dialect=DUCK)}) AS {alias}"


def _label(node: exp.Expression) -> str:
    if isinstance(node, exp.Table):
        return node.name + (f" {node.alias}" if node.alias else "")
    return f"(subquery) {node.alias}".strip()


def _cols_of(con, with_sql: str, src_sql: str) -> pd.DataFrame:
    sql = _with_prefix(with_sql, f"src AS ({src_sql})") + " SELECT * FROM src LIMIT 0"
    return con.execute(sql).df()


def _key_stats(con, with_sql: str, src_sql: str, keys: list[str]) -> KeyStats:
    ks = ", ".join(_ident(k) for k in keys)
    nulls = " OR ".join(f"{_ident(k)} IS NULL" for k in keys)
    ctes = f"src AS ({src_sql}), g AS (SELECT {ks}, COUNT(*) AS c FROM src GROUP BY {ks})"
    head = _with_prefix(with_sql, ctes)
    row = _run(
        con,
        head + f""" SELECT (SELECT COUNT(*) FROM src) AS n_rows,
                           (SELECT COUNT(*) FROM g) AS n_keys,
                           COALESCE((SELECT MAX(c) FROM g), 0) AS max_dup,
                           (SELECT COUNT(*) FROM g WHERE c > 1) AS dup_keys,
                           (SELECT COUNT(*) FROM src WHERE {nulls}) AS null_rows""",
    ).iloc[0]
    try:
        distinct_rows = _count(con, _with_prefix(with_sql, f"src AS ({src_sql})")
                               + " SELECT COUNT(*) FROM (SELECT DISTINCT * FROM src) d")
        exact_dups = int(row["n_rows"]) - distinct_rows
    except AnalysisError:
        exact_dups = 0
    top = _run(con, head + f" SELECT {ks}, c AS rows_per_key FROM g WHERE c > 1 "
                           "ORDER BY c DESC LIMIT 10")
    return KeyStats(
        key_cols=keys, n_rows=int(row["n_rows"]), n_keys=int(row["n_keys"]),
        max_dup=int(row["max_dup"]), dup_keys=int(row["dup_keys"]),
        null_rows=int(row["null_rows"]), exact_dup_rows=exact_dups, top_dups=top,
    )


def _parse_on(on, using, alias: str):
    """Split a join condition into (left_cols, right_cols, only_equalities)."""
    if using:
        names = [u.name for u in using]
        return [exp.column(n) for n in names], [exp.column(n, table=alias) for n in names], True
    if on is None:
        return [], [], False
    conds = list(on.flatten()) if isinstance(on, exp.And) else [on]
    left, right, only_eq = [], [], True
    for c in conds:
        if isinstance(c, exp.Paren):
            c = c.this
        if isinstance(c, exp.EQ) and isinstance(c.this, exp.Column) and isinstance(c.expression, exp.Column):
            a, b = c.this, c.expression
            if b.table.lower() == alias.lower() and a.table.lower() != alias.lower():
                left.append(a); right.append(b); continue
            if a.table.lower() == alias.lower() and b.table.lower() != alias.lower():
                left.append(b); right.append(a); continue
        only_eq = False
    return left, right, only_eq


def _pick_order_col(cols: list[str], keys: list[str]) -> Optional[str]:
    rest = [c for c in cols if c.lower() not in {k.lower() for k in keys}]
    for hint in ORDER_HINTS:
        for c in rest:
            if hint in c.lower():
                return c
    return rest[0] if rest else None


def _verify(con, select, k, replace=None, where=None) -> Optional[int]:
    try:
        q = _strip(select)
        joins = [j.copy() for j in (select.args.get("joins") or [])[: k + 1]]
        if where is not None:
            joins = joins[:k]
        elif replace is not None:
            joins[k] = replace
        q.set("joins", joins or None)
        q.set("expressions", [_count_star()])
        if where is not None:
            q = q.where(where)
        return _count(con, q.sql(dialect=DUCK))
    except Exception:  # noqa: BLE001 - verification is best-effort
        return None


def _subquery(sql: str, alias: str) -> exp.Subquery:
    return exp.Subquery(
        this=sqlglot.parse_one(sql, read=DUCK),
        alias=exp.TableAlias(this=exp.to_identifier(alias)),
    )


def _out(node: exp.Expression, dialect: str) -> str:
    try:
        return node.sql(dialect=dialect, pretty=True)
    except Exception:  # noqa: BLE001
        return node.sql(dialect=DUCK, pretty=True)


# --------------------------------------------------------------------------- fixes
def _build_fixes(con, select, k, join, alias, src_node, with_sql, right_df, stats,
                 left_alias_hint, alias_cols, rows_before, dialect, is_left_or_inner) -> list[Fix]:
    fixes: list[Fix] = []
    keys = stats.key_cols
    sel = ", ".join(_ident(x) for x in keys)
    src_alias = _from_aliased(src_node, "s")
    right_cols = list(right_df.columns)
    lower_map = {c.lower(): c for c in right_cols}

    def add(title, why, new_join=None, where=None, snippet=None):
        rows = _verify(con, select, k, replace=new_join, where=where)
        fixes.append(Fix(
            title=title, why=why,
            sql=snippet if snippet is not None else _out(new_join, dialect),
            verified_rows=rows,
            verified_factor=(rows / rows_before) if rows is not None and rows_before else None,
        ))

    def swap(sub_sql, on=None):
        nj = join.copy()
        nj.set("this", _subquery(sub_sql, alias))
        if on is not None:
            nj.set("on", on)
        return nj

    # columns of this table used anywhere except in this join's own ON clause
    used: set[str] = set()
    uses_star = any(isinstance(e, exp.Star) for e in select.expressions)
    for c in select.find_all(exp.Column):
        if c.table.lower() == alias.lower() and c.find_ancestor(exp.Join) is not join:
            used.add(c.name)
    used.discard("*")
    if any(isinstance(c.this, exp.Star) and c.table.lower() == alias.lower()
           for c in select.find_all(exp.Column)):
        uses_star = True

    # 1. exact duplicate rows
    if stats.exact_dup_rows > 0:
        add("Remove exact duplicate rows",
            f"{stats.exact_dup_rows:,} rows in this table are complete copies of another row.",
            new_join=swap(f"SELECT DISTINCT * FROM {src_alias}"))

    # 1b. add the missing grain column(s) to the ON clause
    if join.args.get("on") is not None:
        for col in _grain_candidates(con, with_sql, src_node, keys, right_cols, alias_cols,
                                     alias, stats):
            l_alias = left_alias_hint(col)
            cond = exp.and_(join.args["on"].copy(),
                            exp.EQ(this=exp.column(col, table=l_alias),
                                   expression=exp.column(col, table=alias)))
            nj = join.copy()
            nj.set("on", cond)
            add(f"Join on the full grain: also match `{col}`",
                f"({', '.join(keys)}, {col}) is unique in `{alias}`, and `{l_alias}` has the same "
                f"column - the original condition was missing part of the table's grain.",
                new_join=nj)

    # 2. join not needed / semi-join
    if not used and not uses_star and is_left_or_inner is not None:
        if is_left_or_inner == "LEFT":
            fixes.append(Fix(
                "Drop the join",
                f"No column of `{alias}` is used outside its ON clause, and a LEFT JOIN never "
                "removes rows - so this join can only multiply rows. Remove it.",
                "-- delete the whole JOIN ... ON ... clause for this table",
                verified_rows=rows_before, verified_factor=1.0))
        else:
            on_sql = join.args["on"].sql(dialect=DUCK) if join.args.get("on") else "TRUE"
            ex = sqlglot.parse_one(
                f"EXISTS (SELECT 1 FROM {_from_aliased(src_node, alias)} WHERE {on_sql})", read=DUCK)
            add("Use EXISTS instead of an INNER JOIN",
                f"Nothing from `{alias}` is selected - the join only filters. EXISTS keeps one "
                "row per left row.", where=ex,
                snippet="WHERE " + _out(ex, dialect))

    # 3. pre-aggregate to one row per key
    measures = sorted(used - {x.lower() for x in keys} - set(keys))
    aggs = []
    for m in measures:
        col = lower_map.get(m.lower(), m)
        fn = "SUM" if col in right_df and pd.api.types.is_numeric_dtype(right_df[col].dtype) else "MAX"
        aggs.append(f"{fn}({_ident(col)}) AS {_ident(col)}")
    aggs.append("COUNT(*) AS _row_count")
    add("Pre-aggregate to one row per join key",
        "Collapse the many-side to the join grain before joining. Numeric columns are summed and "
        "others take MAX - adjust to your business rule.",
        new_join=swap(f"SELECT {sel}, {', '.join(aggs)} FROM {src_alias} GROUP BY {sel}"))

    # 5. keep one row per key
    oc = _pick_order_col(right_cols, keys)
    over = f"PARTITION BY {sel}" + (f" ORDER BY {_ident(oc)} DESC" if oc else "")
    cols_list = ", ".join(_ident(c) for c in right_cols)
    add("Keep one row per key (latest wins)",
        (f"Uses ROW_NUMBER ordered by `{oc}` DESC - confirm this is the row you want."
         if oc else "No obvious ordering column - the surviving row is arbitrary; add an ORDER BY."),
        new_join=swap(f"SELECT {cols_list} FROM (SELECT s.*, ROW_NUMBER() OVER ({over}) AS _rn "
                      f"FROM {src_alias}) AS d WHERE _rn = 1"))
    return fixes


def _grain_candidates(con, with_sql, src_node, keys, right_cols, alias_cols, alias, stats):
    out = []
    src_sql = _source_sql(src_node)
    ks = ", ".join(_ident(x) for x in keys)
    left_names = {c.lower() for a, cols in alias_cols.items() if a != alias.lower() for c in cols}
    for col in right_cols:
        if col.lower() in {x.lower() for x in keys} or col.lower() not in left_names:
            continue
        try:
            n = _count(con, _with_prefix(with_sql, f"src AS ({src_sql})")
                       + f" SELECT COUNT(*) FROM (SELECT DISTINCT {ks}, {_ident(col)} FROM src) d")
        except AnalysisError:
            continue
        if n == stats.n_rows:
            out.append(col)
        if len(out) >= 3:
            break
    return out


# --------------------------------------------------------------------------- main
def analyze(con: duckdb.DuckDBPyConnection, sql: str, dialect: str = "postgres") -> Analysis:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError as e:
        raise AnalysisError(f"Could not parse the SQL as {dialect}: {e}") from e
    if not isinstance(tree, exp.Select):
        raise AnalysisError("Paste a single SELECT statement (CTEs are fine). "
                            "UNION, DDL and DML are not supported.")
    joins = list(tree.args.get("joins") or [])
    if not joins:
        raise AnalysisError("No JOINs found in the top-level SELECT.")
    from_node = _arg(tree, "from", "from_")
    if from_node is None:
        raise AnalysisError("The SELECT has no FROM clause.")
    driving = from_node.this
    with_node = _arg(tree, "with", "with_")
    with_sql = with_node.sql(dialect=DUCK) if with_node is not None else ""

    d_alias = driving.alias_or_name or "t0"
    warnings: list[str] = []

    # columns per alias (used to spot missing grain columns)
    alias_cols: dict[str, list[str]] = {}
    for a, node in [(d_alias, driving)] + [(j.this.alias_or_name or f"t{i+1}", j.this)
                                          for i, j in enumerate(joins)]:
        s = _source_sql(node)
        if s:
            try:
                alias_cols[a.lower()] = list(_cols_of(con, with_sql, s).columns)
            except duckdb.Error:
                pass

    driving_rows = _count(con, _step_query(tree, 0, [_count_star()]).sql(dialect=DUCK))
    prev_rows = driving_rows
    steps: list[JoinStep] = []
    aliases_before = [d_alias]

    for i, j in enumerate(joins):
        node = j.this
        alias = node.alias_or_name or f"t{i+1}"
        kind = (j.args.get("kind") or "").upper()
        side = (j.args.get("side") or "").upper()
        natural = (j.args.get("method") or "").upper() == "NATURAL"
        on, using = j.args.get("on"), j.args.get("using")
        is_cross = kind == "CROSS" or (on is None and not using and not natural)
        jtype = "CROSS" if is_cross else (side or kind or "INNER")

        rows_after = _count(con, _step_query(tree, i + 1, [_count_star()]).sql(dialect=DUCK))
        factor = rows_after / prev_rows if prev_rows else float("nan")
        fanout = is_cross and prev_rows > 0 or (factor > 1 + 1e-9)

        left_cols, right_cols, only_eq = ([], [], False) if is_cross or natural else \
            _parse_on(on, using, alias)
        step = JoinStep(
            index=i, join_type=jtype, source=_label(node), alias=alias,
            on_sql=(on.sql(dialect=dialect) if on is not None
                    else (f"USING ({', '.join(u.name for u in using)})" if using else "")),
            rows_before=prev_rows, rows_after=rows_after, factor=factor, fanout=bool(fanout),
            equi=bool(right_cols) and only_eq, cardinality="unknown",
            left_keys=[c.sql(dialect=dialect) for c in left_cols],
            right_keys=[c.name for c in right_cols],
        )

        src_sql = _source_sql(node)
        if right_cols and src_sql:
            try:
                stats = _key_stats(con, with_sql, src_sql, step.right_keys)
                step.right_stats = stats
                prev_q = _step_query(tree, i, [exp.alias_(c.copy(), f"k{n}")
                                               for n, c in enumerate(left_cols)])
                lk = ", ".join(f"k{n}" for n in range(len(left_cols)))
                try:
                    step.left_max_dup = int(_count(
                        con, f"SELECT COALESCE(MAX(c), 0) FROM (SELECT {lk}, COUNT(*) AS c "
                             f"FROM ({prev_q.sql(dialect=DUCK)}) x GROUP BY {lk}) y"))
                except AnalysisError:
                    step.warnings.append("Could not measure key repetition on the left side.")
                right_unique = stats.max_dup <= 1
                left_unique = step.left_max_dup is not None and step.left_max_dup <= 1
                if step.left_max_dup is None:
                    step.cardinality = "one-to-one/many-to-one" if right_unique else "one-to-many or many-to-many"
                elif right_unique:
                    step.cardinality = "one-to-one" if left_unique else "many-to-one"
                else:
                    step.cardinality = "one-to-many" if left_unique else "many-to-many"
            except AnalysisError as e:
                step.warnings.append(str(e).split("\n")[0])
        prior = next((x.index + 1 for x in steps if x.fanout), None)
        step.explanation = _explain(step, is_cross, natural, prior)

        if step.fanout and step.right_stats and step.right_stats.max_dup > 1:
            try:
                def left_alias_hint(col, _lc=left_cols):
                    for c in _lc:
                        if c.table and col.lower() in {x.lower() for x in alias_cols.get(c.table.lower(), [])}:
                            return c.table
                    for a in aliases_before:
                        if col.lower() in {x.lower() for x in alias_cols.get(a.lower(), [])}:
                            return a
                    return aliases_before[0]

                right_df = _cols_of(con, with_sql, src_sql)
                step.fixes = _build_fixes(
                    con, tree, i, j, alias, node, with_sql, right_df, step.right_stats,
                    left_alias_hint, alias_cols, prev_rows, dialect,
                    side if side in ("LEFT",) else ("INNER" if not side else None))
                step.sample = _sample(con, tree, i, left_cols, step.right_stats)
            except Exception as e:  # noqa: BLE001
                step.warnings.append(f"Could not generate fixes: {e}")

        steps.append(step)
        aliases_before.append(alias)
        prev_rows = rows_after

    agg = _agg_risks(con, tree, steps, d_alias, driving, aliases_before)
    return Analysis(dialect=dialect, driving_source=_label(driving), driving_rows=driving_rows,
                    final_rows=prev_rows, steps=steps, agg_risks=agg, warnings=warnings)


def _explain(s: JoinStep, is_cross: bool, natural: bool, prior_fan: Optional[int] = None) -> str:
    if is_cross:
        return (f"There is no join condition, so every one of the {s.rows_before:,} incoming rows "
                f"is paired with every row of `{s.alias}` - a Cartesian product "
                f"({s.factor:,.1f}x).")
    if natural:
        return "NATURAL JOIN matches on every same-named column; key uniqueness can't be profiled."
    if not s.right_stats:
        return ("The join condition isn't a plain column equality, so key uniqueness can't be "
                "profiled. The row change is still measured directly."
                if not s.equi else "")
    st, keys = s.right_stats, ", ".join(s.right_keys)
    if not s.fanout:
        if st.max_dup <= 1:
            return f"`{s.source}` is unique on ({keys}), so each left row matches at most one row - no fan-out."
        return (f"`{s.source}` repeats ({keys}) but the rows didn't grow "
                f"({s.rows_before:,} -> {s.rows_after:,}); matches are filtered out elsewhere.")
    if st.max_dup <= 1:
        return ("`" + s.source + f"` is unique on ({keys}), yet rows grew {s.factor:.2f}x. "
                "Look at the other predicates in the ON clause (ranges, OR conditions).")
    text = (f"`{s.source}` is **not unique** on ({keys}): {st.n_rows:,} rows share only "
            f"{st.n_keys:,} distinct key values; {st.dup_keys:,} keys repeat and the worst one "
            f"appears {st.max_dup:,}x. Each matching left row is copied once per matching "
            f"`{s.source}` row, so the result grew {s.factor:.2f}x "
            f"({s.rows_before:,} -> {s.rows_after:,}).")
    if s.cardinality == "many-to-many":
        if prior_fan:
            text += (f" The left side already repeats this key (up to {s.left_max_dup:,}x) because "
                     f"join {prior_fan} multiplied it earlier, so the fan-outs compound. "
                     "Fix the earlier join first or aggregate each many-side separately.")
        else:
            text += (f" The left side repeats this key too (up to {s.left_max_dup:,}x), so matches "
                     "multiply against each other - a many-to-many join, usually a sign the join "
                     "key is at the wrong grain.")
    if st.exact_dup_rows:
        text += f" {st.exact_dup_rows:,} of those rows are exact duplicates."
    if st.null_rows:
        text += f" ({st.null_rows:,} rows have a NULL key and never match.)"
    return text


def _sample(con, select, k, left_cols, stats) -> Optional[pd.DataFrame]:
    if stats.top_dups.empty or not left_cols:
        return None
    row = stats.top_dups.iloc[0]
    conds = []
    for n, c in enumerate(left_cols):
        val = row[stats.key_cols[n]]
        val = val.item() if hasattr(val, "item") else val
        conds.append(exp.EQ(this=c.copy(), expression=exp.convert(val)))
    q = _step_query(select, k + 1, [exp.Star()]).where(exp.and_(*conds)).limit(8)
    return _dedupe_cols(_run(con, q.sql(dialect=DUCK)))


def _duplicated_by(table_pos: int, s: JoinStep) -> bool:
    """Are rows of the table at `table_pos` repeated by the fan-out at step `s`?

    Position 0 is the FROM table, join i sits at position i + 1.
    Tables before the join are copied per match; tables joined later match the already
    repeated rows; the joined table itself is only repeated in a many-to-many join.
    """
    own = s.index + 1
    if table_pos == own:
        return s.cardinality == "many-to-many"
    return True


def _agg_risks(con, select, steps, d_alias, driving, aliases) -> list[AggRisk]:
    fans = [s for s in steps if s.fanout]
    if not fans:
        return []
    pos = {a.lower(): n for n, a in enumerate(aliases)}
    risks: list[AggRisk] = []
    for proj in select.expressions:
        for agg in proj.find_all(exp.AggFunc):
            key = agg.key.lower()
            if key in SAFE_AGGS or (key == "count" and agg.find(exp.Distinct)):
                continue
            cols = list(agg.find_all(exp.Column))
            tabs = {c.table.lower() for c in cols if c.table}
            unqualified = any(not c.table for c in cols)
            affected = [s.index + 1 for s in fans
                        if not tabs or unqualified
                        or any(_duplicated_by(pos.get(t, 0), s) for t in tabs)]
            if not affected:
                continue
            risk = AggRisk(expression=agg.sql(dialect=DUCK), kind=key,
                           columns=[c.sql(dialect=DUCK) for c in cols], affected_joins=affected)
            only_driving = not cols or (tabs <= {d_alias.lower()} and not unqualified)
            if only_driving and key in ("sum", "count", "avg"):
                try:
                    base = _step_query(select, 0, [agg.copy()])
                    full = _step_query(select, len(select.args["joins"]), [agg.copy()])
                    b = _run(con, base.sql(dialect=DUCK)).iloc[0, 0]
                    f = _run(con, full.sql(dialect=DUCK)).iloc[0, 0]
                    risk.base_value, risk.joined_value = float(b), float(f)
                    if b:
                        risk.inflation_pct = (float(f) / float(b) - 1) * 100
                except Exception:  # noqa: BLE001
                    pass
            risk.note = ("Counts fanned-out rows." if key == "count" and not cols else
                         "Repeated rows are summed more than once." if key == "sum" else
                         "Average is re-weighted by the repetition." if key == "avg" else
                         "Affected by repeated rows.")
            risks.append(risk)
    return risks
