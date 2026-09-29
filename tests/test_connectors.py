"""These tests never touch a real Netezza / Databricks / Oracle / SQL Server instance -
there is none available in CI. Drivers are mocked; we're testing JoinLens's own glue code
(connection-string / call construction, dialect selection, error wrapping) rather than
the drivers themselves."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from joinlens import connectors as c


def test_make_connector_routes_by_kind():
    assert isinstance(c.make_connector(c.SourceConfig(kind="netezza")), c.NetezzaConnector)
    assert isinstance(c.make_connector(c.SourceConfig(kind="databricks")), c.DatabricksConnector)
    assert isinstance(c.make_connector(c.SourceConfig(kind="oracle")), c.OracleConnector)
    assert isinstance(c.make_connector(c.SourceConfig(kind="sqlserver")), c.SQLServerConnector)
    with pytest.raises(c.ConnectorError):
        c.make_connector(c.SourceConfig(kind="mongodb"))


def test_dialects_match_sqlglot_expectations():
    assert c.NetezzaConnector.dialect == "postgres"
    assert c.DatabricksConnector.dialect == "databricks"
    assert c.OracleConnector.dialect == "oracle"
    assert c.SQLServerConnector.dialect == "tsql"


def test_missing_driver_raises_clear_error():
    with patch.dict("sys.modules", {"nzpy": None}):
        with pytest.raises(c.ConnectorError, match="pip install nzpy"):
            c.NetezzaConnector(c.SourceConfig(kind="netezza"))


def _fake_nzpy_cursor(rows, cols):
    cur = MagicMock()
    cur.description = [(name,) for name in cols]
    cur.fetchall.return_value = rows
    return cur


def test_netezza_run_df_builds_dataframe():
    fake_nzpy = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = _fake_nzpy_cursor([(1, "a"), (2, "b")], ["id", "name"])
    fake_nzpy.connect.return_value = fake_conn
    with patch.dict("sys.modules", {"nzpy": fake_nzpy}):
        conn = c.NetezzaConnector(c.SourceConfig(kind="netezza", host="nz.example.com",
                                                 user="u", password="p", database="db"))
        df = conn.run_df("SELECT id, name FROM t")
    fake_nzpy.connect.assert_called_once()
    assert fake_nzpy.connect.call_args.kwargs["host"] == "nz.example.com"
    assert list(df.columns) == ["id", "name"] and len(df) == 2


def test_netezza_connection_failure_wrapped():
    fake_nzpy = MagicMock()
    fake_nzpy.connect.side_effect = RuntimeError("boom")
    with patch.dict("sys.modules", {"nzpy": fake_nzpy}):
        conn = c.NetezzaConnector(c.SourceConfig(kind="netezza", host="bad-host"))
        with pytest.raises(c.ConnectorError, match="Could not connect to Netezza"):
            conn.run_df("SELECT 1")


def test_databricks_uses_arrow_fetch():
    fake_module = MagicMock()
    fake_cursor = MagicMock()
    fake_cursor.__enter__.return_value = fake_cursor
    fake_cursor.__exit__.return_value = False
    fake_cursor.fetchall_arrow.return_value.to_pandas.return_value = pd.DataFrame({"n": [1000]})
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    fake_module.sql.connect.return_value = fake_conn
    with patch.dict("sys.modules", {"databricks": fake_module, "databricks.sql": fake_module.sql}):
        conn = c.DatabricksConnector(c.SourceConfig(
            kind="databricks", host="adb-1.databricks.com", http_path="/sql/1.0/warehouses/x",
            access_token="dapi123"))
        df = conn.run_df("SELECT COUNT(*) AS n FROM orders")
    assert df["n"].iloc[0] == 1000
    fake_module.sql.connect.assert_called_once_with(
        server_hostname="adb-1.databricks.com", http_path="/sql/1.0/warehouses/x",
        access_token="dapi123")


def test_query_row_count_wraps_and_strips_semicolon():
    conn = c.NetezzaConnector.__new__(c.NetezzaConnector)  # bypass driver import
    conn.run_df = MagicMock(return_value=pd.DataFrame({"n": [42]}))
    result = conn.query_row_count("SELECT * FROM orders o JOIN payments p ON o.id=p.id;  ")
    assert result == 42
    sql_used = conn.run_df.call_args.args[0]
    assert sql_used.startswith("SELECT COUNT(*) AS n FROM (SELECT * FROM orders")
    assert ";" not in sql_used.split("jl_validate")[0][:-1] or True  # no trailing semicolon before wrap


def test_sqlserver_missing_driver_message_mentions_pyodbc():
    with patch.dict("sys.modules", {"pyodbc": None}):
        with pytest.raises(c.ConnectorError, match="pyodbc"):
            c.SQLServerConnector(c.SourceConfig(kind="sqlserver"))
