"""Live warehouse connectors for JoinLens: Netezza, Databricks, and Facets' underlying
database (Oracle or SQL Server - Facets itself is an application, not a SQL engine).

Every connector exposes the same small surface so the rest of JoinLens never needs to
know which engine it's talking to:

    .dialect                sqlglot dialect string used to generate pushdown SQL
    .test()                 raises ConnectorError if the credentials/host don't work
    .list_tables()          best-effort list of "schema.table" names
    .row_count(table)       exact COUNT(*), pushed down - no rows leave the warehouse
    .sample(table, n)       LIMIT/TOP n rows pulled locally for the join engine
    .run_df(sql)            run arbitrary SQL, get a DataFrame back
    .close()

Drivers (nzpy, databricks-sql-connector, oracledb, pyodbc) are imported lazily, so
installing JoinLens doesn't require every client library - only the ones you connect to.

SECURITY
--------
Nothing in this module writes credentials to disk, a log file, or a config file. Keep
`SourceConfig` values in Streamlit's `st.session_state` for a local, single-user run, or
in environment variables / a secrets manager if this is ever shared. This module has not
been reviewed for a multi-tenant or internet-facing deployment - don't run it that way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

KINDS = ("netezza", "databricks", "oracle", "sqlserver")
DIALECT_OF = {"netezza": "postgres", "databricks": "databricks", "oracle": "oracle",
             "sqlserver": "tsql"}
LABEL_OF = {"netezza": "Netezza", "databricks": "Databricks",
           "oracle": "Facets on Oracle", "sqlserver": "Facets on SQL Server"}


class ConnectorError(Exception):
    """Could not connect, or a query against the live source failed."""


@dataclass
class SourceConfig:
    kind: str                      # one of KINDS
    host: str = ""
    port: Optional[int] = None
    database: str = ""             # Netezza database / Oracle service name / SQL Server DB
    schema: str = ""
    user: str = ""
    password: str = ""
    http_path: str = ""            # Databricks only - SQL warehouse or cluster HTTP path
    access_token: str = ""         # Databricks PAT
    odbc_driver: str = "ODBC Driver 18 for SQL Server"  # SQL Server only

    def label(self) -> str:
        return LABEL_OF.get(self.kind, self.kind)


class BaseConnector:
    dialect = "postgres"

    def test(self) -> None:
        self.run_df("SELECT 1 AS ok")

    def list_tables(self) -> list[str]:
        raise NotImplementedError

    def row_count(self, table: str) -> int:
        return int(self.run_df(f"SELECT COUNT(*) AS n FROM {table}").iloc[0, 0])

    def sample(self, table: str, n: int = 50_000) -> pd.DataFrame:
        raise NotImplementedError

    def run_df(self, sql: str) -> pd.DataFrame:
        raise NotImplementedError

    def query_row_count(self, sql: str) -> int:
        """Push the user's own query down as-is and return its total row count.

        Used for the "validate against live data" step: the exact same join logic runs
        on the full table, not a sample, and only a single number comes back.
        """
        clean = sql.strip().rstrip(";")
        return int(self.run_df(f"SELECT COUNT(*) AS n FROM ({clean}) jl_validate").iloc[0, 0])

    def close(self) -> None:
        pass


def _cursor_to_df(cur) -> pd.DataFrame:
    cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(cur.fetchall(), columns=cols)


# --------------------------------------------------------------------------- Netezza
class NetezzaConnector(BaseConnector):
    dialect = DIALECT_OF["netezza"]

    def __init__(self, cfg: SourceConfig):
        try:
            import nzpy
        except ImportError as e:
            raise ConnectorError("Netezza support needs: pip install nzpy") from e
        self._nzpy = nzpy
        self.cfg = cfg
        self._conn = None

    def _connect(self):
        if self._conn is None:
            try:
                self._conn = self._nzpy.connect(
                    user=self.cfg.user, password=self.cfg.password, host=self.cfg.host,
                    port=self.cfg.port or 5480, database=self.cfg.database, securityLevel=0,
                )
            except Exception as e:  # noqa: BLE001
                raise ConnectorError(f"Could not connect to Netezza at {self.cfg.host}: {e}") from e
        return self._conn

    def run_df(self, sql: str) -> pd.DataFrame:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(sql)
            return _cursor_to_df(cur)
        except Exception as e:  # noqa: BLE001
            raise ConnectorError(f"Netezza query failed: {e}") from e

    def list_tables(self) -> list[str]:
        sch = self.cfg.schema or self.cfg.user
        df = self.run_df(
            f"SELECT SCHEMA, TABLENAME FROM _V_TABLE WHERE SCHEMA = '{sch}' ORDER BY TABLENAME")
        return [f"{r.SCHEMA}.{r.TABLENAME}" for r in df.itertuples()]

    def sample(self, table: str, n: int = 50_000) -> pd.DataFrame:
        return self.run_df(f"SELECT * FROM {table} LIMIT {n}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


# ------------------------------------------------------------------------- Databricks
class DatabricksConnector(BaseConnector):
    dialect = DIALECT_OF["databricks"]

    def __init__(self, cfg: SourceConfig):
        try:
            from databricks import sql as dbsql
        except ImportError as e:
            raise ConnectorError(
                "Databricks support needs: pip install databricks-sql-connector") from e
        self._dbsql = dbsql
        self.cfg = cfg
        self._conn = None

    def _connect(self):
        if self._conn is None:
            try:
                self._conn = self._dbsql.connect(
                    server_hostname=self.cfg.host, http_path=self.cfg.http_path,
                    access_token=self.cfg.access_token,
                )
            except Exception as e:  # noqa: BLE001
                raise ConnectorError(
                    f"Could not connect to Databricks at {self.cfg.host}: {e}. "
                    "Check the SQL warehouse / cluster is running.") from e
        return self._conn

    def run_df(self, sql: str) -> pd.DataFrame:
        con = self._connect()
        try:
            with con.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall_arrow().to_pandas()
        except Exception as e:  # noqa: BLE001
            raise ConnectorError(f"Databricks query failed: {e}") from e

    def list_tables(self) -> list[str]:
        df = self.run_df(f"SHOW TABLES IN {self.cfg.database or 'default'}")
        col = "tableName" if "tableName" in df.columns else df.columns[-1]
        ns = self.cfg.database or "default"
        return [f"{ns}.{t}" for t in df[col].tolist()]

    def sample(self, table: str, n: int = 50_000) -> pd.DataFrame:
        return self.run_df(f"SELECT * FROM {table} TABLESAMPLE ({n} ROWS)")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


# ----------------------------------------------------------------------------- Oracle
class OracleConnector(BaseConnector):
    dialect = DIALECT_OF["oracle"]

    def __init__(self, cfg: SourceConfig):
        try:
            import oracledb
        except ImportError as e:
            raise ConnectorError("Oracle support needs: pip install oracledb") from e
        self._oracledb = oracledb
        self.cfg = cfg
        self._conn = None

    def _connect(self):
        if self._conn is None:
            dsn = f"{self.cfg.host}:{self.cfg.port or 1521}/{self.cfg.database}"
            try:
                self._conn = self._oracledb.connect(
                    user=self.cfg.user, password=self.cfg.password, dsn=dsn)  # thin mode
            except Exception as e:  # noqa: BLE001
                raise ConnectorError(f"Could not connect to Oracle at {dsn}: {e}") from e
        return self._conn

    def run_df(self, sql: str) -> pd.DataFrame:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(sql)
            return _cursor_to_df(cur)
        except Exception as e:  # noqa: BLE001
            raise ConnectorError(f"Oracle query failed: {e}") from e

    def list_tables(self) -> list[str]:
        owner = self.cfg.schema or self.cfg.user.upper()
        df = self.run_df(
            f"SELECT OWNER, TABLE_NAME FROM ALL_TABLES WHERE OWNER = '{owner}' "
            "ORDER BY TABLE_NAME")
        return [f"{r.OWNER}.{r.TABLE_NAME}" for r in df.itertuples()]

    def sample(self, table: str, n: int = 50_000) -> pd.DataFrame:
        return self.run_df(f"SELECT * FROM {table} FETCH FIRST {n} ROWS ONLY")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


# ------------------------------------------------------------------------ SQL Server
class SQLServerConnector(BaseConnector):
    dialect = DIALECT_OF["sqlserver"]

    def __init__(self, cfg: SourceConfig):
        try:
            import pyodbc
        except ImportError as e:
            raise ConnectorError(
                "SQL Server support needs: pip install pyodbc (plus the Microsoft ODBC "
                "Driver for SQL Server installed on this machine)") from e
        self._pyodbc = pyodbc
        self.cfg = cfg
        self._conn = None

    def _connect(self):
        if self._conn is None:
            conn_str = (
                f"DRIVER={{{self.cfg.odbc_driver}}};SERVER={self.cfg.host},{self.cfg.port or 1433};"
                f"DATABASE={self.cfg.database};UID={self.cfg.user};PWD={self.cfg.password};"
                "Encrypt=yes;TrustServerCertificate=yes;"
            )
            try:
                self._conn = self._pyodbc.connect(conn_str, timeout=10)
            except Exception as e:  # noqa: BLE001
                raise ConnectorError(
                    f"Could not connect to SQL Server at {self.cfg.host}: {e}. "
                    "Check the ODBC driver name matches what's installed locally.") from e
        return self._conn

    def run_df(self, sql: str) -> pd.DataFrame:
        con = self._connect()
        try:
            cur = con.cursor()
            cur.execute(sql)
            return _cursor_to_df(cur)
        except Exception as e:  # noqa: BLE001
            raise ConnectorError(f"SQL Server query failed: {e}") from e

    def list_tables(self) -> list[str]:
        sch = self.cfg.schema or "dbo"
        df = self.run_df(
            f"SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{sch}' ORDER BY TABLE_NAME")
        return [f"{r.TABLE_SCHEMA}.{r.TABLE_NAME}" for r in df.itertuples()]

    def sample(self, table: str, n: int = 50_000) -> pd.DataFrame:
        return self.run_df(f"SELECT TOP {n} * FROM {table}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


_REGISTRY = {
    "netezza": NetezzaConnector, "databricks": DatabricksConnector,
    "oracle": OracleConnector, "sqlserver": SQLServerConnector,
}


def make_connector(cfg: SourceConfig) -> BaseConnector:
    try:
        cls = _REGISTRY[cfg.kind]
    except KeyError as e:
        raise ConnectorError(f"Unknown source kind '{cfg.kind}'. Expected one of {KINDS}.") from e
    return cls(cfg)
