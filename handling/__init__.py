import sqlite3
import pandas as pd
from pathlib import Path


class sqlite_handler:
    def __init__(
            self, db_name: str,
            source : Path,
            build_file: Path | None):

        self._db_name = db_name
        self._source_path = source
        self._Connector = sqlite3.connect(self._source_path / self._db_name)

        if build_file is not None:
            cursor = self._Connector.cursor()
            with open(build_file, encoding='utf8') as f:
                query = f.read()
                cursor.executescript(query)
                self._Connector.commit()

    def _is_empty(self, table_name : str) -> bool:
        cursor = self._Connector.cursor()
        fetch = cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        return True if fetch.fetchone() == (0,) else False

    def write_from_DF(
            self, frame: pd.DataFrame,
            table: str,
            idxname: str = "") -> None:

        if self._is_empty(table):
            frame.to_sql(
                table,
                self._Connector,
                if_exists='replace',
                index=True,
                index_label=idxname
            )
        else:
            print("table already written, override is disabled")

    def execute_query(self, sql: str, output : bool = False) -> str | None:
        cursor = self._Connector.cursor()
        query = cursor.execute(sql)
        fetches = query.fetchall()
        if output:
            for fetch in fetches:
                fetch = str(fetch)
                print(fetch[1:-1])
        return None

    def execute_df_query(self,
                         sql: str,
                         index_col: str | None = None,
                         parse_dates: str | None = None) -> pd.DataFrame:

        if index_col is not None and parse_dates is not None:
            df = pd.read_sql(sql, self._Connector,
                             index_col=index_col,
                             parse_dates=parse_dates
                             )
            return df
        else:
            df = pd.read_sql(sql, self._Connector)
            return df
