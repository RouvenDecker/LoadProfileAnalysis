import sqlite3
import pandas as pd
from pathlib import Path


class sqlite_handler:
    def __init__(
            self, db_name: Path,
            source : Path,
            buildSQL: Path = Path(""),
            build_path: Path = Path("")):

        self._db_name = db_name
        self._source_path = source
        self._Connector = sqlite3.connect(self._source_path / self._db_name)

        if buildSQL != Path(""):
            cursor = self._Connector.cursor()
            with open(Path(buildSQL) / Path(build_path), "r") as f:
                query = f.read()
                cursor.executescript(query)
                self._Connector.commit()

    def _is_empty(self, table_name : str) -> bool:
        cursor = self._Connector.cursor()
        fetch = cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        return True if fetch.fetchone() == (0,) else False

    def write_from_DF(
            self,
            frame: pd.DataFrame,
            table: str,
            idxname: str = "") -> None:
        if self._is_empty(table):
            frame.to_sql(
                table,
                self._Connector,
                if_exists='replace',
                index=True,
                index_lable=idxname
            )
        else:
            print("table already written, override is disabled")

    def execute_str_query(self, sql: str) -> None:
        cursor = self._Connector.cursor()
        query = cursor.execute(sql)
        fetches = query.fetchall()
        for fetch in fetches:
            print(fetch)
        print("Done")

    def execute_df_query(self,
                         sql: str,
                         index_col: str | None,
                         parse_dates: str | None) -> pd.DataFrame:
        df = pd.read_sql(
            sql,
            self._Connector,
            index_col=index_col,
            parse_dates=parse_dates
        )
        return df


class csv_handler:
    def __init__(self):
        pass
