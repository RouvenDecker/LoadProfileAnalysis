import sqlite3
import pandas as pd
from pathlib import Path


class sqlite_handler:
    def __init__(
            self,
            db_name: str,
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
        '''
        check if the given table is empty

        Parameters
        ----------
        table_name : str
            table to check

        Returns
        -------
        bool

        '''
        cursor = self._Connector.cursor()
        fetch = cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        return True if fetch.fetchone() == (0,) else False

    def write_from_DF(
            self,
            frame: pd.DataFrame,
            table: str,
            idxname: str = "") -> None:
        '''
        write to database from DataFrame

        Parameters
        ----------
        frame : pd.DataFrame
            Frame to write from
        table : str
            table to write in
        idxname : str, optional
            give a index name, by default ""
        '''
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

    def execute_query(self, sql: str, output : bool = False) -> None:
        '''
        perform a query

        Parameters
        ----------
        sql : str
            query
        output : bool, optional
            printed output if there are fetches, by default False

        '''
        cursor = self._Connector.cursor()
        query = cursor.execute(sql)
        fetches = query.fetchall()
        if output:
            for fetch in fetches:
                fetch = str(fetch)
                print(fetch[1:-1])

    def execute_df_query(self,
                         sql: str,
                         index_col: str | None = None,
                         parse_dates: str | None = None) -> pd.DataFrame:
        '''
        query the database and generate a Dataframe from it

        Parameters
        ----------
        sql : str
            query
        index_col : str | None, optional
            name for the generated index column, by default None
        parse_dates : str | None, optional
            parse the dates , by default None

        Returns
        -------
        pd.DataFrame
            resulting DataFrame
        '''
        if index_col is not None and parse_dates is not None:
            df = pd.read_sql(sql, self._Connector,
                             index_col=index_col,
                             parse_dates=parse_dates
                             )
            return df
        else:
            df = pd.read_sql(sql, self._Connector)
            return df
