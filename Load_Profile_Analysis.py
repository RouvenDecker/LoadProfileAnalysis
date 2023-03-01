from sys import exit
import time
import argparse

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import holidays as hd
from pathlib import Path
import sqlite3
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv",
    help="want to generate csv output? (bool)",
    type=lambda x: True if x == "True" else False,
    required=False
)

parser.add_argument(
    "--year",
    help="want to add the Year for output diagramms? (str)",
    type=str,
    required=False
)

args = parser.parse_args()
USE_CSV: bool = args.csv
YEAR: str = args.year


# used directorys
CWD = Path.cwd()
WORKING_DATA = CWD / "data"
INPUT_DIR = CWD / "input"
OUTPUT_DIR = CWD / "output"
INPUT_FILE = INPUT_DIR / "Load_Profile_Analysis.csv"

if not Path.exists(WORKING_DATA):
    Path.mkdir(WORKING_DATA, exist_ok=True)

CONN = sqlite3.connect(WORKING_DATA / "database.db")

DAY_MAPPING = {0: 'Monday',
               1: 'Tuesday',
               2: 'Wednesday',
               3: 'Thursday',
               4: 'Friday',
               5: 'Saturday',
               6: 'Sunday'
               }


def build_dir() -> None:
    '''
    build directory
    '''
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    check_file_existence(INPUT_DIR / "build_database.sql")
    check_file_existence(INPUT_FILE)
    check_file_existence(INPUT_DIR / "ibt.mplstyle")
    plt.style.use(INPUT_DIR / "ibt.mplstyle")


def check_file_existence(p: Path) -> None:
    '''
    check if the given File exists

    Parameters
    ----------
    p : Path
        File to check
    '''
    if not Path.exists(p):
        exit(f"ERROR: file:{p.name} does not exist in {p.parent}")


def build_database() -> None:
    '''
    perform all querys necessary to build the database
    '''
    cursor = CONN.cursor()
    with open(INPUT_DIR / "build_database.sql", "r") as f:
        query = f.read()
        cursor.executescript(query)
        CONN.commit()


def stof(series: pd.Series) -> pd.Series:
    '''
    casting sqlite_strings to floats ,
    necessary because sqlite does not cast floats on itself

    Parameters
    ----------
    series : pd.Series
        Column to cast

    Returns
    -------
    pd.Series
        casted column
    '''
    series = series.apply(lambda a: a.replace(',', '.'))
    series = pd.to_numeric(series, downcast='float')
    return series


def is_empty(table_name: str) -> bool:
    '''
    check if or if not a table is empty

    Parameters
    ----------
    table_name : str
        table to check

    Returns
    -------
    bool
        empty ? yes -> true , else false
    '''
    cursor = CONN.cursor()
    fetch = cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    return True if fetch.fetchone() == (0,) else False


def write_to_db(Frame: pd.DataFrame,
                table_name: str,
                indexname: str = "") -> None:
    '''
    writes the dataframe in the destination SQL-Table

    Parameters
    ----------
    Frame : pd.DataFrame
        Frame to write
    table_name : str
        target_frame
    '''
    if is_empty(table_name):
        Frame.to_sql(
            table_name,
            CONN,
            if_exists='replace',
            index=True,
            index_label=indexname
        )


def datetime_from_utc_to_local(utc_datetime: dt.datetime) -> dt.datetime:
    '''
    change the utc datetime to your local datetime

    Parameters
    ----------
    utc_datetime : dt.datetime
        utc datetime object

    Returns
    -------
    dt.datetime
        local datetime object
    '''
    now = dt.datetime.fromtimestamp(time.time())
    now_utc = dt.datetime.utcfromtimestamp(time.time())
    offset = now - now_utc
    return utc_datetime + offset


def read_input() -> pd.DataFrame:
    '''
    read in the unformated data

    Returns
    -------
    pd.DataFrame
        the unformated DataFrame

    '''
    df = pd.read_csv(
        INPUT_FILE,
        sep=';',
        thousands='.',
        parse_dates=["Timestamp in UTC"],
        date_parser=lambda stamp: dt.datetime.strptime(stamp, '%d.%m.%Y %H:%M')
    )

    return df


def add_localtime_to_input(df: pd.DataFrame) -> None:
    '''
    format the input dataframe and write it in the db

    Parameters
    ----------
    df : pd.DataFrame
        raw input data

    Returns
    -------
    pd.DataFrame
        formated frame
    '''
    df.rename(columns={"Timestamp in UTC": "Timestamp_in_UTC",
                       "Energy in kWh": "Energy_in_kWh"},
              inplace=True
              )

    df.set_index(df["Timestamp_in_UTC"], inplace=True)
    df["Local_Time"] = df["Timestamp_in_UTC"].apply(datetime_from_utc_to_local)

    del df["Timestamp_in_UTC"]
    df = df[["Local_Time", "Energy_in_kWh"]]
    df["Energy_in_kWh"] = stof(df["Energy_in_kWh"])

    check_for_invalids(df)

    # make input updateable
    cursor = CONN.cursor()
    cursor.execute("DELETE FROM input")
    CONN.commit()

    write_to_db(df, "input", indexname='Timestamp_in_UTC')


def check_for_invalids(df: pd.DataFrame) -> pd.DataFrame:
    '''
    check for invalid format and values

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check

    Returns
    -------
    pd.DataFrame
        formated Frame
    '''
    data = np.array([1, 2, 3, 4]).reshape(2, 2)
    day_1 = "01.01.2022 00:00:00"
    day_365 = "31.12.2022 23:00:00"

    showcase = pd.DataFrame(
        index=[day_1, day_365],
        columns=["Timestamp_in_Hour_resolution", "Energy"],
        data=data
    )

    error_msg = f"Wrong Data Format: use this formating\n {showcase}"
    assert df.shape == (8760, 2), error_msg

    if np.sum(df["Energy_in_kWh"] < 0):

        error_msg = """
            negative Values in InputData.\n
            Do u want to change invalids to 0 an proceed?\n
            not changing will leave the invalid in the dataset\n
            YES: 1, NO: some other Key"""
        print(error_msg)
        decission = input()
        if decission == "1":
            df.loc[df["Energy_in_kWh"] < 0, ["Energy_in_kWh"]] = 0
            return df
        else:
            return df
    else:
        return df


def execute_query(sql: str) -> None:
    '''
    helper for frameless querys

    Parameters
    ----------
    sql : str
        query to perform
    '''
    cursor = CONN.cursor()
    query = cursor.execute(sql)
    fetches = query.fetchall()
    for fetch in fetches:
        print(fetch)
    print("done")


def yearly_calculation(year: str = "...") -> None:
    '''
    caluclate the yearly Energy consumption and create tables

    Parameters
    ----------
    year : str
        year of interest
    '''
    year_in_GWh = pd.read_sql("SELECT Energy_in_kWh FROM input", CONN)
    sum = year_in_GWh["Energy_in_kWh"].sum()
    sum = round(sum / 1000, 2)

    year_in_GWh = pd.DataFrame(data={"Consumption_in_GWh": sum}, index=[year])
    year_in_GWh.index.name = "Year"

    if USE_CSV:
        if not Path.exists(OUTPUT_DIR / "yearly_consumption_in_GWh.csv"):
            year_in_GWh.to_csv(
                OUTPUT_DIR / "yearly_consumption_in_GWh.csv",
                index=True,
                sep=';'
            )

    write_to_db(year_in_GWh, "YearlyConsumption", indexname="Year")


def monthly_calculation() -> None:
    '''
    calculate the monthly indicators and create tables
    '''
    month_in_GWh = pd.read_sql(
        "SELECT Timestamp_in_UTC, Energy_in_kWh FROM input",
        CONN,
        index_col="Timestamp_in_UTC",
        parse_dates=["Timestamp_in_UTC"]
    )

    monthly_maximum(month_in_GWh)

    group_by_conditon = month_in_GWh.index.month
    month_in_GWh = (month_in_GWh.groupby(group_by_conditon).sum() / 1000)
    month_in_GWh.index.name = "Month"
    month_in_GWh.rename(
        columns={"Energy_in_kWh": "Energy_in_GWh"},
        inplace=True
    )
    rounded = month_in_GWh["Energy_in_GWh"].apply(lambda a: round(a, 2))
    month_in_GWh["Energy_in_GWh"] = rounded

    if USE_CSV:
        if not Path.exists(OUTPUT_DIR / "monthly_consumption_in_GWh.csv"):
            month_in_GWh.to_csv(
                OUTPUT_DIR / "monthly_consumption_in_GWh.csv",
                index=True,
                sep=';'
            )

    write_to_db(month_in_GWh, "MonthlyConsumption", indexname="Month")


def monthly_maximum(frame: pd.DataFrame) -> None:
    '''
    calculate the maximum output for each month ,then create tables

    Parameters
    ----------
    frame : pd.DataFrame
        month to calculate
    '''
    month_max = frame.groupby(frame.index.month).max()
    rounded = month_max["Energy_in_kWh"].apply(lambda a: round(a, 2))
    month_max["Energy_in_kWh"] = rounded
    month_max.index.name = "Month"

    if USE_CSV:
        if not Path.exists(OUTPUT_DIR / "monthly_maximum_Energy_in_kWh.csv"):
            month_max.to_csv(
                OUTPUT_DIR / "monthly_maximum_Energy_in_kWh.csv",
                sep=';',
                index=True
            )

    write_to_db(month_max, "MaxEnergyPerMonth", indexname="Hours")


def calculate_KPIs(year: str) -> None:
    '''
    calculate the Key Performance Indicator

    Parameters
    ----------
    year : str
       year of interest
    '''
    yearly_calculation(year)
    monthly_calculation()


def create_weekday_table(country: str = "DE") -> pd.DataFrame:
    '''
    append the weekday/holiday information to the DataFrame

    Parameters
    ----------
    country : str, optional
        country from whom the holidays to get , by default "DE"

    Returns
    -------
    pd.DataFrame
        weekday/holiday added Dataframe
    '''
    df = pd.read_sql(
        "SELECT Timestamp_in_UTC,Energy_in_kWh FROM input",
        CONN,
        index_col=["Timestamp_in_UTC"],
        parse_dates=["Timestamp_in_UTC"]
    )

    df["weekday"] = df.index.day % 6
    df["weekday"] = df["weekday"].apply(lambda day: DAY_MAPPING[day])

    de_holidays = hd.country_holidays(country, years=2022)

    # code creates "holiday_hours" containing all hourly holiday timestamps
    # it then updates the info in df
    holidays = []
    holiday_hours = pd.DataFrame()

    for day, _ in de_holidays.items():
        holidays.append(day)

    for day in holidays:
        hours = get_hours_per_day(pd.to_datetime(day))
        holiday_hours = pd.concat([holiday_hours, hours])

    holiday_hours.set_index(holiday_hours[0], inplace=True)
    idxs = df.index.isin(holiday_hours.index)
    df.loc[idxs, ["weekday"]] = "holiday"

    return df


def get_hours_per_day(given_day) -> pd.Series:
    '''
    return the hourly Timestamps from 00:00 till 23:00 of given day

    Parameters
    ----------
    given_day : pd.DatetimeIndex
        given Day  eg. : 01.01.2022

    Returns
    -------
    pd.Series
        all Timestamps with hours eg.: 01.01.2022 00:00 - 01.01.2022 23:00
    '''

    end_hour = given_day + pd.Timedelta(hours=23)
    holiday_hours = pd.date_range(start=given_day, end=end_hour, freq='1H')
    return pd.Series(holiday_hours, name="date")


def median_for_ReferenceDays(df: pd.DataFrame) -> None:
    '''
    generate the median_ReferenceDays tables

    Parameters
    ----------
    df : pd.DataFrame
        input data

    Returns
    -------
    pd.DataFrame
        output as ReferenceDays
    '''

    remap = {'Monday': 'workday',
             'Tuesday': 'workday',
             'Wednesday': 'workday',
             'Thursday': 'workday',
             'Friday': 'workday',
             'Saturday': 'Saturday',
             'Sunday': 'holiday',
             'holiday': 'holiday'
             }

    df["weekday"] = df["weekday"].apply(lambda day: remap[day])
    work_days = df[df["weekday"] == 'workday']
    holidays = df[df["weekday"] == 'holiday']
    saturdays = df[df["weekday"] == 'Saturday']

    work_days = format_ReferenceDay(
        work_days,
        new_col_name='Workdays_median_in_kWh'
    )
    holidays = format_ReferenceDay(
        holidays,
        new_col_name='Holidays_median_in_kWh'
    )
    saturdays = format_ReferenceDay(
        saturdays,
        new_col_name='Saturdays_median_in_kWh'
    )

    combined_ReferenceDays = pd.concat(
        [work_days, holidays, saturdays],
        axis=1
    )

    if USE_CSV:
        if not Path.exists(OUTPUT_DIR / "median_Referencedays.csv"):
            combined_ReferenceDays.to_csv(
                OUTPUT_DIR / "median_Referencedays.csv",
                sep=';',
                decimal='.'
            )

    write_to_db(combined_ReferenceDays,
                "MedianReferenceDays",
                indexname="Hours"
                )


def format_ReferenceDay(frame: pd.DataFrame,
                        new_col_name: str = "") -> pd.DataFrame:
    '''
    format the given Day to wanted design and grouping

    Parameters
    ----------
    frame : pd.DataFrame
        unorderd Frame
    new_col_name : str, optional
        new Columname, by default ""

    Returns
    -------
    pd.DataFrame
        formated Frame
    '''
    frame = frame.groupby(frame.index.hour).median(numeric_only=True)
    frame.index = frame.index.map(pretty_time_h)
    frame.index.rename('Hours', inplace=True)
    frame.rename(columns={'Energy_in_kWh': new_col_name}, inplace=True)
    frame[new_col_name] = frame[new_col_name].apply(lambda a: round(a, 2))
    return frame


def pretty_time_h(time: int) -> str:
    '''
    return a number (0-99) as Timestring
    e.g: 5 -> 05:00

    Parameters
    ----------
    time : int
        hours

    Returns
    -------
    str
        hours as string
    '''
    if time < 10:
        formated = "0"
        return str(time).join([formated, ':00'])
    else:
        return str(time) + ':00'


def plot_ReferenceDay_boxplots(year: str) -> None:
    '''
    plot 1 Boxplot per Reference Day and the
    comparisson Boxplot over the entire Year


    Parameters
    ----------
    year : str
        year of interest
    '''
    data = pd.read_sql(
        "SELECT * FROM MedianReferenceDays",
        CONN,
        index_col=["Hours"]
    )

    input = pd.read_sql(
        "SELECT * FROM input",
        CONN,
        index_col=["Timestamp_in_UTC"],
        parse_dates=["Timestamp_in_UTC"]
    )

    input = input["Energy_in_kWh"]

    workdays = data["Workdays_median_in_kWh"]
    saturdays = data["Saturdays_median_in_kWh"]
    holidays = data["Holidays_median_in_kWh"]

    fig, axes = plt.subplots(1, 4, sharey=True)
    fig.suptitle(year)
    axes[0].boxplot(workdays)
    axes[0].set_xlabel("workdays")
    axes[0].set_xticks([])
    axes[0].set_ylabel("Energy in kWh")

    axes[1].boxplot(saturdays)
    axes[1].set_xlabel("saturdays")
    axes[1].set_xticks([])

    axes[2].boxplot(holidays)
    axes[2].set_xlabel("holidays")
    axes[2].set_xticks([])

    axes[3].boxplot(input)
    axes[3].set_xlabel("total")
    axes[3].set_xticks([])

    if not Path.exists(OUTPUT_DIR / "ReferenceDay_boxplots.png"):
        fig.savefig(OUTPUT_DIR / "ReferenceDay_boxplots.png")


def plot_daily_consumption() -> None:
    '''
    plot the daily Energy consumption over the entire Year
    '''
    daily = pd.read_sql(
        "SELECT Timestamp_in_UTC,Energy_in_kWh FROM input",
        CONN,
        index_col=["Timestamp_in_UTC"],
        parse_dates=["Timestamp_in_UTC"]
    )

    daily = daily.resample("D", axis=0).sum()

    fig, ax = plt.subplots()
    ax.plot(daily.index, daily["Energy_in_kWh"])
    ax.set_ylabel("EnergyConsumption_in_kWh")

    if not Path.exists(OUTPUT_DIR / "DailyEnergyConsumption.png"):
        fig.savefig(OUTPUT_DIR / "DailyEnergyConsumption.png")


def create_Heatmap(year: str) -> None:
    '''
    create Heatmap for Energy consumption
    x - axis : Days
    y - axis : Hours

    Parameters
    ----------
    year : str
        year of interest
    '''
    input = pd.read_sql(
        "SELECT Timestamp_in_UTC,Energy_in_kWh FROM input",
        CONN,
        index_col=["Timestamp_in_UTC"],
        parse_dates=["Timestamp_in_UTC"]
    )

    # values get read row wise , they need to be transposed
    values = input.values.reshape(365, 24)
    values = values.T

    hours = [y for y in range(0, 24)]
    days = [x for x in range(1, 366)]
    grid = pd.DataFrame(index=hours, columns=days, data=values)

    fig, ax = plt.subplots()
    fig.suptitle(year)
    sns.heatmap(grid, robust=True, cbar_kws={"label": "Energy_in_kWh"})
    ax.set(xlabel="Days", ylabel="Hours")

    if not Path.exists(OUTPUT_DIR / "HeatMap.png"):
        fig.savefig(OUTPUT_DIR / "HeatMap.png")


def main():
    '''
    Script to Analyse a Yearly Energy Load Profile.
    '''

    build_dir()
    build_database()
    df = read_input()
    add_localtime_to_input(df)
    calculate_KPIs(YEAR)
    weekday_table = create_weekday_table()
    median_for_ReferenceDays(weekday_table)
    plot_ReferenceDay_boxplots(YEAR)
    plot_daily_consumption()
    create_Heatmap(YEAR)
    print("Done")


if __name__ == "__main__":
    main()
