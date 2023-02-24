DROP TABLE IF EXISTS input;
CREATE TABLE input(
    Timestamp_in_UTC TIMESTAMP PRIMARY KEY NOT NULL,
    Local_Time TIMESTAMP NOT NULL,
    Energy_in_kWh REAL NOT NULL);

DROP TABLE IF EXISTS YearlyConsumption;
CREATE TABLE YearlyConsumption(
    Year TEXT NOT NULL,
    Energy_in_GWh REAL NOT NULL);

DROP TABLE IF EXISTS MonthlyConsumption;
CREATE TABLE MonthlyConsumption(
    Month TEXT NOT NULL,
    Energy_in_GWh REAL NOT NULL);

DROP TABLE IF EXISTS MaxEnergyPerMonth;
CREATE TABLE MaxEnergyPerMonth(
    Month TEXT NOT NULL,
    Energy_in_kWh REAL NOT NULL);

DROP TABLE IF EXISTS MedianReferenceDays;
CREATE TABLE MedianReferenceDays(
    Hours TEXT NOT NULL,
    Workdays_median_in_kWh REAL NOT NULL,
    Holidays_median_in_kWh REAL NOT NULL,
    Saturdays_median_in_kWh REAL NOT NULL);
