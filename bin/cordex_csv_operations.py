import pandas as pd
import sys
from cordex_6hour_file_functions import transform_6hourly_to_daily, add_wind_speed
from calendar_operations import is_360_day_calendar, revert_360_day_calendar

loc = sys.argv[1]
model = sys.argv[2]

# File paths for the datasets
file1 = f"/home/ubuntu/data/cordex/{model}_daily_grid2x2_{loc}_rcp85.csv"
file2 = f"/home/ubuntu/data/cordex/{model}_6hours_grid2x2_{loc}_rcp85.csv"

# Read the daily CSV file
df_daily = pd.read_csv(file1, usecols=['time', 'lat', 'lon', 'pr', 'hurs', 'tasmax', 'tasmin'])
df_daily.rename(columns={'time': 'date'}, inplace=True)
df_daily['date'] = pd.to_datetime(df_daily['date'], errors='coerce').dt.normalize()

# Check and revert 360-day calendar if necessary
if is_360_day_calendar(df_daily, 'date'):
    df_daily = revert_360_day_calendar(df_daily, 'date')

# Read the 6-hourly CSV file
df_6hourly = pd.read_csv(file2, usecols=['time', 'lat', 'lon', 'uas', 'vas'])
df_6hourly.rename(columns={'time': 'date'}, inplace=True)
df_6hourly['date'] = pd.to_datetime(df_6hourly['date'], errors='coerce').dt.normalize()

# Check and revert 360-day calendar if necessary
if is_360_day_calendar(df_6hourly, 'date'):
    df_6hourly = revert_360_day_calendar(df_6hourly, 'date')

# Transform 6-hourly data to daily data
df_daily_uas_vas = transform_6hourly_to_daily(df_6hourly)
df_daily_uas_vas = add_wind_speed(df_daily_uas_vas)

# Merge the daily and resampled 6-hourly data on date
df_daily_combined = pd.merge(df_daily, df_daily_uas_vas, on=['date','lat','lon'], how='left')
#print(df_daily_combined.head())

# Pivot the data to have one row per day with columns like pr-1, pr-2, etc.
df_daily_combined['Point'] = df_daily_combined.groupby(['lat', 'lon']).ngroup() + 1
df_daily_combined = df_daily_combined.drop_duplicates(subset=['date', 'Point'])
df_pivot = df_daily_combined.pivot(index='date', columns='Point').reset_index()

# Flatten the multi-level columns
df_pivot.columns = ['_'.join(map(str, col)).strip() for col in df_pivot.columns.values]
df_pivot.rename(columns={'date_': 'utctime'}, inplace=True)

# Extract lat and lon columns
lat_lon_columns = [col for col in df_pivot.columns if col.startswith('lat_') or col.startswith('lon_')]

# Add dayOfYear column
df_pivot['dayOfYear'] = pd.to_datetime(df_pivot['utctime']).dt.dayofyear

# Reorder columns to place lat and lon columns after the date column
df_pivot = df_pivot[['utctime'] + lat_lon_columns + [col for col in df_pivot.columns if col not in ['utctime'] + lat_lon_columns]]

# Save the combined DataFrame to a new CSV file
df_pivot.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/{model}-{loc}-cordex.csv", index=False)
print(df_pivot.head())
print(f'{model}-{loc} completed successfully.')