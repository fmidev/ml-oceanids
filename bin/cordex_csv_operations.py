import pandas as pd
import sys
import os
from cordex_6hour_file_functions import transform_6hourly_to_daily, add_wind_speed, read_weather_data, process_daily_data
from calendar_operations import is_360_day_calendar, revert_360_day_calendar

# Get command line arguments
loc = sys.argv[1]
model = sys.argv[2]
# Optional format argument - can be 'standard' or 'single_file'
data_format = sys.argv[3] if len(sys.argv) > 3 else 'standard'

if data_format == 'single_file':
    # Process data from a single file that contains all variables
    input_file = f"/home/ubuntu/data/cordex/rcp85/{loc}/{model}_daily_grid2x2_{loc}_rcp85.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
    
    # Read the data using the new function
    df = read_weather_data(input_file)
    
    # Process the data
    df_daily_combined = process_daily_data(df)
    
    # Add wind speed if it's not already in the data
    if 'sfcWind' not in df_daily_combined.columns and 'uas' in df_daily_combined.columns and 'vas' in df_daily_combined.columns:
        df_daily_combined = add_wind_speed(df_daily_combined)
    
else:
    # Standard processing with separate daily and 6-hourly files
    # File paths for the datasets
    file1 = f"/home/ubuntu/data/cordex/rcp85/{loc}/{model}_daily_grid2x2_{loc}_rcp85.csv"
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

# Common processing for both data formats
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
output_format = "single" if data_format == "single_file" else "combined"
df_pivot.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/{model}-{loc}-{output_format}-cordex.csv", index=False)

print(df_pivot.head())
print(f'{model}-{loc} completed successfully.')