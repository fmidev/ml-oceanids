import calendar_operations
import sys
import pandas as pd
import numpy as np
import glob
import os

def calculate_wind_speed(u, v):
    """Calculate wind speed from u and v components."""
    return np.sqrt(u**2 + v**2)

def process_6hour(scenario, location, model):
    """Process 6-hourly wind data to calculate daily wind speed."""
    # Construct exact file path based on provided pattern
    file_path = f"/home/ubuntu/data/cordex/{scenario}/{location}/{model}_6hours_grid2x2_{location}.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Read the 6-hourly data
    df = pd.read_csv(file_path)
    
    # Check if data uses 360-day calendar
    is_360_day = calendar_operations.is_360_day_calendar(df, 'time')
    if is_360_day:
        print("Detected 360-day calendar in 6-hour data, adjusting...")
        df = calendar_operations.revert_360_day_calendar(df, 'time')

    df['date'] = pd.to_datetime(df['time']).dt.date

    # Identify spatial columns (lat/lon)
    spatial_cols = []
    for col in ['lat', 'lon', 'latitude', 'longitude', 'x', 'y']:
        if col in df.columns:
            spatial_cols.append(col)
    

    # Group by date AND spatial columns to preserve the 4 points per date
    group_cols = ['date'] + spatial_cols
    daily_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
    
    # Calculate wind speed from components and rename to sfcWind
    daily_df['sfcWind'] = calculate_wind_speed(daily_df['uas'], daily_df['vas'])
    
    return daily_df

def process_max_wind(scenario, location, model):
    """Process the MaxWindSpeed daily data file."""
    file_path = f"/home/ubuntu/data/cordex/{scenario}/{location}/{model}_MaxWindSpeed_daily_grid2x2_{location}.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    max_wind_df = pd.read_csv(file_path)
    
    # Check if data uses 360-day calendar
    is_360_day = calendar_operations.is_360_day_calendar(max_wind_df, 'time')
    if is_360_day:
        print("Detected 360-day calendar in MaxWindSpeed data, adjusting...")
        max_wind_df = calendar_operations.revert_360_day_calendar(max_wind_df, 'time')
    
    max_wind_df['date'] = pd.to_datetime(max_wind_df['time']).dt.date

    max_wind_df = max_wind_df.rename(columns={'MaxWindSpeed': 'maxWind'})
    
    return max_wind_df

def reset_date_range(df, max_wind_df=None):
    """Reset date range using MaxWindSpeed dataset dates.
    
    Args:
        df: DataFrame to process
        calendar_type: Not used but kept for compatibility
        max_wind_df: MaxWindSpeed DataFrame to use its dates
    """
    if max_wind_df is None:
        print("Error: MaxWindSpeed dataset is required")
        return df
 
    # Reset index first
    df.reset_index(drop=True, inplace=True)
    
    # Use the time values directly from max_wind_df since they already have 4 points per date
    max_wind_times = max_wind_df['time'].values
    
    # Make sure we have enough dates
    if len(max_wind_times) < len(df):
        print(f"Warning: MaxWindSpeed has only {len(max_wind_times)} timestamps, but we need {len(df)}")
        # Repeat the last available dates if needed
        while len(max_wind_times) < len(df):
            max_wind_times = np.append(max_wind_times, max_wind_times[-4:])
    
    # Trim to exact size needed
    dates_to_use = max_wind_times[:len(df)]
    
    # Assign to dataframe
    df['time'] = dates_to_use
    
    return df

def process_daily(scenario, location, model, max_wind_df=None):
    """Process the regular daily data file."""
    file_path = f"/home/ubuntu/data/cordex/{scenario}/{location}/{model}_daily_grid2x2_{location}.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Read the daily data file
    daily_df = pd.read_csv(file_path, index_col=['time'], parse_dates=['time'])
    reset_date_range(daily_df, max_wind_df)

    # Check if data uses 360-day calendar
    temp_df = daily_df.reset_index()
    is_360_day = calendar_operations.is_360_day_calendar(temp_df, 'time')
    if is_360_day:
        print("Detected 360-day calendar in daily data, adjusting...")
        daily_df = calendar_operations.revert_360_day_calendar(daily_df, 'time')
    
    # Create date column and drop time column (we'll only use date)
    daily_df['date'] = pd.to_datetime(daily_df['time']).dt.date
    daily_df = daily_df.drop(columns=['time'])
    
    return daily_df

def process_cordex_data(scenario, location, model):
    """Process CORDEX data for the given location and model."""
    # Process 6-hourly wind data
    wind_data = process_6hour(scenario, location, model)
    if wind_data is None:
        print("Failed to process 6-hour wind data")
        return
    
    # Process MaxWindSpeed data
    max_wind_data = process_max_wind(scenario, location, model)
    if max_wind_data is None:
        print("Failed to process MaxWindSpeed data")
        return
    
    # Process regular daily data, passing the MaxWindSpeed data for date reference
    daily_data = process_daily(scenario, location, model, max_wind_df=max_wind_data)
    if daily_data is None:
        print("Failed to process daily data")
        return
    
    # Print data counts for debugging
    print(f"Daily data rows: {len(daily_data)}")
    print(f"Wind data rows: {len(wind_data)}")
    print(f"Max wind data rows: {len(max_wind_data)}")
    
    # Print date ranges to identify mismatches
    print(f"Daily data date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print(f"Wind data date range: {wind_data['date'].min()} to {wind_data['date'].max()}")
    print(f"Max wind date range: {max_wind_data['date'].min()} to {max_wind_data['date'].max()}")
    
    # Calculate number of days difference
    daily_days = daily_data['date'].nunique()
    wind_days = wind_data['date'].nunique()
    max_wind_days = max_wind_data['date'].nunique()
    print(f"Daily data has {daily_days} unique days")
    print(f"Wind data has {wind_days} unique days")
    print(f"Max wind data has {max_wind_days} unique days")
    
    # Find which dates are in daily_data but not in wind_data
    daily_unique_dates = set(daily_data['date'].unique())
    wind_unique_dates = set(wind_data['date'].unique())
    max_wind_unique_dates = set(max_wind_data['date'].unique())
    
    dates_only_in_daily = daily_unique_dates - wind_unique_dates
    if dates_only_in_daily:
        print(f"Found {len(dates_only_in_daily)} dates in daily data that are not in wind data")
        print(f"First few missing dates: {sorted(list(dates_only_in_daily))[:5]}")
    
    # Align the datasets by using only dates present in all three
    common_dates = daily_unique_dates.intersection(wind_unique_dates).intersection(max_wind_unique_dates)
    print(f"Found {len(common_dates)} dates common to all datasets")
    
    # Filter all datasets to only include common dates
    daily_data = daily_data[daily_data['date'].isin(common_dates)]
    wind_data = wind_data[wind_data['date'].isin(common_dates)]
    max_wind_data = max_wind_data[max_wind_data['date'].isin(common_dates)]
    
    print(f"After filtering to common dates:")
    print(f"Daily data rows: {len(daily_data)}")
    print(f"Wind data rows: {len(wind_data)}")
    print(f"Max wind data rows: {len(max_wind_data)}")
    
    # After filtering to common dates, verify spatial point counts
    if len(daily_data) > 0:
        daily_points_per_date = len(daily_data) / daily_data['date'].nunique()
        print(f"Daily data points per date after filtering: {daily_points_per_date}")
        
    if len(wind_data) > 0:
        wind_points_per_date = len(wind_data) / wind_data['date'].nunique()
        print(f"Wind data points per date after filtering: {wind_points_per_date}")
    
    # If we still have unequal points per date, force the datasets to have 4 points per date
    if len(daily_data) > 0 and daily_points_per_date != 4:
        print("Adjusting daily data to ensure 4 points per date...")
        
        spatial_cols = [col for col in ['lat', 'lon', 'latitude', 'longitude', 'x', 'y'] 
                       if col in daily_data.columns]
        
        if spatial_cols:
            # Get unique spatial combinations
            spatial_combos = daily_data[spatial_cols].drop_duplicates()
            
            if len(spatial_combos) < 4:
                print(f"Warning: Only found {len(spatial_combos)} unique spatial points, not 4")
            elif len(spatial_combos) > 4:
                print(f"Warning: Found {len(spatial_combos)} unique spatial points, more than expected 4")
                # Take the first 4 combinations
                spatial_combos = spatial_combos.head(4)
            
            # Force exactly 4 points per date by keeping only data with these spatial combinations
            daily_data = daily_data.merge(spatial_combos, on=spatial_cols, how='inner')
    
    # Identify the correct merge columns
    merge_cols = ['date']
    spatial_cols = []
    for col in ['lat', 'lon', 'latitude', 'longitude', 'x', 'y']:
        if (col in daily_data.columns and 
            col in wind_data.columns and 
            col in max_wind_data.columns):
            merge_cols.append(col)
            spatial_cols.append(col)
    
    print(f"Merging on columns: {merge_cols}")
    
    # Standardize data types for merge columns to ensure consistency
    for col in spatial_cols:
        wind_data[col] = wind_data[col].astype(float).round(6)
        max_wind_data[col] = max_wind_data[col].astype(float).round(6)
        daily_data[col] = daily_data[col].astype(float).round(6)
    
    # Before merging, check if merge keys are unique in each dataset
    for dataset, name in [(daily_data, "daily_data"), (wind_data, "wind_data"), (max_wind_data, "max_wind_data")]:
        duplicate_keys = dataset[merge_cols].duplicated().sum()
        if duplicate_keys > 0:
            print(f"Warning: Found {duplicate_keys} duplicate merge keys in {name}")
            print(f"Taking first occurrence of each duplicate key in {name}")
            dataset = dataset.drop_duplicates(subset=merge_cols, keep='first')

    # Check if sfcWind already exists in daily_data and remove it to avoid duplication
    if 'sfcWind' in daily_data.columns:
        print("Found existing sfcWind column in daily data. Replacing it with values from 6-hourly data.")
        daily_data = daily_data.drop(columns=['sfcWind'])
    
    # Keep only necessary columns to avoid duplicates
    wind_subset = wind_data[merge_cols + ['sfcWind']]
    max_wind_subset = max_wind_data[merge_cols + ['maxWind']]
    
    # First, merge daily data with wind data
    # Changed validation to allow for one-to-many or many-to-one relations if needed
    merged_df = pd.merge(daily_data, wind_subset, on=merge_cols, how='left')
    
    # Check for unexpected row count after first merge
    unique_dates = merged_df['date'].nunique()
    rows_per_date = len(merged_df) / unique_dates if unique_dates > 0 else 0
    print(f"After first merge: {len(merged_df)} rows, {unique_dates} unique dates")
    print(f"Rows per date: {rows_per_date}")
    
    # Then merge with max wind data
    # Changed validation to allow for many-to-many if needed
    final_data = pd.merge(merged_df, max_wind_subset, on=merge_cols, how='left')
    
    # Final check
    unique_dates_final = final_data['date'].nunique()
    rows_per_date_final = len(final_data) / unique_dates_final if unique_dates_final > 0 else 0
    print(f"After final merge: {len(final_data)} rows, {unique_dates_final} unique dates")
    print(f"Rows per date: {rows_per_date_final}")
    
    # Verify each date has exactly 4 rows
    date_counts = final_data['date'].value_counts()
    irregular_dates = date_counts[date_counts != 4]
    if not irregular_dates.empty:
        print("Warning: Some dates don't have exactly 4 rows:")
        print(irregular_dates)
        
        # Fix dates with too many rows - keep only 4 rows per date, 1 per unique spatial point
        print("Fixing dates with incorrect number of rows...")
        
        fixed_dfs = []
        for date, group in final_data.groupby('date'):
            if len(group) > 4:
                # Keep only one row per unique lat/lon combination
                group_deduped = group.drop_duplicates(subset=spatial_cols)
                
                # If we still have more than 4, take the first 4
                if len(group_deduped) > 4:
                    group_deduped = group_deduped.head(4)
                    
                # If we have less than 4, print warning
                if len(group_deduped) < 4:
                    print(f"Warning: After deduplication, date {date} has only {len(group_deduped)} unique spatial points")
                
                fixed_dfs.append(group_deduped)
            else:
                fixed_dfs.append(group)
        
        final_data = pd.concat(fixed_dfs)
        
        # Verify fix worked
        date_counts = final_data['date'].value_counts()
        irregular_dates = date_counts[date_counts != 4]
        if irregular_dates.empty:
            print("Fix successful! All dates now have exactly 4 rows.")
        else:
            print(f"Warning: Still have {len(irregular_dates)} dates with incorrect row counts")
    
    # Save to a new file
    output_filename = f"/home/ubuntu/data/cordex/{scenario}/cordex_{scenario}_{location}_{model}.csv"
    
    # Ensure 'date' is the first column and drop 'time' if it exists
    if 'time' in final_data.columns:
        final_data = final_data.drop(columns=['time'])
    
    # Reorder columns to have date first
    cols = final_data.columns.tolist()
    if 'date' in cols:
        cols.remove('date')
        cols = ['date'] + cols
        final_data = final_data[cols]
    
    final_data.to_csv(output_filename, index=False)
    print(f"Combined data saved to {output_filename}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python cordex-csv-format.py <scenario> <location> <model>")
        sys.exit(1)
    
    scenario = sys.argv[1]
    location = sys.argv[2]
    model = sys.argv[3]
    
    
    process_cordex_data(scenario, location, model)

if __name__ == "__main__":
    main()

