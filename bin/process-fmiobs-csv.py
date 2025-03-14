import pandas as pd
import sys

def convert_to_daily(csv_file, start_date=None, column_mapping=None, agg_rules=None):
    """
    Convert high-frequency observations to daily values.
    
    Args:
        csv_file (str): Path to input CSV file
        start_date (str): Optional start date in YYYY-MM-DD format
        column_mapping (dict): Mapping of original to simplified column names 
        agg_rules (dict): Aggregation rules for each column
    
    Returns:
        pd.DataFrame: Daily aggregated data
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty")
        return None
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("Columns 'lat' and/or 'lon' are missing in the CSV file")
        return None
    df['time'] = pd.to_datetime(df['time'])
    
    # Store first row's lat/lon
    location_info = {
        'latitude': df['lat'].iloc[0],
        'longitude': df['lon'].iloc[0]
    }
    
    # Filter by start date if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['time'] >= start_date]
        if df.empty:
            return None
    
    # Create continuous date range
    date_range = pd.date_range(
        start=max(df['time'].min().date(), start_date.date()) if start_date else df['time'].min().date(),
        end=df['time'].max().date(),
        freq='D'
    )
    
    df = df.set_index('time')
    
    # Default column mappings with multiple possible input names
    if column_mapping is None:
        column_mapping = {
            'WG_PT1H_MAX': 'WG',   # Wind gust hourly
            'WG_PT10M_MAX': 'WG',  # Wind gust 10-min
            'TA_PT12H_MIN': 'TN',  # Min temperature
            'TA_PT12H_MAX': 'TX',  # Max temperature
            'PRA_PT12H_ACC': 'TP', # Precipitation
            'RH_PT1H_AVG': 'RH',   # Relative humidity hourly
            'RH_PT1M_AVG': 'RH',    # Relative humidity 1-min
            'WS_PT1H_AVG': 'WS',   # Wind speed hourly
            'WD_PT1H_AVG': 'WD',    # Wind direction hourly
            'WS_PT10M_AVG': 'WS',  # Wind speed 10-min
            'WD_PT10M_AVG': 'WD'   # Wind direction 10-min
        }
    
    # Default aggregation rules
    if agg_rules is None:
        agg_rules = {
            'WG': 'max',    # Maximum wind gust
            'TN': 'min',    # Minimum temperature
            'TX': 'max',    # Maximum temperature
            'TP': 'sum',    # Total precipitation
            'RH': 'mean',    # Average relative humidity
            'WS': 'mean',   # Average wind speed
            'WD': 'mean'    # Average wind direction
        }
    
    # Select only columns we need
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    if not available_cols:
        return None
        
    df = df[available_cols]
    
    # Group columns by their target name before aggregating
    agg_dict = {}
    for col in available_cols:
        target = column_mapping[col]
        if target not in agg_dict:
            agg_dict[col] = agg_rules[target]
    
    # Aggregate to daily values with updated aggregation dictionary
    daily_data = df.groupby(df.index.date).agg(agg_dict)
    
    # Ensure all dates exist
    daily_data = daily_data.reindex(date_range.date)
    
    # Rename columns to simplified names
    daily_data.rename(
        columns={old: column_mapping[old] for old in available_cols},
        inplace=True
    )
    
    # Reset index to get date as column
    daily_data.index = pd.to_datetime(daily_data.index)
    daily_data = daily_data.reset_index()
    daily_data = daily_data.rename(columns={'index': 'utctime'})
    
    # Add lat/lon columns with their new names
    daily_data['latitude'] = location_info['latitude']
    daily_data['longitude'] = location_info['longitude']
    
    return daily_data

def merge_daily_files(daily_dfs, date_column='utctime'):
    """
    Merge multiple daily DataFrames on the date column.
    
    Args:
        daily_dfs (list): List of daily DataFrames to merge
        date_column (str): Name of the date column to merge on
    
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Filter out None values from failed conversions
    daily_dfs = [df for df in daily_dfs if df is not None]
    if not daily_dfs:
        return None
        
    # Start with the first DataFrame, keeping lat/lon
    merged_df = daily_dfs[0]
    location_info = {
        'latitude': merged_df['latitude'].iloc[0],
        'longitude': merged_df['longitude'].iloc[0]
    }
    
    # Merge with remaining DataFrames, excluding their lat/lon columns
    for df in daily_dfs[1:]:
        df_without_location = df.drop(columns=['latitude', 'longitude'])
        merged_df = merged_df.merge(df_without_location, on=date_column, how='outer')
    
    # Sort by date
    merged_df = merged_df.sort_values(date_column)
    
    # Ensure single lat/lon values
    merged_df['latitude'] = location_info['latitude']
    merged_df['longitude'] = location_info['longitude']
    
    # Round numeric columns to 2 decimal places
    numeric_cols = merged_df.select_dtypes(include=['float64']).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].round(2)
    
    return merged_df

def standardize_column_names(df):
    """
    Standardize column names to FMI format with PT24H period and reorder columns.
    
    Args:
        df (pd.DataFrame): DataFrame with simplified column names
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names and ordered columns
    """
    column_mapping = {
        'WG': 'WG_PT24H_MAX',
        'TX': 'TA_PT24H_MAX',
        'TN': 'TA_PT24H_MIN',
        'TP': 'TP_PT24H_ACC',
        'RH': 'RH_PT24H_AVG',
        'WS': 'WS_PT24H_AVG',
        'WD': 'WD_PT24H_AVG'
    }
    
    # Rename columns first
    df = df.rename(columns=column_mapping)
    
    # Define column order
    base_cols = ['utctime', 'latitude', 'longitude', 'name']
    pred_cols = ['WG_PT24H_MAX', 'TA_PT24H_MAX', 'TA_PT24H_MIN', 'TP_PT24H_ACC', 'RH_PT24H_AVG','WS_PT24H_AVG', 'WD_PT24H_AVG']
    
    # Reorder columns, only including prediction columns that exist
    available_pred_cols = [col for col in pred_cols if col in df.columns]
    ordered_cols = base_cols + available_pred_cols
    
    return df[ordered_cols]

if __name__ == "__main__":    
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Process FMI observations to daily values')
    parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format')
    # NEW: Add harbour name argument
    parser.add_argument('--harbour', type=str, default="Plaisance", help='Name of the harbour')
    parser.add_argument('input_files', nargs='+', help='List of CSV filenames to process (located in /home/ubuntu/data/synop)')
    args = parser.parse_args()
    
    harbour_name = args.harbour  # NEW: Read harbour name
    
    # Base directory for input CSV files
    base_csv_directory = "/home/ubuntu/data/synop"
    # Build full file paths using the base directory
    files = [os.path.join(base_csv_directory, f) for f in args.input_files]
    
    # Process each file and merge
    daily_dfs = [convert_to_daily(f, start_date=args.start_date) for f in files]
    merged_df = merge_daily_files(daily_dfs)
    
    if merged_df is not None:
        # Convert temperature observations from Celsius to Kelvin.
        # If temperature columns exist and are in Celsius, add 273.15.
        if 'TN' in merged_df.columns:
            merged_df['TN'] = merged_df['TN'] + 273.15
        if 'TX' in merged_df.columns:
            merged_df['TX'] = merged_df['TX'] + 273.15
        
        merged_df["name"] = harbour_name  # UPDATED: Use command line harbour name
        merged_df = standardize_column_names(merged_df)
        
        # Get year range for filename
        year_range = f"{merged_df['utctime'].dt.year.min()}-{merged_df['utctime'].dt.year.max()}"
        # UPDATED: Use harbour name in the output file path
        output_file = f"/home/ubuntu/data/ML/training-data/OCEANIDS/{harbour_name}/obs-oceanids-{harbour_name}.csv"
        merged_df.to_csv(output_file, index=False)
        print(merged_df.head())
        print(f"Saved daily observations to {output_file}")
