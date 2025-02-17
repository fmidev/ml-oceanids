import pandas as pd

def convert_to_daily(csv_file, start_date=None, column_mapping=None, agg_rules=None):
    """
    Convert high-frequency observations to daily values with specific aggregation rules.
    
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
    df['time'] = pd.to_datetime(df['time'])
    
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
            'RH_PT1M_AVG': 'RH'    # Relative humidity 1-min
        }
    
    # Default aggregation rules
    if agg_rules is None:
        agg_rules = {
            'WG': 'max',    # Maximum wind gust
            'TN': 'min',    # Minimum temperature
            'TX': 'max',    # Maximum temperature
            'TP': 'sum',    # Total precipitation
            'RH': 'mean'    # Average relative humidity
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
        
    # Start with the first DataFrame
    merged_df = daily_dfs[0]
    
    # Merge with remaining DataFrames
    for df in daily_dfs[1:]:
        merged_df = merged_df.merge(df, on=date_column, how='outer')
    
    # Sort by date
    merged_df = merged_df.sort_values(date_column)
    
    # Round numeric columns to 2 decimal places
    numeric_cols = merged_df.select_dtypes(include=['float64']).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].round(2)
    
    return merged_df

def standardize_column_names(df):
    """
    Standardize column names to FMI format with PT24H period.
    
    Args:
        df (pd.DataFrame): DataFrame with simplified column names
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    column_mapping = {
        'WG': 'WG_PT24H_MAX',
        'TX': 'TA_PT24H_MAX',
        'TN': 'TA_PT24H_MIN',
        'TP': 'TP_PT24H_ACC',
        'RH': 'RH_PT24H_AVG'
    }
    
    return df.rename(columns=column_mapping)

if __name__ == "__main__":    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process FMI observations to daily values')
    parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format')
    args = parser.parse_args()
    
    # Convert files to daily values
    files = [
        "/home/ubuntu/data/synop/fmisid115857_PraiaDaVittoria2012-2024_ws-wd-wg-rh_3h.csv",
        "/home/ubuntu/data/synop/fmisid115857_PraiaDaVittoria2012-2024_tanx-pra_12h.csv"
    ]
    
    # Process each file and merge
    daily_dfs = [convert_to_daily(f, start_date=args.start_date) for f in files]
    merged_df = merge_daily_files(daily_dfs)
    
    if merged_df is not None:
        # Standardize column names
        merged_df = standardize_column_names(merged_df)
        
        # Get year range for filename
        year_range = f"{merged_df['utctime'].dt.year.min()}-{merged_df['utctime'].dt.year.max()}"
        output_file = f"/home/ubuntu/data/synop/obs_data_PraiaDaVittoria_2012-2024.csv"
        merged_df.to_csv(output_file, index=False)
        print(merged_df.head())
        print(f"Saved daily observations to {output_file}")
