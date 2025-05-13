import pandas as pd
import os
import sys

def pivot_cordex_data(location, model, scenario):
    """
    Pivot CORDEX data from row-based format to column-based format with variable-# columns.
    
    Args:
        location (str): Location name
        model (str): Model name
        scenario (str): Scenario (e.g., "rcp85")
        
    Returns:
        pd.DataFrame: Pivoted dataframe with all variables
    """
    # List of variables to process
    variables = ["pr", "sfcWind", "tasmax", "tasmin", "maxWind", "hurs"]
    all_pivoted_dfs = []
    lat_lon_df = None
    
    # Process each variable
    for variable in variables:
        # Define the input file path
        input_file = f"/home/ubuntu/data/cordex/{scenario}/cordex_{scenario}_{location}_{model}.csv"
        if not os.path.exists(input_file):
            print(f"Skipping {variable}: File {input_file} not found")
            continue
            
        print(f"Processing {variable} from {input_file}")
        
        # Load data with space separator
        df = pd.read_csv(input_file)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create point ID for each unique lat/lon pair
        df['Point'] = df.groupby(['lat', 'lon']).ngroup() + 1
        
        # Save lat/lon reference data from first processed variable
        if lat_lon_df is None:
            lat_lon_df = df[['Point', 'lat', 'lon']].drop_duplicates().set_index('Point')
        
        # Pivot the data to have columns for each point
        df_pivot = df.pivot(index='date', columns='Point', values=variable).reset_index()
        
        # Rename columns to include variable name
        df_pivot.columns = ['date'] + [f"{variable}-{i}" for i in range(1, len(df_pivot.columns))]
        
        # Add to list of pivoted dataframes
        all_pivoted_dfs.append(df_pivot)
    
    # Make sure we processed at least one variable
    if not all_pivoted_dfs:
        print("No data was processed. Please check input files.")
        sys.exit(1)
    
    # Merge all dataframes on date
    combined_df = all_pivoted_dfs[0]
    for df in all_pivoted_dfs[1:]:
        combined_df = pd.merge(combined_df, df, on='date', how='outer')
    
    # Add lat and lon columns
    lat_lon_columns = []
    for point in lat_lon_df.index:
        combined_df[f'lat-{point}'] = lat_lon_df.loc[point, 'lat']
        combined_df[f'lon-{point}'] = lat_lon_df.loc[point, 'lon']
        lat_lon_columns.extend([f'lat-{point}', f'lon-{point}'])
    
    # Rename date to utctime and add dayOfYear
    combined_df.rename(columns={'date': 'utctime'}, inplace=True)
    combined_df['dayOfYear'] = pd.to_datetime(combined_df['utctime']).dt.dayofyear
    
    return combined_df, lat_lon_columns

def add_observations(df, location):
    """
    Add observations data from location-specific file.
    
    Args:
        df (pd.DataFrame): DataFrame to add observations to
        location (str): Location name
        
    Returns:
        pd.DataFrame: DataFrame with observations added
    """
    # Path to the observations file
    obs_file = f"/home/ubuntu/data/ML/training-data/OCEANIDS/{location}/obs-oceanids-{location}.csv.gz"
    
    # Check if the file exists
    if not os.path.exists(obs_file):
        print(f"Warning: Observations file not found at {obs_file}")
        return df
    
    # Load observations data
    obs_df = pd.read_csv(obs_file, compression='gzip')
    obs_df['utctime'] = pd.to_datetime(obs_df['utctime'])
    
    # Extract static location columns before merging
    static_columns = {}
    for col in ['name', 'latitude', 'longitude']:
        if col in obs_df.columns:
            # Get the most common non-null value for this column
            values = obs_df[col].dropna()
            if not values.empty:
                static_columns[col] = values.value_counts().idxmax()
    
    # Merge with CORDEX data
    merged_df = pd.merge(df, obs_df, on='utctime', how='left')
    
    # Fill static columns for the entire dataset
    for col, value in static_columns.items():
        merged_df[col] = merged_df[col].fillna(value)
    
    print(f"Added observations data: {list(obs_df.columns)[1:]}")
    print(f"Extended static location columns: {list(static_columns.keys())}")
    
    return merged_df

def reorder_columns(df, lat_lon_columns):
    """
    Reorder columns to follow a consistent format.
    
    Args:
        df (pd.DataFrame): DataFrame to reorder
        lat_lon_columns (list): List of lat/lon column names
        
    Returns:
        pd.DataFrame: DataFrame with reordered columns
    """
    # Define column groups
    time_cols = ['utctime']
    location_cols = [col for col in ['name', 'latitude', 'longitude'] if col in df.columns]
    obs_cols = [col for col in ['WG_PT24H_MAX', 'TA_PT24H_MAX', 'TA_PT24H_MIN', 
                'TP_PT24H_ACC', 'RH_PT24H_AVG', 'WS_PT24H_AVG', 'WD_PT24H_AVG'] 
                if col in df.columns]
    
    # Sort lat-lon columns in numerical order (lat-1, lon-1, lat-2, lon-2, etc.)
    latlon_paired = []
    for i in range(1, 5): 
        lat_col = f'lat-{i}'
        lon_col = f'lon-{i}'
        if lat_col in df.columns and lon_col in df.columns:
            latlon_paired.extend([lat_col, lon_col])
    
    # Organize variable columns by type and number
    var_types = set()
    for col in df.columns:
        if '-' in col and col not in latlon_paired and col != 'dayOfYear':
            var_type = col.split('-')[0]
            var_types.add(var_type)
    
    var_cols = []
    for var_type in sorted(var_types):
        for i in range(1, 5):
            var_col = f"{var_type}-{i}"
            if var_col in df.columns:
                var_cols.append(var_col)
    
    # Other columns
    other_cols = ['dayOfYear']
    
    # Create and apply column order
    new_order = time_cols + location_cols + obs_cols + latlon_paired + var_cols + other_cols
    return df[new_order]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python cordex-csv-transform.py <scenario> <location> <model>")
        sys.exit(1)
    
    scenario = sys.argv[1]
    location = sys.argv[2]
    model = sys.argv[3]
    
    # Step 1: Pivot the CORDEX data
    print("Pivoting CORDEX data...")
    pivoted_df, lat_lon_columns = pivot_cordex_data(location, model, scenario)
    
    # Step 2: Add observations
    print("Adding observation data...")
    with_obs = add_observations(pivoted_df, location)
    
    # Step 3: Reorder columns
    print("Reordering columns...")
    final_df = reorder_columns(with_obs, lat_lon_columns)
    
    # Create output directory
    output_dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{location}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the output
    output_file = f'{output_dir}/cordex_{scenario}_{model}_{location}.csv'
    final_df.to_csv(output_file, index=False)
    print(f"Final output saved to: {output_file}")
