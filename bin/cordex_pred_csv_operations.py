import pandas as pd
import os
import sys
from calendar_operations import is_360_day_calendar, revert_360_day_calendar


def handle_optional_header(file_path):
    """
    Reads a CSV file with optional headers. If the first line starts with a comment marker (e.g., '#'),
    it skips the commented line and loads the CSV with the correct headers. Otherwise, it reads the CSV without headers.
    
    Args:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Path to the file
    file_path_full = f"/home/ubuntu/data/cordex/{file_path}"

    # Read the first line to check for headers
    with open(file_path_full, 'r') as f:
        first_line = f.readline().strip()

    # Detect if the first line starts with a comment marker (e.g., '#')
    if first_line.startswith('#'):
        # Skip the commented line and load the CSV with the correct headers
        df = pd.read_csv(
            file_path_full,
            sep=r'\s+',
            header=0,  # Read the first non-commented row as the header
            comment='#',  # Skip the comment line
            names=['date', 'lat', 'lon', 'value'],
            engine='python'
        )
    else:
        # Read the CSV without headers
        df = pd.read_csv(
            file_path_full,
            sep=r'\s+',
            header=None,  # No header row
            names=['date', 'lat', 'lon', 'value'],  # Assign column names
            engine='python'
        )
    
    return df

loc = sys.argv[1]
model = sys.argv[2]
preds = ["pr", "sfcWind", "tasmax", "tasmin", "maxWind"]

# Construct file paths for each predictor
file_paths = [f"{model}-{loc}-rcp45-{pred}.csv" for pred in preds]
all_dfs = []

# Load and process each predictor file
for pred_index, file_path in enumerate(file_paths):

    # Step 1: Load the CSV data
    df = handle_optional_header(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if is_360_day_calendar(df, 'date'):
        df = revert_360_day_calendar(df, 'date')

    # Step 2: Create a unique identifier for each (lat, lon) pair
    df['Point'] = df.groupby(['lat', 'lon']).ngroup() + 1

    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'Point'])

    # Step 3: Pivot the DataFrame to make each point's value a separate column
    df_pivot = df.pivot(index='date', columns='Point', values='value').reset_index()

    # Step 4: Extract unique lat and lon values
    if pred_index == 0:
        lat_lon_df = df[['Point', 'lat', 'lon']].drop_duplicates().set_index('Point')
        lat_columns = {f'lat-{point}': lat_lon_df.loc[point, 'lat'] for point in lat_lon_df.index}
        lon_columns = {f'lon-{point}': lat_lon_df.loc[point, 'lon'] for point in lat_lon_df.index}

    # Step 5: Rename the columns as pred1-1, pred1-2, etc.
    df_pivot.columns = ['date'] + [f"{preds[pred_index]}-{i}" for i in range(1, len(df_pivot.columns))]

    # Append the processed DataFrame to the list
    all_dfs.append(df_pivot)

# Step 6: Merge DataFrames on 'date' to combine predictor columns side by side
combined_df = all_dfs[0]
for df in all_dfs[1:]:
    combined_df = combined_df.merge(df, on='date', how='outer')

# Step 7: Add lat and lon columns to the combined DataFrame
lat_lon_columns = []
for point in lat_lon_df.index:
    combined_df[f'lat-{point}'] = lat_columns[f'lat-{point}']
    combined_df[f'lon-{point}'] = lon_columns[f'lon-{point}']
    lat_lon_columns.extend([f'lat-{point}', f'lon-{point}'])

# Rename 'date' column to 'utctime'
combined_df.rename(columns={'date': 'utctime'}, inplace=True)

# Add dayofyear column
combined_df['dayOfYear'] = pd.to_datetime(combined_df['utctime']).dt.dayofyear

# Reorder columns to place lat and lon columns after the date column
combined_df = combined_df[['utctime'] + lat_lon_columns + [col for col in combined_df.columns if col not in ['utctime'] + lat_lon_columns]]

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(f"/home/ubuntu/data/ML/training-data/OCEANIDS/{model}-{loc}-cordex.csv", index=False)

print(f'{model}-{loc} completed successfully.')
#print(combined_df)