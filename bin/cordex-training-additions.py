import pandas as pd
import numpy as np
import sys
import json
import os
from datetime import datetime

# Load harbor config file with proper path
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'harbors_config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

# Read command line arguments
scenario = sys.argv[1]
loc = sys.argv[2]
model = sys.argv[3]

# Get location-specific dates from config
if loc in config:
    # Convert date format YYYYMMDDTHHMMSSZ to YYYY-MM-DD
    start_str = config[loc]['start']
    end_str = config[loc]['end']
    start_date = datetime.strptime(start_str, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d")
    obs_date = datetime.strptime(end_str, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d")
    print(f"Using location-specific dates for {loc}: Start={start_date}, End={obs_date}")
else:
    print(f"Warning: Location {loc} not found in config. Using default dates.")
    start_date = '2000-01-01'
    obs_date = '2025-01-01'

# Read the CSV file into a DataFrame
df = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/cordex_{scenario}_{model}_{loc}.csv')

# Define the correlation mappings
correlation_mappings = {
    'WG_PT24H_MAX': 'maxWind',
    'TA_PT24H_MIN': 'tasmin',
    'TA_PT24H_MAX': 'tasmax',
    'TP_PT24H_ACC': 'pr',
    'RH_PT24H_AVG': 'hurs',
    'WS_PT24H_AVG': 'sfcWind'
}

# Convert 'utctime' to datetime and extract year and month
df['utctime'] = pd.to_datetime(df['utctime'])
df['year'] = df['utctime'].dt.year
df['month'] = df['utctime'].dt.month

# Initialize output DataFrames with the original input data
all_training_data = df[df['utctime'] <= obs_date].copy()
all_training_data = all_training_data[all_training_data['utctime'] >= start_date].copy()
all_prediction_data = df[df['utctime'] >= start_date].copy()

# Loop through each predictor in correlation_mappings
for pred in correlation_mappings.keys():
    # Process data for the current predictor
    variable_prefix = correlation_mappings[pred]
    
    # Calculate sum for the current variable (average of the 4 models)
    sum_col = f'{variable_prefix}_sum'
    all_training_data[sum_col] = all_training_data[[f'{variable_prefix}-1', f'{variable_prefix}-2', 
                                                 f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4
    all_prediction_data[sum_col] = all_prediction_data[[f'{variable_prefix}-1', f'{variable_prefix}-2', 
                                                     f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4
    
    # Calculate monthly aggregates - using extreme values
    agg_dict = {
        sum_col: ['mean', lambda x: x.min(), lambda x: x.max()], 
        pred: ['mean', lambda x: x.min(), lambda x: x.max()]
    }
    monthly_stats = all_training_data.groupby('month').agg(agg_dict)
    
    # Rename the columns properly
    monthly_stats.columns = [
        f'{col[0]}_{"mean" if col[1] == "mean" else "min" if "<lambda_0>" in str(col[1]) else "max"}' 
        for col in monthly_stats.columns
    ]
    monthly_stats.reset_index(inplace=True)
    
    # Calculate difference columns
    monthly_stats[f'{variable_prefix}_{pred}_diff_mean'] = monthly_stats[f'{sum_col}_mean'] - monthly_stats[f'{pred}_mean']
    monthly_stats[f'{variable_prefix}_{pred}_diff_min'] = monthly_stats[f'{sum_col}_min'] - monthly_stats[f'{pred}_min']
    monthly_stats[f'{variable_prefix}_{pred}_diff_max'] = monthly_stats[f'{sum_col}_max'] - monthly_stats[f'{pred}_max']
    
    # Define columns to be added to both datasets
    new_columns = [f'{sum_col}_mean', f'{sum_col}_min', f'{sum_col}_max',
                  f'{pred}_mean', f'{pred}_min', f'{pred}_max',
                  f'{variable_prefix}_{pred}_diff_mean', f'{variable_prefix}_{pred}_diff_min', f'{variable_prefix}_{pred}_diff_max']
    
    # Apply the monthly stats to both training and prediction data by month
    all_training_data = all_training_data.merge(monthly_stats[['month'] + new_columns], 
                                            on=['month'], how='left')
    
    all_prediction_data = all_prediction_data.merge(monthly_stats[['month'] + new_columns],
                                                on=['month'], how='left')
    
    print(f'Processed {pred} data')

# Save the combined DataFrames to CSV files
all_training_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/training_data_oceanids_{loc}_cordex_{scenario}_{model}.csv', index=False)
print(f'Combined training data saved for {scenario}_{model}_{loc}')

all_prediction_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/prediction_data_oceanids_{loc}_cordex_{scenario}_{model}.csv', index=False)
print(f'Combined prediction data saved for {scenario}_{model}_{loc}')