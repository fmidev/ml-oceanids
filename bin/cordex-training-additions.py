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
training_data = df[df['utctime'] <= obs_date].copy()
training_data = training_data[training_data['utctime'] >= start_date].copy()
prediction_data = df[df['utctime'] >= start_date].copy()

# Loop through each predictor in correlation_mappings
for pred in correlation_mappings.keys():
    # Process data for the current predictor
    variable = correlation_mappings[pred]
    
    # Calculate sum for the current variable (average of the 4 models)
    sum_col = f'{variable}_sum'
    training_data[sum_col] = training_data[[f'{variable}-1', f'{variable}-2', 
                                                 f'{variable}-3', f'{variable}-4']].sum(axis=1) / 4
    prediction_data[sum_col] = prediction_data[[f'{variable}-1', f'{variable}-2', 
                                                     f'{variable}-3', f'{variable}-4']].sum(axis=1) / 4
    
    # Calculate year-month aggregates for variables using ALL prediction data
    # This ensures we have stats for the entire prediction period
    variable_stats = prediction_data.groupby(['year', 'month'])[sum_col].agg(['mean', 'min', 'max'])
    variable_stats.columns = [f'{sum_col}_{col}' for col in variable_stats.columns]
    variable_stats.reset_index(inplace=True)
    
    # Calculate month-only aggregates for preds (still using only training data)
    pred_stats = training_data.groupby('month')[pred].agg(['mean', 'min', 'max'])
    pred_stats.columns = [f'{pred}_{col}' for col in pred_stats.columns]
    pred_stats.reset_index(inplace=True)
    
    # Merge the two stats dataframes to calculate differences
    monthly_stats = variable_stats.merge(pred_stats, on='month', how='left')
    
    # Calculate difference columns
    monthly_stats[f'{variable}_{pred}_diff_mean'] = monthly_stats[f'{sum_col}_mean'] - monthly_stats[f'{pred}_mean']
    monthly_stats[f'{variable}_{pred}_diff_min'] = monthly_stats[f'{sum_col}_min'] - monthly_stats[f'{pred}_min']
    monthly_stats[f'{variable}_{pred}_diff_max'] = monthly_stats[f'{sum_col}_max'] - monthly_stats[f'{pred}_max']
    
    # Define columns to be added to both datasets
    new_columns = [f'{sum_col}_mean', f'{sum_col}_min', f'{sum_col}_max',
                  f'{pred}_mean', f'{pred}_min', f'{pred}_max',
                  f'{variable}_{pred}_diff_mean', f'{variable}_{pred}_diff_min', f'{variable}_{pred}_diff_max']
    
    # Apply the year-month stats to both training and prediction data
    training_data = training_data.merge(monthly_stats[['year', 'month'] + new_columns], 
                                            on=['year', 'month'], how='left')
    
    prediction_data = prediction_data.merge(monthly_stats[['year', 'month'] + new_columns],
                                                on=['year', 'month'], how='left')
    
    print(f'Processed {pred} data')

# Save the combined DataFrames to CSV files
training_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/training_data_oceanids_{loc}_cordex_{scenario}_{model}.csv', index=False)
print(f'Combined training data saved for {scenario}_{model}_{loc}')

prediction_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/prediction_data_oceanids_{loc}_cordex_{scenario}_{model}.csv', index=False)
print(f'Combined prediction data saved for {scenario}_{model}_{loc}')