import pandas as pd
import numpy as np
import sys

# Function to process the variable group based on the pred
def process_variable_group(df, pred, include_pred_stats=True):
    variable_prefix = None
    for key, value in correlation_mappings.items():
        if pred in key:
            variable_prefix = value
            break

    df = df.copy()
    df[f'{variable_prefix}_sum'] = df[[f'{variable_prefix}-1', f'{variable_prefix}-2', f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4

    agg_dict = {f'{variable_prefix}_sum': ['mean', 'min', 'max']}
    if include_pred_stats:
        agg_dict[f'{pred}'] = ['mean', 'min', 'max']
    
    monthly_stats = df.groupby(['year', 'month']).agg(agg_dict)
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
    monthly_stats.reset_index(inplace=True)

    monthly_stats[f'{variable_prefix}_{pred}_diff_mean'] = monthly_stats[f'{variable_prefix}_sum_mean'] - monthly_stats.get(f'{pred}_mean', 0)
    monthly_stats[f'{variable_prefix}_{pred}_diff_min'] = monthly_stats[f'{variable_prefix}_sum_min'] - monthly_stats.get(f'{pred}_min', 0)
    monthly_stats[f'{variable_prefix}_{pred}_diff_max'] = monthly_stats[f'{variable_prefix}_sum_max'] - monthly_stats.get(f'{pred}_max', 0)

    df = df.merge(monthly_stats, on=['year', 'month'], how='left')
    return df

# Function to fill data after set date with monthly statistics
def fill_data(df, monthly_stats, pred):
    future_data = df[df['utctime'] > obs_date].copy()
    for month in range(1, 13):
        future_data.loc[future_data['month'] == month, f'{pred}_mean'] = monthly_stats.loc[month, 'mean']
        future_data.loc[future_data['month'] == month, f'{pred}_min'] = monthly_stats.loc[month, 'min']
        future_data.loc[future_data['month'] == month, f'{pred}_max'] = monthly_stats.loc[month, 'max']
    return future_data

def adjust_future(df, monthly_stats, pred, variable_prefix):
    future_data = df[df['utctime'] > obs_date].copy()
    past_data = df[df['utctime'] <= obs_date].copy()

    for month in range(1, 13):
        future_data_month = future_data[future_data['month'] == month]
        past_data_month = past_data[past_data['month'] == month]

        diff_mean_future = future_data_month[f'{variable_prefix}_{pred}_diff_mean']
        diff_min_future = future_data_month[f'{variable_prefix}_{pred}_diff_min']
        diff_max_future = future_data_month[f'{variable_prefix}_{pred}_diff_max']

        diff_mean_past = past_data_month[f'{variable_prefix}_{pred}_diff_mean'].mean()
        diff_min_past = past_data_month[f'{variable_prefix}_{pred}_diff_min'].mean()
        diff_max_past = past_data_month[f'{variable_prefix}_{pred}_diff_max'].mean()

        future_data.loc[future_data['month'] == month, f'{pred}_mean'] = monthly_stats.loc[month, 'mean'] + diff_mean_future - diff_mean_past
        future_data.loc[future_data['month'] == month, f'{pred}_min'] = monthly_stats.loc[month, 'min'] + diff_min_future - diff_min_past
        future_data.loc[future_data['month'] == month, f'{pred}_max'] = monthly_stats.loc[month, 'max'] + diff_max_future - diff_max_past

    return future_data

# Read the CSV file into a DataFrame
start_date = '2006-01-01'
obs_date = '2024-12-01'
loc = sys.argv[1]
model = sys.argv[2]
df = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{model}-{loc}-cordex-with-obs.csv')

# Define the correlation mappings
correlation_mappings = {
    'WG_PT24H_MAX': 'maxWind',
    'TN_PT24H_MIN': 'tasmin',
    'TX_PT24H_MAX': 'tasmax',
    'TP_PT24H_SUM': 'pr'
}

# Convert 'utctime' to datetime and extract year and month
df['utctime'] = pd.to_datetime(df['utctime'])
df['year'] = df['utctime'].dt.year
df['month'] = df['utctime'].dt.month

# Loop through each predictor in correlation_mappings
for pred in correlation_mappings.keys():
    # Calculate monthly statistics for the chosen pred up to set date
    monthly_stats = df[df['utctime'] <= obs_date].groupby('month')[pred].agg(['mean', 'min', 'max'])

    # Fill data after set date with monthly statistics
    future_data = fill_data(df, monthly_stats, pred)

    # Split the original DataFrame into two parts: up to set date and after set date
    df_past = df[df['utctime'] <= obs_date]
    df_future = future_data

    # Process the variable group based on the pred for data up to set date
    processed_df_past = process_variable_group(df_past, pred)

    # Filter processed_df_past from start_date
    processed_df_past = processed_df_past[processed_df_past['utctime'] >= start_date]

    # Process the variable group based on the pred for data after set date, excluding pred stats
    processed_df_future = process_variable_group(df_future, pred, include_pred_stats=False)

    # Combine the processed data up to set date with the processed future data
    combined_df = pd.concat([processed_df_past, processed_df_future], ignore_index=True)

    adjusted_future = adjust_future(combined_df, monthly_stats, pred, correlation_mappings[pred])
    past = combined_df[combined_df['utctime'] <= obs_date]

    combined_df = pd.concat([past, adjusted_future], ignore_index=True)

    start_year = processed_df_past['utctime'].min().year
    end_year = processed_df_past['utctime'].max().year

    prediction_start_year = combined_df['utctime'].min().year
    prediction_end_year = combined_df['utctime'].max().year

    # Rename processed_df_past to training_data and combined_df to prediction_data
    training_data = processed_df_past
    prediction_data = combined_df

    # Remove pred columns
    pred_columns = correlation_mappings.keys()
    prediction_data.drop(columns=pred_columns, inplace=True)
    pred_columns = [col for col in correlation_mappings.keys() if col != pred]
    training_data.drop(columns=pred_columns, inplace=True)

    # Save the updated DataFrame to a new CSV file
    training_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_{model}_{loc.capitalize()}_{pred}_{start_year}-{end_year}.csv', index=False)
    print(f'Training data saved for {model}_{loc}_{pred}')
    prediction_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_oceanids_{model}-{loc.capitalize()}_{pred}_{prediction_start_year}-{prediction_end_year}.csv', index=False)
    print(f'Prediction data saved for {model}_{loc}_{pred}')