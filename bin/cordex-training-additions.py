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

# Read the CSV file into a DataFrame
start_date = '2013-07-01'
obs_date = '2025-01-01'
scenario = sys.argv[1]
loc = sys.argv[2]
model = sys.argv[3]
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
# Modified to start prediction data from same date as training data
all_prediction_data = df[df['utctime'] >= start_date].copy()

# Loop through each predictor in correlation_mappings
for pred in correlation_mappings.keys():
    # Calculate monthly statistics for the chosen pred up to set date
    monthly_stats = df[df['utctime'] <= obs_date].groupby('month')[pred].agg(['mean', 'min', 'max'])

    # Process data for the current predictor
    variable_prefix = correlation_mappings[pred]
    
    # Calculate sum for the current variable
    sum_col = f'{variable_prefix}_sum'
    all_training_data[sum_col] = all_training_data[[f'{variable_prefix}-1', f'{variable_prefix}-2', 
                                                 f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4
    all_prediction_data[sum_col] = all_prediction_data[[f'{variable_prefix}-1', f'{variable_prefix}-2', 
                                                     f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4
    
    # Calculate monthly aggregates for the current variable using historical data
    agg_dict = {sum_col: ['mean', 'min', 'max'], pred: ['mean', 'min', 'max']}
    monthly_aggs = all_training_data.groupby(['year', 'month']).agg(agg_dict)
    monthly_aggs.columns = ['_'.join(col).strip() for col in monthly_aggs.columns.values]
    monthly_aggs.reset_index(inplace=True)
    
    # Calculate difference columns
    monthly_aggs[f'{variable_prefix}_{pred}_diff_mean'] = monthly_aggs[f'{sum_col}_mean'] - monthly_aggs[f'{pred}_mean']
    monthly_aggs[f'{variable_prefix}_{pred}_diff_min'] = monthly_aggs[f'{sum_col}_min'] - monthly_aggs[f'{pred}_min']
    monthly_aggs[f'{variable_prefix}_{pred}_diff_max'] = monthly_aggs[f'{sum_col}_max'] - monthly_aggs[f'{pred}_max']
    
    # Create another aggregation by month only (not by year) for future data filling
    monthly_only_aggs = monthly_aggs.groupby('month').mean().reset_index()
    
    # Add aggregated columns to both datasets
    new_columns = [f'{sum_col}_mean', f'{sum_col}_min', f'{sum_col}_max',
                  f'{pred}_mean', f'{pred}_min', f'{pred}_max',
                  f'{variable_prefix}_{pred}_diff_mean', f'{variable_prefix}_{pred}_diff_min', f'{variable_prefix}_{pred}_diff_max']
    
    all_training_data = all_training_data.merge(monthly_aggs[['year', 'month'] + new_columns], 
                                             on=['year', 'month'], how='left')
    
    # For prediction data, we need special handling for future dates
    # First merge historical dates normally
    past_prediction = all_prediction_data[all_prediction_data['utctime'] <= obs_date].copy()
    past_prediction = past_prediction.merge(monthly_aggs[['year', 'month'] + new_columns], 
                                         on=['year', 'month'], how='left')
    
    # For future dates, use the monthly averages (not specific to year)
    future_prediction = all_prediction_data[all_prediction_data['utctime'] > obs_date].copy()
    
    # Merge the monthly stats for future data
    for month in range(1, 13):
        month_mask = future_prediction['month'] == month
        if month in monthly_stats.index:
            # Fill predictor statistics
            future_prediction.loc[month_mask, f'{pred}_mean'] = monthly_stats.loc[month, 'mean']
            future_prediction.loc[month_mask, f'{pred}_min'] = monthly_stats.loc[month, 'min'] 
            future_prediction.loc[month_mask, f'{pred}_max'] = monthly_stats.loc[month, 'max']
            
            # Fill sum statistics using monthly averages
            month_data = monthly_only_aggs[monthly_only_aggs['month'] == month]
            if not month_data.empty:
                future_prediction.loc[month_mask, f'{sum_col}_mean'] = month_data[f'{sum_col}_mean'].values[0]
                future_prediction.loc[month_mask, f'{sum_col}_min'] = month_data[f'{sum_col}_min'].values[0]
                future_prediction.loc[month_mask, f'{sum_col}_max'] = month_data[f'{sum_col}_max'].values[0]
                
                # Fill difference columns
                future_prediction.loc[month_mask, f'{variable_prefix}_{pred}_diff_mean'] = month_data[f'{variable_prefix}_{pred}_diff_mean'].values[0]
                future_prediction.loc[month_mask, f'{variable_prefix}_{pred}_diff_min'] = month_data[f'{variable_prefix}_{pred}_diff_min'].values[0]
                future_prediction.loc[month_mask, f'{variable_prefix}_{pred}_diff_max'] = month_data[f'{variable_prefix}_{pred}_diff_max'].values[0]
    
    # Combine past and future prediction data
    all_prediction_data = pd.concat([past_prediction, future_prediction], ignore_index=True)
    
    # Adjust future values using differences
    future_mask = all_prediction_data['utctime'] > obs_date
    for month in range(1, 13):
        month_mask = all_prediction_data['month'] == month
        future_month_mask = future_mask & month_mask
        
        # Get average differences from historical data for this month
        past_month_mask = (~future_mask) & month_mask
        diff_mean_past = all_prediction_data.loc[past_month_mask, f'{variable_prefix}_{pred}_diff_mean'].mean()
        
        # Apply adjustments to future data
        if month in monthly_stats.index:
            mean_stat = monthly_stats.loc[month, 'mean']
            diff_means = all_prediction_data.loc[future_month_mask, f'{variable_prefix}_{pred}_diff_mean']
            
            # Adjust the actual predictor value
            all_prediction_data.loc[future_month_mask, pred] = mean_stat + diff_means - diff_mean_past
    
    print(f'Processed {pred} data')

# Save the combined DataFrames to CSV files
all_training_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/training_data_oceanids_{loc}_cordex_{scenario}_{model}.csv', index=False)
print(f'Combined training data saved for {scenario}_{model}_{loc}')

all_prediction_data.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{loc}/prediction_data_oceanids_{loc}_cordex_{scenario}_{model}.csv', index=False)
print(f'Combined prediction data saved for {scenario}_{model}_{loc}')