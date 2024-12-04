import pandas as pd
import numpy as np


# Function to process the variable group based on the pred
def process_variable_group(df, pred, include_pred_stats=True):
    # Find the variable group that correlates with the given pred
    variable_prefix = None
    for key, value in correlation_mappings.items():
        if pred in key:
            variable_prefix = value
            break

    # Sum the values for columns like 'sfcWind-1' to 'sfcWind-4' into a single column
    df[f'{variable_prefix}_sum'] = df[[f'{variable_prefix}-1', f'{variable_prefix}-2', f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4

    # Calculate monthly statistics
    agg_dict = {f'{variable_prefix}_sum': ['mean', 'min', 'max']}
    if include_pred_stats:
        agg_dict[f'{pred}'] = ['mean', 'min', 'max']
    
    monthly_stats = df.groupby(['year', 'month']).agg(agg_dict)

    # Flatten the MultiIndex columns
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]

    # Reset index to make 'year' and 'month' columns again
    monthly_stats.reset_index(inplace=True)

    # Calculate the difference between the summed variable and the predictor
    monthly_stats[f'{variable_prefix}_{pred}_diff_mean'] = monthly_stats[f'{variable_prefix}_sum_mean'] - monthly_stats.get(f'{pred}_mean', 0)
    monthly_stats[f'{variable_prefix}_{pred}_diff_min'] = monthly_stats[f'{variable_prefix}_sum_min'] - monthly_stats.get(f'{pred}_min', 0)
    monthly_stats[f'{variable_prefix}_{pred}_diff_max'] = monthly_stats[f'{variable_prefix}_sum_max'] - monthly_stats.get(f'{pred}_max', 0)

    # Merge the calculated statistics back to the original DataFrame
    df = df.merge(monthly_stats, on=['year', 'month'], how='left')

    # Select only the required columns
    required_columns = ['utctime','lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4','pr-1','pr-2','pr-3','pr-4',
                        'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4','tasmax-1','tasmax-2','tasmax-3','tasmax-4','tasmin-1','tasmin-2','tasmin-3','tasmin-4',
                        'dayofyear', 'year', 'month', f'{pred}_mean', f'{pred}_min', f'{pred}_max', f'{variable_prefix}_sum',
                        f'{variable_prefix}_sum_mean', f'{variable_prefix}_sum_min', f'{variable_prefix}_sum_max',
                        f'{variable_prefix}_{pred}_diff_mean', f'{variable_prefix}_{pred}_diff_min', f'{variable_prefix}_{pred}_diff_max']
    return df[required_columns]

# Function to fill data after 2023-08-31 with monthly statistics
def fill_data(df, monthly_stats, pred):
    future_data = df[df['utctime'] > '2023-08-31'].copy()
    for month in range(1, 13):
        future_data.loc[future_data['month'] == month, f'{pred}_mean'] = monthly_stats.loc[month, 'mean']
        future_data.loc[future_data['month'] == month, f'{pred}_min'] = monthly_stats.loc[month, 'min']
        future_data.loc[future_data['month'] == month, f'{pred}_max'] = monthly_stats.loc[month, 'max']
    return future_data

def adjust_future(df, monthly_stats, pred, variable_prefix):
    future_data = df[df['utctime'] > '2023-08-31'].copy()
    past_data = df[df['utctime'] <= '2023-08-31'].copy()

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
df = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_oceanids_ece3-Bremerhaven-2000-2100.csv')
pred = 'TP_PT24H_SUM'

# Define the correlation mappings
correlation_mappings = {
    'WG_PT24H_MAX': 'sfcWind',
    'WS_PT24H_AVG': 'sfcWind',
    'TN_PT24H_MIN': 'tasmin',
    'TX_PT24H_MAX': 'tasmax',
    'TP_PT24H_SUM': 'pr'
}

# Convert 'utctime' to datetime and extract year and month
df['utctime'] = pd.to_datetime(df['utctime'])
df['year'] = df['utctime'].dt.year
df['month'] = df['utctime'].dt.month

# Calculate monthly statistics for the chosen pred up to 2023-08-31
monthly_stats = df[df['utctime'] <= '2023-08-31'].groupby('month')[pred].agg(['mean', 'min', 'max'])

# Fill data after 2023-08-31 with monthly statistics
future_data = fill_data(df, monthly_stats, pred)

# Split the original DataFrame into two parts: up to 2023-08-31 and after 2023-08-31
df_past = df[df['utctime'] <= '2023-08-31']
df_future = future_data

# Process the variable group based on the pred for data up to 2023-08-31
processed_df_past = process_variable_group(df_past, pred)

# Process the variable group based on the pred for data after 2023-08-31, excluding pred stats
processed_df_future = process_variable_group(df_future, pred, include_pred_stats=False)

# Combine the processed data up to 2023-08-31 with the processed future data
combined_df = pd.concat([processed_df_past, processed_df_future], ignore_index=True)

adjusted_future = adjust_future(combined_df, monthly_stats, pred, correlation_mappings[pred])
past = combined_df[combined_df['utctime'] <= '2023-08-31']

combined_df = pd.concat([past, adjusted_future], ignore_index=True)

# Save the updated DataFrame to a new CSV file
combined_df.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_oceanids_ece3-Bremerhaven-{pred}-2000-2100.csv', index=False)
print(combined_df)