import pandas as pd
import sys, json
import numpy as np  # added for ERA5 predictors

harbor_name = sys.argv[1]

data_dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
fname = f'training_data_oceanids_{harbor_name}-sf.csv'
output_file = f'training_data_oceanids_{harbor_name}-sf-addpreds.csv'
df = pd.read_csv(data_dir+fname)

# add day of year and hour of day as columns
df['utctime'] = pd.to_datetime(df['utctime'])
df['dayOfYear'] = df['utctime'].dt.dayofyear
df['year'] = df['utctime'].dt.year
df['month'] = df['utctime'].dt.month
df['hour'] = df['utctime'].dt.hour

# --- ERA5 Predictors: Wind Speed, Wind Direction, and Relative Humidity ---
for i in range(1, 5):
    # Wind Speed and Direction per measurement index
    u_00 = df[f'u10-00-{i}']
    u_12 = df[f'u10-12-{i}']
    v_00 = df[f'v10-00-{i}']
    v_12 = df[f'v10-12-{i}']
    
    # Compute the average for u and v
    u_avg = (u_00 + u_12) / 2
    v_avg = (v_00 + v_12) / 2
    
    # Calculate wind speed and meteorological wind direction
    df[f'ws-{i}'] = np.sqrt(u_avg**2 + v_avg**2)
    df[f'wd-{i}'] = (270 - np.degrees(np.arctan2(v_avg, u_avg))) % 360

    # Relative Humidity from 2m dewpoint and 2m temperature per measurement index
    td2_00 = df[f'td2-00-{i}']
    td2_12 = df[f'td2-12-{i}']
    t2_00 = df[f't2-00-{i}']
    t2_12 = df[f't2-12-{i}']
    
    # Compute average dewpoint and temperature
    dew_avg = (td2_00 + td2_12) / 2
    temp_avg = (t2_00 + t2_12) / 2
    
    # Calculate relative humidity using the standard formula
    df[f'rh-{i}'] = 100 * (np.exp((17.625 * dew_avg) / (243.04 + dew_avg)) / 
                           np.exp((17.625 * temp_avg) / (243.04 + temp_avg)))

# add pressure change from previous day to predictors
for i in range(1, 5):  # For columns 1 to 4
    df[f'Dmsl-00-{i}'] = df[f'msl-00-{i}'].diff()
    df[f'Dmsl-12-{i}'] = df[f'msl-12-{i}'].diff()

# predictand mappings from JSON file
with open('predictand_mappings.json', 'r') as f:
    predictand_mappings = json.load(f)

# Initialize climatology DataFrame for 2020
df_2020_combined = df[df['year'] == 2020][['utctime', 'month']].copy()

for predictand, mapping in predictand_mappings.items():
    variable_prefix = mapping["parameter"]
    df[f'{variable_prefix}_sum'] = df[[f'{variable_prefix}-1', f'{variable_prefix}-2', f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4
    # Group by year and month and calculate the average, minimum, and maximum for each month
    monthly_stats = df.groupby(['year', 'month']).agg({
        f'{variable_prefix}_sum': ['mean', 'min', 'max'],
        predictand: ['mean', 'min', 'max']
    })
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
    monthly_stats.reset_index(inplace=True)
    # Calculate the difference between the summed variable and the predictor
    monthly_stats[f'{variable_prefix}_{predictand}_diff_mean'] = monthly_stats[f'{variable_prefix}_sum_mean'] - monthly_stats[f'{predictand}_mean']
    monthly_stats[f'{variable_prefix}_{predictand}_diff_min'] = monthly_stats[f'{variable_prefix}_sum_min'] - monthly_stats[f'{predictand}_min']
    monthly_stats[f'{variable_prefix}_{predictand}_diff_max'] = monthly_stats[f'{variable_prefix}_sum_max'] - monthly_stats[f'{predictand}_max']
    df = df.merge(monthly_stats, on=['year', 'month'], how='left')

    # Climatology file for SF prediction
    monthly_climatology = df.groupby('month')[[f'{predictand}_mean', f'{predictand}_max', f'{predictand}_min']].mean().reset_index()
    monthly_climatology.rename(columns={
        f'{predictand}_mean': f'{predictand}_mean_climatology',
        f'{predictand}_max': f'{predictand}_max_climatology',
        f'{predictand}_min': f'{predictand}_min_climatology'
    }, inplace=True)
    df = df.merge(monthly_climatology, on='month', how='left')
    df_2020 = df[df['year'] == 2020][['utctime', 'month']]
    df_2020 = df_2020.merge(monthly_climatology, on='month', how='left')
    df_2020 = df_2020[['utctime', f'{predictand}_mean_climatology', f'{predictand}_max_climatology', f'{predictand}_min_climatology']]
    df_2020_combined = df_2020_combined.merge(df_2020, on='utctime', how='left')
df_2020_combined.drop(columns=['month'], inplace=True)

df.to_csv(f'{data_dir}{output_file}', index=False)
df_2020_combined.to_csv(f'{data_dir}training_data_oceanids_{harbor_name}-sf_2020-clim.csv', index=False)