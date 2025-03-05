import pandas as pd
import sys, json
# Script to add predictors to the training dataset for OCEANIDS
# (AK 2025)

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
    # Group by 'month' and calculate climatology (mean) for the specified columns
    monthly_climatology = df.groupby('month')[[f'{predictand}_mean', f'{predictand}_max', f'{predictand}_min']].mean().reset_index()
    # Rename the columns
    monthly_climatology.rename(columns={
        f'{predictand}_mean': f'{predictand}_mean_climatology',
        f'{predictand}_max': f'{predictand}_max_climatology',
        f'{predictand}_min': f'{predictand}_min_climatology'
    }, inplace=True)
    # Merge the climatology data back into the original DataFrame
    df = df.merge(monthly_climatology, on='month', how='left')
    # Filter the DataFrame for the year 2020 to create a climatology csv for seasonal forecast prediction
    df_2020 = df[df['year'] == 2020][['utctime', 'month']]
    df_2020 = df_2020.merge(monthly_climatology, on='month', how='left')
    df_2020 = df_2020[['utctime', f'{predictand}_mean_climatology', f'{predictand}_max_climatology', f'{predictand}_min_climatology']]
    df_2020_combined = df_2020_combined.merge(df_2020, on='utctime', how='left')
df_2020_combined.drop(columns=['month'], inplace=True)

df.to_csv(f'{data_dir}{output_file}', index=False)
df_2020_combined.to_csv(f'{data_dir}training_data_oceanids_{harbor_name}-sf_2020-clim.csv', index=False)