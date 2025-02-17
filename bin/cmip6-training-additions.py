import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_Raahe-sf_2013-2023.csv')
pred = 'TX_PT24H_MAX'

# Define the correlation mappings
correlation_mappings = {
    'WG_PT24H_MAX': 'fg10',
    'WS_PT24H_AVG': '',
    'TN_PT24H_MIN': 'mn2t',
    'TX_PT24H_MAX': 'mx2t',
    'TP_PT24H_SUM': 'tp'
}

# Convert 'utctime' to datetime and extract year and month
df['utctime'] = pd.to_datetime(df['utctime'])
df['year'] = df['utctime'].dt.year
df['month'] = df['utctime'].dt.month
#df['mn2t-1'] = df['mn2t-1'] - 273.15
#df['mn2t-2'] = df['mn2t-2'] - 273.15
#df['mn2t-3'] = df['mn2t-3'] - 273.15
#df['mn2t-4'] = df['mn2t-4'] - 273.15
#df['mx2t-1'] = df['mx2t-1'] - 273.15
#df['mx2t-2'] = df['mx2t-2'] - 273.15
#df['mx2t-3'] = df['mx2t-3'] - 273.15
#df['mx2t-4'] = df['mx2t-4'] - 273.15

# Function to process the variable group based on the pred
def process_variable_group(pred):
    # Find the variable group that correlates with the given pred
    variable_prefix = None
    for key, value in correlation_mappings.items():
        if pred in key:
            variable_prefix = value
            break

    if variable_prefix is None:
        raise ValueError(f'No variable group found for the predictor: {pred}')

    # Sum the values for columns like 'sfcWind-1' to 'sfcWind-4' into a single column
    df[f'{variable_prefix}_sum'] = df[[f'{variable_prefix}-1', f'{variable_prefix}-2', f'{variable_prefix}-3', f'{variable_prefix}-4']].sum(axis=1) / 4

    # Group by year and month and calculate the average, minimum, and maximum for each month
    monthly_stats = df.groupby(['year', 'month']).agg({
        f'{variable_prefix}_sum': ['mean', 'min', 'max'],
        pred: ['mean', 'min', 'max']
    })

    # Flatten the MultiIndex columns
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]

    # Reset index to make 'year' and 'month' columns again
    monthly_stats.reset_index(inplace=True)

    # Calculate the difference between the summed variable and the predictor
    monthly_stats[f'{variable_prefix}_{pred}_diff_mean'] = monthly_stats[f'{variable_prefix}_sum_mean'] - monthly_stats[f'{pred}_mean']
    monthly_stats[f'{variable_prefix}_{pred}_diff_min'] = monthly_stats[f'{variable_prefix}_sum_min'] - monthly_stats[f'{pred}_min']
    monthly_stats[f'{variable_prefix}_{pred}_diff_max'] = monthly_stats[f'{variable_prefix}_sum_max'] - monthly_stats[f'{pred}_max']

    # Merge the calculated statistics back to the original DataFrame
    return df.merge(monthly_stats, on=['year', 'month'], how='left')

# Process the variable group based on the pred
df = process_variable_group(pred)

# Save the updated DataFrame to a new CSV file
df.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_Raahe-sf_2013-2023-{pred}.csv', index=False)
print(df)