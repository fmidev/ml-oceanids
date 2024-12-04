import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-Bremerhaven.csv')
pred = 'TX_PT24H_MAX'

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
df.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-Bremerhaven-{pred}.csv', index=False)
print(df)