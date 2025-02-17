import pandas as pd

pred = 'WG_PT24H_MAX'
harbor='Raahe'
starty='2013'

# Read the CSV file into a DataFrame
df = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_{harbor}-sf_{starty}-2023-{pred}.csv')

# Group by 'month' and calculate climatology (mean) for the specified columns
monthly_climatology = df.groupby('month')[[f'{pred}_mean', f'{pred}_max', f'{pred}_min']].mean().reset_index()

# Rename the columns
monthly_climatology.rename(columns={
    f'{pred}_mean': f'{pred}_mean_climatology',
    f'{pred}_max': f'{pred}_max_climatology',
    f'{pred}_min': f'{pred}_min_climatology'
}, inplace=True)

# Merge the climatology data back into the original DataFrame
df = df.merge(monthly_climatology, on='month', how='left')

print(df)

# Filter the DataFrame for the year 2014 to create a climatology csv for seasonal forecast prediction
df_2014 = df[df['year'] == 2014][['utctime', 'month']]
df_2014 = df_2014.merge(monthly_climatology, on='month', how='left')
df_2014 = df_2014[['utctime', f'{pred}_mean_climatology', f'{pred}_max_climatology', f'{pred}_min_climatology']]
print(df_2014)
df_2014.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_{harbor}-sf_2014-{pred}-clim.csv', index=False)
