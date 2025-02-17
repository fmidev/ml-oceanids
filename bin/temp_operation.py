import pandas as pd

hourly_csv = '/home/ubuntu/data/synop/fmisid106787_Malaga-puerto_2018-2024_ws-wd-wg-rh_3h.csv'

# Read and process hourly data
hourly_df = pd.read_csv(hourly_csv, usecols=['time', 'RH_PT1M_AVG'])
hourly_df.rename(columns={'time': 'date'}, inplace=True)
hourly_df['date'] = pd.to_datetime(hourly_df['date']).dt.floor('D')  # Convert to daily
hourly_daily = hourly_df.groupby('date')['RH_PT1M_AVG'].mean().reset_index()
hourly_daily.rename(columns={'RH_PT1M_AVG': 'RH_PT24H_AVG'}, inplace=True)

# Create complete date range
date_range = pd.date_range(start=hourly_daily['date'].min(),
                          end=hourly_daily['date'].max(),
                          freq='D')

# Create a new DataFrame with all dates
complete_daily = pd.DataFrame({'date': date_range})

# Merge with the averaged data
hourly_daily = pd.merge(complete_daily, hourly_daily, on='date', how='left')

print(hourly_daily)
hourly_daily.to_csv('/home/ubuntu/data/synop/fmisid106787_Malaga-puerto_2018-2024_rh_24h.csv', index=False)