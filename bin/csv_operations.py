import pandas as pd

# Load the datasets
file1 = "/home/ubuntu/data/synop/fmisid115857_PraiaDaVittoria2012-2024_ws-wd-wg_3h.csv"

# Read the CSV file
df1 = pd.read_csv(file1, usecols=['time', 'WG_PT10M_MAX'])

# Convert the 'time' column to datetime
df1['time'] = pd.to_datetime(df1['time'])
df1.rename(columns={'time': 'utctime'}, inplace=True)

# Set the 'utctime' column as the index
df1.set_index('utctime', inplace=True)

# Resample to daily frequency and calculate sum values, preserving NaNs
daily_sum = df1['WG_PT10M_MAX'].resample('D').max()
#daily_sum_min = df1['TA_PT12H_MIN'].resample('D').min()
#daily_sum = pd.concat([daily_sum_max, daily_sum_min], axis=1)

# Reset the index to include the 'utctime' column
daily_sum = daily_sum.reset_index()

# Save the modified DataFrame back to a CSV file
daily_sum.to_csv('/home/ubuntu/data/synop/wmo-praiadavittoria_wg_2012-2024.csv', index=False)



