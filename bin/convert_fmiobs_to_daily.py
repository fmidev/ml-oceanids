import pandas as pd

# Load the datasets
file_indicator = '106787_Malaga-puerto_2018-2024'
file1 = f"/home/ubuntu/data/synop/fmisid{file_indicator}_ws-wd-wgx_3h.csv"
file2 = f"/home/ubuntu/data/synop/fmisid{file_indicator}_tanx-pra_12h.csv"

# Read the CSV files
df1 = pd.read_csv(file1, usecols=['time', 'WG_PT1H_MAX'])
df2 = pd.read_csv(file2, usecols=['time', 'TA_PT12H_MAX', 'TA_PT12H_MIN', 'PRA_PT12H_ACC'])

# Convert the 'time' column to datetime
df1['time'] = pd.to_datetime(df1['time'])
df2['time'] = pd.to_datetime(df2['time'])

# Set the 'time' column as the index
df1.set_index('time', inplace=True)
df2.set_index('time', inplace=True)

# Resample to daily frequency and calculate the required values
daily_wg_max = df1['WG_PT1H_MAX'].resample('D').max()
daily_ta_max = df2['TA_PT12H_MAX'].resample('D').max()
daily_ta_min = df2['TA_PT12H_MIN'].resample('D').min()
daily_pra_sum = df2['PRA_PT12H_ACC'].resample('D').sum(min_count=1)

# Combine the daily values into a single DataFrame
daily_df = pd.concat([daily_ta_max, daily_ta_min, daily_pra_sum, daily_wg_max], axis=1)
daily_df.columns = ['TX_PT24H_MAX', 'TN_PT24H_MIN', 'TP_PT24H_ACC', 'WG_PT24H_MAX']

# Reset the index to include the 'time' column
daily_df = daily_df.reset_index()

# Rename the 'time' column to 'utctime'
daily_df.rename(columns={'time': 'utctime'}, inplace=True)

# Save the modified DataFrame back to a CSV file
daily_df.to_csv(f'/home/ubuntu/data/synop/fmisid{file_indicator}_tn-tx-tp-wg_24h.csv', index=False)



