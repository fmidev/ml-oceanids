import pandas as pd

# Load the datasets
file1 = "/home/ubuntu/data/ML/training-data/OCEANIDS/prediction_data_oceanids_ece3-Bremerhaven-2000-2100.csv" 
file2 = "/home/ubuntu/data/ML/training-data/OCEANIDS/ece3-Bremerhaven.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# List of columns to replace
columns_to_replace = ['WS_PT24H_AVG', 'TP_PT24H_SUM', 'WG_PT24H_MAX', 'TN_PT24H_MIN', 'TX_PT24H_MAX']

# Replace the columns in df1 with the ones from df2
for column in columns_to_replace:
    df1[column] = df2[column]

# Save the modified df1 back to a CSV file
df1.to_csv(file1, index=False)



