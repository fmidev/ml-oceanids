import pandas as pd

# Read the CSV file as a DataFrame
add_file_path = '/home/ubuntu/data/synop/hnms/limnos-obs-relhum-2012-2024.csv'
main_file_path = '/home/ubuntu/data/synop/hnms/limnos-obs-relhum-2000-2011.csv'

main_df = pd.read_csv(main_file_path)
add_df = pd.read_csv(add_file_path)

# Append the relevant columns from add_df to main_df
main_df = pd.concat([main_df, add_df[['timestamp', 'station', 'measurement']]], ignore_index=True)
# Fill 'type' and 'unit' columns with static values from main_df
main_df['type'] = main_df['type'].fillna(method='ffill')
main_df['unit'] = main_df['unit'].fillna(method='ffill')

# Save the updated DataFrame to a new CSV file
output_file_path = '/home/ubuntu/data/synop/hnms/limnos-obs-relhum-2000-2024.csv'
main_df.to_csv(output_file_path, index=False)