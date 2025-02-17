import pandas as pd
import sys

def add_observations(data_path, obs_path, output_path, obs_columns):
    # Read both datasets
    data_df = pd.read_csv(data_path)
    obs_df = pd.read_csv(obs_path)
    obs_df.rename(columns={'date': 'utctime'}, inplace=True)
    
    # Convert dates to datetime for proper matching
    data_df['utctime'] = pd.to_datetime(data_df['utctime'])
    obs_df['utctime'] = pd.to_datetime(obs_df['utctime'])
    
    # List of observation columns to add
    
    # Create merged dataframe by matching on dates
    merged_df = pd.merge(
        data_df,
        obs_df[['utctime'] + obs_columns],
        on='utctime',
        how='left'
    )
    
    # Save to output file
    merged_df.to_csv(output_path, index=False)
    
    print(f"Added observation columns from {obs_path} to {data_path}")
    print(f"Saved result to {output_path}")
    
if __name__ == "__main__":
    data_path = "/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_Rauma-sf_2013-2023.csv"
    obs_path = "/home/ubuntu/data/synop/fmisid101061_Rauma-Kylmäpihlaja_2000-2024_rh_24h.csv"
    output_path = "/home/ubuntu/data/ML/training-data/OCEANIDS/training_data_oceanids_Rauma-sf_2013-2023_with_rh.csv"
    
    add_observations(data_path, obs_path, output_path, obs_columns=['RH_PT24H_AVG'])