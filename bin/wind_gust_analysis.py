import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def analyze_wg_distribution(obs_file_path, era5_file_path, ecxsf_file_path, output_file_path):
    global ecxsf_df_corrected  # Define as global to access in other functions
    # Read the CSV files
    obs_df = pd.read_csv(obs_file_path)
    era5_df = pd.read_csv(era5_file_path)
    ecxsf_df = pd.read_csv(ecxsf_file_path)
    
    # Ensure 'utctime' column exists
    if 'valid_time' in ecxsf_df.columns:
        ecxsf_df['valid_time'] = pd.to_datetime(ecxsf_df['valid_time'])
        ecxsf_df['utctime'] = ecxsf_df['valid_time'] - pd.Timedelta(hours=1)
    else:
        print("Column 'valid_time' not found in the ECXSF dataset.")
    
    if 'utctime' in obs_df.columns:
        obs_df['utctime'] = pd.to_datetime(obs_df['utctime'])
    else:
        print("Column 'utctime' not found in the observation dataset.")
    
    # Create a figure for all distributions
    plt.figure(figsize=(12, 8))
    
    # Analyze WG_PT24H columns from ECXSF file first to put them in the back
    wg_pt24h_columns = [col for col in ecxsf_df.columns if col.startswith('WG_PT24H')]
    
    # Define colors for different groups
    colors = {
        '0': 'black',
        '1-10': 'gray',
        '11-20': 'dimgray',
        '21-30': 'darkgray',
        '31-40': 'silver',
        '41-50': 'lightgray'
    }
    
    for col in wg_pt24h_columns:
        member_number = int(col.split('_')[-1])
        if member_number == 0:
            color = colors['0']
            sns.kdeplot(data=ecxsf_df, x=col, label=col, linewidth=1.5, color=color)
        else:
            color = colors[f'{(member_number - 1) // 10 * 10 + 1}-{(member_number - 1) // 10 * 10 + 10}']
            sns.kdeplot(data=ecxsf_df, x=col, linewidth=1.5, color=color)
    
    # List of fg10 columns
    fg10_columns = ['fg10-1', 'fg10-2', 'fg10-3', 'fg10-4']
    
    # Calculate a single correction factor
    if 'utctime' in obs_df.columns and 'utctime' in ecxsf_df.columns:
        obs_mean = obs_df['WG_PT24H_MAX'].mean()
        ecxsf_mean = ecxsf_df[wg_pt24h_columns].mean().mean()
        correction_factor = obs_mean / ecxsf_mean if ecxsf_mean != 0 else 1
        
        print(f"Correction Factor: {correction_factor}")
        
        # Apply correction to ECXSF data
        ecxsf_df_corrected = ecxsf_df.copy()
        for col in wg_pt24h_columns:
            ecxsf_df_corrected[col] *= correction_factor
        
        # Plot corrected ECXSF data
        for col in wg_pt24h_columns:
            member_number = int(col.split('_')[-1])
            if member_number == 0:
                color = colors['0']
                sns.kdeplot(data=ecxsf_df_corrected, x=col, label=f'{col}_corrected', linewidth=1.5, color=color, linestyle='--')
            else:
                color = colors[f'{(member_number - 1) // 10 * 10 + 1}-{(member_number - 1) // 10 * 10 + 10}']
                sns.kdeplot(data=ecxsf_df_corrected, x=col, linewidth=1.5, color=color, linestyle='--')
    
    # Plot ERA5 distribution
    for col in fg10_columns:
        if col in era5_df.columns:
            sns.kdeplot(data=era5_df, x=col, label=col, linewidth=1.5)
        else:
            print(f"Column {col} not found in the ERA5 dataset.")
    
    # Plot observation distribution
    if 'WG_PT24H_MAX' in obs_df.columns:
        sns.kdeplot(data=obs_df, x='WG_PT24H_MAX', label='WG_PT24H_MAX', color='black', linewidth=2.5)
    else:
        print("Column WG_PT24H_MAX not found in the observation dataset.")
    
    # Customize the plot
    plt.title('Distribution of Wind Gust Speeds')
    plt.xlabel('Wind Gust Speed (m/s)')
    plt.ylabel('Density')
    plt.legend(labels=['WG_PT24H_MAX', 'fg10-1', 'fg10-2', 'fg10-3', 'fg10-4', 'WG_PT24H_0', 'WG_PT24H_0_corrected'])
    plt.xlim(0, 30)
    
    # Save the plot
    plt.savefig(output_file_path)
    plt.close()

def plot_corrected_ecxsf(ecxsf_df_corrected, output_file_path_corrected):
    plt.figure(figsize=(12, 8))
    
    # Define colors for different groups
    colors = {
        '0': 'black',
        '1-10': 'gray',
        '11-20': 'dimgray',
        '21-30': 'darkgray',
        '31-40': 'silver',
        '41-50': 'lightgray'
    }
    
    # First plot observations
    obs_df = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/{location_suffix}/obs-oceanids-20130701T000000Z-20240101T000000Z-{location_suffix}.csv')
    if 'WG_PT24H_MAX' in obs_df.columns:
        sns.kdeplot(data=obs_df, x='WG_PT24H_MAX', label='Observations', color='red', linewidth=2.5)
    
    wg_pt24h_columns = [col for col in ecxsf_df_corrected.columns if col.startswith('WG_PT24H')]
    
    for col in wg_pt24h_columns:
        member_number = int(col.split('_')[-1])
        if member_number == 0:
            color = colors['0']
            sns.kdeplot(data=ecxsf_df_corrected, x=col, label=f'{col}_corrected', linewidth=1.5, color=color, linestyle='--')
        else:
            color = colors[f'{(member_number - 1) // 10 * 10 + 1}-{(member_number - 1) // 10 * 10 + 10}']
            sns.kdeplot(data=ecxsf_df_corrected, x=col, linewidth=1.5, color=color, linestyle='--')
    
    # Customize the plot
    plt.title('Corrected Distribution of Wind Gust Speeds')
    plt.xlabel('Wind Gust Speed (m/s)')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0, 30)
    
    # Save the plot
    plt.savefig(output_file_path_corrected)
    plt.close()

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python wind_gust_analysis.py <location_suffix>")
        sys.exit(1)

    location_suffix = sys.argv[1]
    location_folder = location_suffix.split('_')[0]

    analyze_wg_distribution(
        f'/home/ubuntu/data/ML/training-data/OCEANIDS/{location_folder}/obs-oceanids-20130701T000000Z-20240101T000000Z-{location_folder}.csv',
        f'/home/ubuntu/data/ML/training-data/OCEANIDS/{location_folder}/era5_oceanids_fg10_20130701T000000Z-20240101T000000Z_{location_folder}.csv',
        f'/home/ubuntu/data/ML/ECXSF_202501_WG_PT24H_MAX_{location_suffix}.csv',
        f'/home/ubuntu/data/ML/wind_gust_distribution_{location_suffix}.png'
    )
    
    # Assuming ecxsf_df_corrected is available after calling analyze_wg_distribution
    plot_corrected_ecxsf(ecxsf_df_corrected, f'/home/ubuntu/data/ML/wind_gust_distribution_corrected_{location_suffix}.png')
