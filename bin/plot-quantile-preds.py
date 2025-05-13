import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    # ...existing code...
    base_file = "/home/ubuntu/data/ML/results/OCEANIDS/Raahe/ECXSF_202503_TA_PT24H_MIN_Raahe.csv"
    quantile_high_file = "/home/ubuntu/data/ML/results/OCEANIDS/Raahe/ECXSF_202503_TA_PT24H_MIN_Raahe-0.98.csv"
    quantile_low_file = "/home/ubuntu/data/ML/results/OCEANIDS/Raahe/ECXSF_202503_TA_PT24H_MIN_Raahe-0.02.csv"
    training_data_file = "/home/ubuntu/data/ML/training-data/OCEANIDS/Raahe/training_data_oceanids_Raahe-sf-addpreds.csv"
    
    # helper function to load CSV and return ensemble DataFrame and date range
    def load_ensemble_data(file_path):
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['utctime'])
            time_range = (df['utctime'].min(), df['utctime'].max())
            return df.drop('utctime', axis=1), time_range
        else:
            print(f"File {file_path} not found!")
            return None, None

    ensemble_base, base_time_range = load_ensemble_data(base_file)
    ensemble_high, high_time_range = load_ensemble_data(quantile_high_file)
    ensemble_low, low_time_range = load_ensemble_data(quantile_low_file)

    df = pd.read_csv(training_data_file, parse_dates=['utctime'])
    train_time_range = (df['utctime'].min(), df['utctime'].max())
    df = df[['mx2t-1', 'mx2t-2', 'mx2t-3', 'mx2t-4', 'TA_PT24H_MIN']]
    
    # Format time ranges
    def format_range(time_range):
        if time_range is None:
            return "Unknown"
        return f"{time_range[0].strftime('%Y-%m-%d')} to {time_range[1].strftime('%Y-%m-%d')}"
    
    base_range_str = format_range(base_time_range)
    high_range_str = format_range(high_time_range)
    low_range_str = format_range(low_time_range)
    train_range_str = format_range(train_time_range)
    
    plt.figure(figsize=(12,6))
    
    # define specific colors for each ensemble, training predictions, and observation
    colors = {
        'base': 'blue',
        'quantile_high': 'orange',
        'quantile_low': 'green',
        'training': 'red',
        'observation': 'purple'
    }
    
    ax = plt.gca()
    
    # Plot individual ensemble members as density plots (KDE) with low opacity lines (without legend labels)
    if ensemble_base is not None:
        for col in ensemble_base.columns:
            pd.Series(ensemble_base[col]).dropna().plot.kde(ax=ax, color=colors['base'], alpha=0.1)
    if ensemble_high is not None:
        for col in ensemble_high.columns:
            pd.Series(ensemble_high[col]).dropna().plot.kde(ax=ax, color=colors['quantile_high'], alpha=0.1)
    if ensemble_low is not None:
        for col in ensemble_low.columns:
            pd.Series(ensemble_low[col]).dropna().plot.kde(ax=ax, color=colors['quantile_low'], alpha=0.1)
    
    # Plot training predictions and observation data separately
    training_label_added = False
    for col in df.columns:
        if col == 'TA_PT24H_MIN':
            # Plot observation with its own color
            pd.Series(df[col]).dropna().plot.kde(ax=ax, color=colors['observation'], lw=2, alpha=0.8, label='Observation')
        else:
            # Plot training predictions; add the label only once.
            label = 'Training Ensemble' if not training_label_added else None
            pd.Series(df[col]).dropna().plot.kde(ax=ax, color=colors['training'], lw=2, alpha=0.8, label=label)
            training_label_added = True

    # Create custom legend handles with colors corresponding to each file/series
    legend_handles = [
        Line2D([0], [0], color=colors['base'], lw=2, 
               label=f'Base Ensemble ({base_range_str})'),
        Line2D([0], [0], color=colors['quantile_high'], lw=2, 
               label=f'Quantile 0.98 Ensemble ({high_range_str})'),
        Line2D([0], [0], color=colors['quantile_low'], lw=2, 
               label=f'Quantile 0.02 Ensemble ({low_range_str})'),
        Line2D([0], [0], color=colors['training'], lw=2, 
               label=f'Training Ensemble ({train_range_str})'),
        Line2D([0], [0], color=colors['observation'], lw=2, 
               label=f'Observation ({train_range_str})')
    ]
    
    plt.xlabel("TA_PT24H_MIN Value")
    plt.ylabel("Density")
    plt.title("Raahe Ensemble Prediction Density Plot")
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    
    # Set x-axis limits to observation min and MIN values
    observation_data = df['TA_PT24H_MIN'].dropna()
    obs_min = observation_data.min()
    obs_max = observation_data.max()
    ax.set_xlim(obs_min, obs_max+10)

    plt.tight_layout()
    plt.savefig("Raahe_TA_PT24H_MIN_density.png")
    plt.show()
    
if __name__ == '__main__':
    main()
