import pandas as pd
import matplotlib.pyplot as plt
import calendar
import os,sys

harbor_name = sys.argv[1]
predictand = sys.argv[2]

# EOBS
file1 = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/obs-{harbor_name}-{predictand}_EOBS.csv'
df_eobs = pd.read_csv(file1, parse_dates=['utctime'])
df_eobs.set_index('utctime', inplace=True)

# synop 
file2 = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/obs-oceanids-{harbor_name}.csv'
df_synop = pd.read_csv(file2, parse_dates=['utctime'])
df_synop.set_index('utctime', inplace=True)
if predictand in df_synop.columns:
    ts_synop = df_synop[predictand]
else:
    raise ValueError("Column not found")

# Read ERA5D data if file exists
era5d_file = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/obs-{harbor_name}-{predictand}_ERA5D.csv'
if os.path.exists(era5d_file):
    df_era5d = pd.read_csv(era5d_file, parse_dates=['utctime'])
    df_era5d.set_index('utctime', inplace=True)
    ts_era5d = df_era5d[f'{predictand}_ERA5D']
else:
    raise FileNotFoundError(f"ERA5D file not found: {era5d_file}")

# --- Plot 1: synop vs EOBS ---
plt.figure(figsize=(12,6))
plt.plot(ts_synop.index, ts_synop, label='synop')
plt.plot(df_eobs.index, df_eobs[f'{predictand}_EOBS'], label='EOBS', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Predictand')
plt.title(f'{harbor_name} {predictand}: synop vs EOBS')
plt.legend()

common_eobs = pd.concat([
    ts_synop.rename("synop"),
    df_eobs[f'{predictand}_EOBS'].rename("EOBS")
], axis=1, join='outer')
common_eobs = common_eobs.dropna()  # ignore rows where any value is NaN for MAE calc
common_eobs['abs_error'] = (common_eobs['synop'] - common_eobs['EOBS']).abs()
mae_eobs = common_eobs['abs_error'].mean()
print("Overall MAE (synop vs EOBS):", mae_eobs)

plt.text(0.05, 0.95, f"synop vs EOBS MAE: {mae_eobs:.2f}",
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top',
         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
plt.tight_layout()
plt.savefig(f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/{harbor_name}-compare-{predictand}-synop_vs_eobs.png')

# Monthly MAE for synop vs EOBS
monthly_mae_eobs = common_eobs.groupby(common_eobs.index.month)['abs_error'].mean()
for month, mae in monthly_mae_eobs.items():
    print(f"{calendar.month_name[month]} MAE (synop vs EOBS): {mae}")

# --- Plot 2: synop vs ERA5D ---
plt.figure(figsize=(12,6))
plt.plot(ts_synop.index, ts_synop, label='synop', )
plt.plot(ts_era5d.index, ts_era5d, label='ERA5D', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Predictand')
plt.title(f'{harbor_name} {predictand}: synop vs ERA5D')
plt.legend()

common_era5d = pd.concat([
    ts_synop.rename("synop"),
    ts_era5d.rename("ERA5D")
], axis=1, join='outer')
common_era5d = common_era5d.dropna()  # ignore rows where any value is NaN for MAE calc
common_era5d['abs_error'] = (common_era5d['synop'] - common_era5d['ERA5D']).abs()
mae_era5d = common_era5d['abs_error'].mean()
print("Overall MAE (synop vs ERA5D):", mae_era5d)

plt.text(0.05, 0.95, f"synop vs ERA5D MAE: {mae_era5d:.2f}",
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top',
         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
plt.tight_layout()
plt.savefig(f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/{harbor_name}-compare-{predictand}-synop_vs_era5d.png')

# Monthly MAE for synop vs ERA5D
monthly_mae_era5d = common_era5d.groupby(common_era5d.index.month)['abs_error'].mean()
for month, mae in monthly_mae_era5d.items():
    print(f"{calendar.month_name[month]} MAE (synop vs ERA5D): {mae}")