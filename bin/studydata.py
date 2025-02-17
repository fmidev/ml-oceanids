import pandas as pd

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/Vuosaari/'

#cols=['utctime','fg10-1','fg10-2','fg10-3','fg10-4']
df_era5_name='era5_oceanids_fg10_20130701T000000Z-20240101T000000Z_Vuosaari.csv'
df_era5d_name='era5d-fg10/era5_oceanids_fg10_20130701T000000Z-20240101T000000Z_Vuosaari.csv'

df_era5=pd.read_csv(data_dir+df_era5_name)#,usecols=cols)
df_era5d=pd.read_csv(data_dir+df_era5d_name)#,usecols=cols)

print(df_era5)
#print(df_era5d)

#df_era5d['utctime'] = pd.to_datetime(df_era5d['utctime'])

# Specify the columns to shift (all forecast columns)
cols_to_shift = ['fg10-1', 'fg10-2', 'fg10-3', 'fg10-4']

# Shift the data downwards by one (so each row gets the previous day's values)
df_era5d[cols_to_shift] = df_era5d[cols_to_shift].shift(1)

print(df_era5d)
