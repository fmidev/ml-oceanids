import pandas as pd
import numpy as np

def transform_6hourly_to_daily(df):
    # Read the 6-hourly CSV file
    df_daily = df.groupby([df['date'].dt.date, 'lat', 'lon'])[['uas', 'vas']].mean().reset_index()
        # Convert date column back to datetime for consistency
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    return df_daily

def add_wind_speed(df):
    # Compute daily wind speed
    df["sfcWind"] = np.sqrt(df["uas"]**2 + df["vas"]**2)
    return df