import pandas as pd
import numpy as np
import argparse

def dms_to_decimal(dms_str):
    # Remove any whitespace
    dms_str = dms_str.strip()
    # Determine the sign from the first character
    sign = -1 if dms_str.startswith('-') else 1
    # Remove the sign for splitting
    dms_str = dms_str.lstrip('+-')
    parts = dms_str.split(':')
    if len(parts) != 3:
        return None
    degrees, minutes, seconds = map(float, parts)
    return sign * (degrees + minutes/60 + seconds/3600)

# New helper function to retrieve station details from stations.txt
def get_station_details(station_id):
    station_id = int(station_id)  # Remove leading zeros by converting to int
    details_file = "/home/ubuntu/data/eobs/rr_blend/stations.txt"
    with open(details_file, 'r') as f:
        # Skip header lines until the column header is found.
        for line in f:
            if line.strip().startswith("STAID"):
                break
        # Now process remaining lines
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            # Compare station id (convert to string and strip spaces)
            if int(parts[0]) == station_id:
                # parts: [STAID, STANAME, CN, LAT, LON, HGHT]
                # Convert DMS to decimal for latitude and longitude
                lat_decimal = dms_to_decimal(parts[3])
                lon_decimal = dms_to_decimal(parts[4])
                return parts[1], lat_decimal, lon_decimal
    return None, None, None

def process_eobs(loc, station_id):
    # Define observation variables and their standardized names.
    # Modify or add predictors as needed.
    obs_preds = {
        'FG': 'WG_PT24H_MAX',
        'RR': 'TP_PT24H_ACC',
        'TN': 'TA_PT24H_MIN',
        'TX': 'TA_PT24H_MAX'
    }
    # Initialize main observations DataFrame.
    obs_df = None
    for key, col_name in obs_preds.items():
        file_path = f"/home/ubuntu/data/eobs/{key.lower()}_blend/{key}_STAID{station_id}.txt"
        dates = []
        values = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 4:
                        continue
                    # Skip header lines where the fourth column equals the key (ignoring case)
                    if parts[3].strip().upper() == key.upper():
                        continue
                    dates.append(parts[2].strip())  # Expecting date in YYYYMMDD format.
                    try:
                        value = np.nan if parts[3].strip() == '-9999' else float(parts[3].strip()) / 10
                    except ValueError:
                        value = np.nan
                    values.append(value)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        df_new = pd.DataFrame({
            'utctime': pd.to_datetime(dates, format='%Y%m%d', errors='coerce'),
            col_name: values
        })
        if obs_df is None:
            obs_df = df_new
        else:
            obs_df = obs_df.merge(df_new, on='utctime', how='outer')
    if obs_df is not None:
        obs_df = obs_df.sort_values('utctime').reset_index(drop=True)
    return obs_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process EOBS observation files to CSV")
    parser.add_argument('--loc', required=True, help="Location name for the output file")
    parser.add_argument('--id', required=True, help="Station ID")
    # Removed lat, lon, name arguments.
    args = parser.parse_args()
    
    # Obtain station details from stations.txt
    station_name, latitude, longitude = get_station_details(args.id)
    
    obs_data = process_eobs(args.loc, args.id)
    if obs_data is not None:
        obs_data = obs_data.sort_values('utctime').reset_index(drop=True)
        obs_data.insert(1, 'latitude', latitude)
        obs_data.insert(2, 'longitude', longitude)
        obs_data.insert(3, 'name', station_name)
        output_file = f"/home/ubuntu/data/ML/training-data/OCEANIDS/{args.loc}/obs-oceanids-{args.loc}.csv"
        obs_data.to_csv(output_file, index=False)
        print(f"Saved observation data to {output_file}")
    else:
        print("No observation data was processed.")
