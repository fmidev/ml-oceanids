#!/usr/bin/env python3
# filepath: /home/ubuntu/ml-oceanids/bin/greece-obs-csv-converter.py
import pandas as pd
import argparse
from pathlib import Path
import io

def read_csv_with_comments(file_path, comment_prefix='//'):
    """Read CSV file and skip lines starting with comment prefix."""
    with open(file_path, 'r') as f:
        # Skip comment lines starting with the prefix
        lines = [line for line in f if not line.strip().startswith(comment_prefix)]
    
    # Create a file-like object from the filtered lines
    data = io.StringIO(''.join(lines))
    
    # Read the CSV content
    return pd.read_csv(data)

def convert_to_oceanids_format(humidity_file, temp_file, wind_file, output_file):
    """
    Convert Greek observation data to match Oceanids format.
    
    Args:
        humidity_file (str): Path to humidity CSV file
        temp_file (str): Path to temperature CSV file (contains both max and min)
        wind_file (str): Path to wind speed CSV file
        output_file (str): Path where the converted file will be saved
    """
    # Read input files, skipping comment lines
    humidity_df = read_csv_with_comments(humidity_file)
    temp_df = read_csv_with_comments(temp_file)
    wind_df = read_csv_with_comments(wind_file)
    
    # Filter by type
    humidity_data = humidity_df[humidity_df['type'] == 'humidity'][['timestamp', 'station', 'measurement']]
    humidity_data.rename(columns={'measurement': 'RH_PT24H_AVG'}, inplace=True)
    
    # Extract max and min temperature from the same file
    max_temp_data = temp_df[temp_df['type'] == 'max_temperature'][['timestamp', 'station', 'measurement']]
    max_temp_data.rename(columns={'measurement': 'TA_PT24H_MAX'}, inplace=True)
    
    min_temp_data = temp_df[temp_df['type'] == 'min_temperature'][['timestamp', 'station', 'measurement']]
    min_temp_data.rename(columns={'measurement': 'TA_PT24H_MIN'}, inplace=True)
    
    wind_data = wind_df[wind_df['type'] == 'wind_speed'][['timestamp', 'station', 'measurement']]
    wind_data.rename(columns={'measurement': 'WS_PT24H_AVG'}, inplace=True)
    
    # Merge data frames on timestamp and station
    merged_data = pd.merge(humidity_data, max_temp_data, on=['timestamp', 'station'], how='outer')
    merged_data = pd.merge(merged_data, min_temp_data, on=['timestamp', 'station'], how='outer')
    merged_data = pd.merge(merged_data, wind_data, on=['timestamp', 'station'], how='outer')
    
    # Convert timestamp to datetime format
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])
    
    # Sort by timestamp
    merged_data = merged_data.sort_values(by='timestamp')
    
    # Add coordinates for all Greek stations
    station_coords = {
        'Chios': (38.3453, 26.142),
        'Heraklion': (35.3353, 25.182),
        'Kastelli': (35.1888, 25.3289),
        'Kerkyra': (39.6081, 19.9139),
        'Limnos': (39.9246, 25.2442),
        'Milos': (36.7386, 24.4293),
        'Mytilene': (39.0541, 26.6038),
        'Mykonos': (37.4359, 25.3458),
        'Naxos': (37.1013, 25.3728),
        'Palaiochora': (35.2253, 23.6757),
        'Saint_Eustratios': (39.5417, 24.9894),
        'Siteia': (35.2156, 26.1029),
    }
    
    # Add latitude and longitude columns
    merged_data['latitude'] = merged_data['station'].map(lambda x: station_coords.get(x, (0, 0))[0])
    merged_data['longitude'] = merged_data['station'].map(lambda x: station_coords.get(x, (0, 0))[1])
    merged_data['name'] = merged_data['station']
    
    # Convert Celsius to Kelvin for temperature values to match Oceanids format
    if 'TA_PT24H_MAX' in merged_data.columns:
        merged_data['TA_PT24H_MAX'] = merged_data['TA_PT24H_MAX'] + 273.15
    
    if 'TA_PT24H_MIN' in merged_data.columns:
        merged_data['TA_PT24H_MIN'] = merged_data['TA_PT24H_MIN'] + 273.15
    
    # Convert wind speed from knots to m/s if present
    if 'WS_PT24H_AVG' in merged_data.columns:
        # 1 knot = 0.514444 m/s
        merged_data['WS_PT24H_AVG'] = merged_data['WS_PT24H_AVG'] * 0.514444
    
    # Create empty columns for precipitation and wind direction to match format
    merged_data['TP_PT24H_ACC'] = float('nan')  # Precipitation
    merged_data['WD_PT24H_AVG'] = float('nan')  # Wind direction
    merged_data['WG_PT24H_MAX'] = float('nan')  # Wind gust
    
    # Reorder columns to match Oceanids format
    result = merged_data[['timestamp', 'latitude', 'longitude', 'name', 
                          'WG_PT24H_MAX', 'TA_PT24H_MAX', 'TA_PT24H_MIN', 
                          'TP_PT24H_ACC', 'RH_PT24H_AVG', 'WS_PT24H_AVG', 'WD_PT24H_AVG']]
    
    # Rename timestamp to utctime to match Oceanids format
    result = result.rename(columns={'timestamp': 'utctime'})
    
    # Save the result
    print(f"Converting files to Oceanids format...")
    result.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Successfully created {output_file} with {len(result)} records")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Greek data to Oceanids format')
    parser.add_argument('--humidity', required=True, help='Input humidity CSV file')
    parser.add_argument('--temp', required=True, help='Input temperature CSV file (contains both max and min)')
    parser.add_argument('--wind', required=True, help='Input wind speed CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    convert_to_oceanids_format(
        args.humidity, 
        args.temp, 
        args.wind, 
        args.output
    )