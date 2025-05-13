#!/usr/bin/env python3
"""
Script to analyze CORDEX dataset issues, particularly investigating 
why the daily dataset has different row counts and date patterns.
"""

import sys
import pandas as pd
import numpy as np
import os
from collections import Counter

def reset_date_range(location, model):
    """Analyze the daily dataset to understand structure and potential issues."""
    file_path = f"/home/ubuntu/data/cordex/rcp85/Vuosaari/{model}_daily_grid2x2_Helsinki_{location}.csv"
    
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    # Clear the 'date' column by resetting the index and dropping it
    df.reset_index(drop=True, inplace=True)
    # Generate a complete date range from 2006-01-01 to 2101-01-01 with 4 entries per day
    full_date_range = pd.date_range(start="2006-01-01", end="2100-12-31", freq="D").repeat(4)
    full_date_range = full_date_range[~((full_date_range.month == 2) & (full_date_range.day == 29))]
    
    # Ensure the dataframe has the full date range
    df = df.set_index(full_date_range).reset_index()
    df.rename(columns={"index": "time"}, inplace=True)

    # Remove all data from the year 2093 except for the first day
    df = df[~((df['time'].dt.year == 2093) & (df['time'].dt.dayofyear != 1))]
    
    # Duplicate the rows for 2100-12-31 for 2101-01-01
    last_day = df[df['time'] == "2100-12-31"]
    if not last_day.empty:
        duplicate_day = last_day.copy()
        duplicate_day['time'] = pd.to_datetime("2101-01-01")
        df = pd.concat([df, duplicate_day], ignore_index=True)

    # Save the modified dataframe back to the file
    print(df.tail())
    df.to_csv(f"/home/ubuntu/data/cordex/rcp85/Vuosaari/{model}_daily_grid2x2_Helsinki_{location}-TEST.csv", index=False)

def analyze(location, model):
    file_path = f"/home/ubuntu/data/cordex/rcp85/Vuosaari/{model}_daily_grid2x2_Helsinki_{location}-TEST.csv"
    file2_path = f"/home/ubuntu/data/cordex/rcp85/Vuosaari/{model}_MaxWindSpeed_daily_grid2x2_Helsinki_{location}.csv"
    
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    df2 = pd.read_csv(file2_path, parse_dates=['time'], index_col='time')

    # Compare the date ranges of both dataframes
    print(f"Rows in {file_path}: {len(df)}")
    print(f"Rows in {file2_path}: {len(df2)}")




def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_cordex_dataset.py <location> <model>")
        sys.exit(1)
    
    location = sys.argv[1]
    model = sys.argv[2]
    
    reset_date_range(location, model)
    analyze(location, model)

if __name__ == "__main__":
    main()
