import pandas as pd
import sys

loc = sys.argv[1]
dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{loc}/'

input_file = dir+f'training_data_oceanids_{loc}-sf.csv'
output_file = input_file

# Read CSV (assumes first row is header)
df = pd.read_csv(input_file)

# Identify temperature columns (for example, columns with "TA" in their name)
temp_cols = [col for col in df.columns if "TA" in col]

# Convert values in these columns from Celsius to Kelvin
for col in temp_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') + 273.15

# Round the converted values to three decimals
df[temp_cols] = df[temp_cols].round(3)

df.to_csv(output_file, index=False)
print(f"Converted file saved as {output_file}")