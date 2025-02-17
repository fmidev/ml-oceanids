import xarray as xr
import pandas as pd
import sys

# Open the NetCDF file
file=sys.argv[1]
filepath=f'/home/ubuntu/data/cordex/{file}'

ds = xr.open_dataset(filepath)

# Convert to a DataFrame
df = ds.to_dataframe()

# Reset index (to flatten multi-index columns)
df.reset_index(inplace=True)

# Save to CSV
df.to_csv(filepath[:-3]+".csv", index=False)

print(f"Conversion complete! Saved as {filepath[:-3]}.csv")
