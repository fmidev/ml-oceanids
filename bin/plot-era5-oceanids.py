import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pandas as pd
import json,sys,os

request = cimgt.OSM()

harbor_name=sys.argv[1]

# Load harbor config file
with open('harbors_config.json', 'r') as file1:
    config = json.load(file1)
harbor = config.get(harbor_name, {})
latitude = harbor.get('latitude')
longitude = harbor.get('longitude')

# training data to get lat lon points
res_dir=f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file=f'{harbor_name}_training-locs.png'
data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
fname=f'training_data_oceanids_{harbor_name}-sf-addpreds.csv'
cols=['latitude','longitude','lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4']
df=pd.read_csv(data_dir+fname,usecols=cols)

# Read the first row of the dataframe to get lat/lon points
first_row = df.iloc[0]
lat1, lon1 = first_row['lat-1'], first_row['lon-1']
lat2, lon2 = first_row['lat-2'], first_row['lon-2']
lat3, lon3 = first_row['lat-3'], first_row['lon-3']
lat4, lon4 = first_row['lat-4'], first_row['lon-4']

# Adjust the figure size and layout to reduce white space
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=request.crs)
extent = [min(lon1, lon2, lon3, lon4, longitude) - 0.1, max(lon1, lon2, lon3, lon4, longitude) + 0.1,
          min(lat1, lat2, lat3, lat4, latitude) - 0.1, max(lat1, lat2, lat3, lat4, latitude) + 0.1]
ax.set_extent(extent)
ax.add_image(request, 13)    # 13 = zoom level

# Plot training locations and harbor with smaller points and border
plt.scatter(lon1, lat1, transform=ccrs.PlateCarree(), color='yellow', edgecolor='black', s=50, label=f'1 ({lon1}, {lat1})')
plt.scatter(lon2, lat2, transform=ccrs.PlateCarree(), color='green', edgecolor='black', s=50, label=f'2 ({lon2}, {lat2})')
plt.scatter(lon3, lat3, transform=ccrs.PlateCarree(), color='blue', edgecolor='black', s=50, label=f'3 ({lon3}, {lat3})')
plt.scatter(lon4, lat4, transform=ccrs.PlateCarree(), color='orange', edgecolor='black', s=50, label=f'4 ({lon4}, {lat4})')
plt.scatter(longitude, latitude, transform=ccrs.PlateCarree(), color='red', edgecolor='black', s=75, label=f'obs ({round(longitude, 2)}, {round(latitude, 2)})')

# Replace the old legend placement with a lower right placement on top of the map:
plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), facecolor="lightblue", framealpha=0.7)
plt.title(f'{harbor_name}', fontsize=16)

# Save plot with tight layout to reduce white space
plt.tight_layout()
plt.savefig(res_dir+res_file, dpi=200, bbox_inches='tight', pad_inches=0)