import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.spatial import Delaunay  # Add this import
from cartopy.feature import GSHHSFeature  # Add this import for detailed coastlines
import cartopy.io.img_tiles as cimgt  # Add this import



def extract_lat_lon_columns(df):
    """Extracts latitude and longitude columns from a dataframe."""
    lat_columns = sorted([col for col in df.columns if col.startswith("lat-")])
    lon_columns = sorted([col for col in df.columns if col.startswith("lon-")])
    return lat_columns, lon_columns

def plot_harbor_points(csv_dir, output_file, loc, harbor_coords=None):
    """Plots harbor prediction points from multiple CSVs, grouped by shared grids."""
    with open('harbors_config.json', 'r') as f:
        harbors = json.load(f)
    station_lat = harbors[loc]['latitude']
    station_lon = harbors[loc]['longitude']
    
    # Extract start and end years from config
    start_year = harbors[loc]['start'][:4]
    end_year = harbors[loc]['end'][:4]
    
    # Define model groups based on shared grids with full names - move MOHC to group1
    model_groups = {
        'aladin': ['cnrm_cerfacs_cm5-cnrm_aladin63'],
        'group1': [
            'cnrm_cerfacs_cm5-knmi_racmo22e',
            'ncc_noresm1_m-dmi_hirham5',
            'ncc_noresm1_m-smhi_rca4',
            'mohc_hadgem2_es-dmi_hirham5',
            'mohc_hadgem2_es-knmi_racmo22e',
            'mohc_hadgem2_es-smhi_rca4'
        ],
        'sf': ['sf']
    }

    # Colors for each grid pattern - updated color scheme
    grid_colors = {
        'aladin': '#FFA500',   # orange
        'group1': '#FF69B4',   # hot pink
        'sf': '#9B59B6'        # purple
    }

    # Create OSM tile source
    osm_tiles = cimgt.OSM()
    
    # Create figure with OSM projection
    fig, ax = plt.subplots(figsize=(12, 8), 
                          subplot_kw={'projection': osm_tiles.crs})
    
    # Set map extent around the observation station
    ax.set_extent([
        station_lon - 0.5, station_lon + 0.5,
        station_lat - 0.25, station_lat + 0.25
    ], crs=ccrs.PlateCarree())
    
    # Add OSM tiles
    ax.add_image(osm_tiles, 13)  # Zoom level 13 provides good detail
    
    # Plot observation station location
    ax.plot(station_lon, station_lat, marker='*', color='red', 
            markersize=15, label='Observation Station', zorder=10,
            transform=ccrs.PlateCarree())
    
    # Plot harbor location if coordinates are provided
    if harbor_coords:
        harbor_lon, harbor_lat = harbor_coords
        ax.plot(harbor_lon, harbor_lat, marker='$\u2693$', color='darkred',  # Unicode anchor symbol
                markersize=15, label='Harbor', zorder=10,
                transform=ccrs.PlateCarree())
    
    # Adjust zorder to ensure visibility
    point_zorder = {
        'group1': 7,    # yellow points on top
        'aladin': 6,    # orange points next
        'sf': 4         # purple points at bottom
    }

    # Plot points for each grid pattern
    for grid_name in reversed(list(model_groups.keys())):
        models = model_groups[grid_name]
        try:
            if grid_name == 'sf':
                pattern = f"{csv_dir}training_data_oceanids_{loc}-sf_*-WG_PT24H_MAX.csv"
            else:
                pattern = f"{csv_dir}training_data_oceanids_{models[0]}_{loc}_WG_PT24H_MAX*.csv"
            
            # Add debug print for troubleshooting SF grid
            print(f"Looking for file: {pattern}")
            
            matching_files = glob.glob(pattern)
            if not matching_files:
                raise FileNotFoundError(f"No files found matching pattern: {pattern}")
            
            csv_file = matching_files[0]
            df = pd.read_csv(csv_file)
            lat_cols, lon_cols = extract_lat_lon_columns(df)
            
            # Get points as arrays
            lats = df[lat_cols].values[0]
            lons = df[lon_cols].values[0]
            
            # Plot grid pattern
            if grid_name == 'sf':
                # Special case for SF's regular grid
                for i in range(len(lat_cols)):
                    ax.plot(df[lon_cols].values[0], 
                           df[lat_cols[i]].values[0] * np.ones_like(df[lon_cols].values[0]),
                           color=grid_colors[grid_name], alpha=0.6,  # Increased from 0.4
                           linestyle=':', linewidth=2, zorder=4,
                           transform=ccrs.PlateCarree())
                    ax.plot(df[lon_cols[i]].values[0] * np.ones_like(df[lat_cols].values[0]), 
                           df[lat_cols].values[0],
                           color=grid_colors[grid_name], alpha=0.6,  # Increased from 0.4
                           linestyle=':', linewidth=2, zorder=4,
                           transform=ccrs.PlateCarree())
            else:
                # Create triangulated mesh for other grids
                points = np.column_stack((lons, lats))
                tri = Delaunay(points)
                edges = set()
                
                for triangle in tri.simplices:
                    for i in range(3):
                        p1, p2 = sorted([triangle[i], triangle[((i + 1) % 3)]])
                        # Skip the diagonal line from point 1 to 4
                        if ((p1 == 0 and p2 == 3) or  # 1 to 4
                            (p1 == 3 and p2 == 0)):   # 4 to 1
                            continue
                        if (p1, p2) not in edges:
                            edges.add((p1, p2))
                            ax.plot([points[p1,0], points[p2,0]], 
                                   [points[p1,1], points[p2,1]],
                                   color=grid_colors[grid_name], 
                                   alpha=0.7,  # Increased from 0.5
                                   linestyle=':', linewidth=2.5, zorder=4,
                                   transform=ccrs.PlateCarree())
            
            # Plot points without white outline and fix legend markers
            for model in models:
                if model == models[0]:
                    # Use marker='o' to avoid the dash through symbols
                    ax.scatter(df[lon_cols], df[lat_cols], 
                             color=grid_colors[grid_name],
                             label=model, s=70, alpha=1.0,
                             marker='o',  # Explicitly set marker
                             zorder=point_zorder[grid_name],
                             transform=ccrs.PlateCarree())
                else:
                    # Add legend entry without plotting points
                    ax.scatter([], [], color=grid_colors[grid_name],
                             label=model, s=70, alpha=1.0,
                             marker='o')  # Match marker style
        
        except (FileNotFoundError, IndexError) as e:
            print(f"Warning: Could not process grid {grid_name}: {str(e)}")
            continue

    # Update legend style to ensure clean markers
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
             title='Models', frameon=True, framealpha=1,
             scatterpoints=1, # Use single point for legend
             markerscale=1.5) # Adjust marker size in legend
    
    plt.title(f"Model grid points for {loc} Harbor")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

# Modified example usage:
if len(sys.argv) < 2:
    print("Usage: python plot-oceanids-multimodel.py <location> [harbor_lon harbor_lat]")
    sys.exit(1)

loc = sys.argv[1]
harbor_coords = None
if len(sys.argv) == 4:
    try:
        harbor_coords = (float(sys.argv[2]), float(sys.argv[3]))
    except ValueError:
        print("Error: Harbor coordinates must be valid numbers")
        sys.exit(1)

csv_directory = "/home/ubuntu/data/ML/training-data/OCEANIDS/"
output_file = f"../{loc}_points.png"

plot_harbor_points(csv_directory, output_file, loc, harbor_coords)
