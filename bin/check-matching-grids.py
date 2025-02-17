import pandas as pd
import numpy as np
from itertools import combinations
from math import radians, sin, cos, sqrt, asin

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_grid_differences(grid1, grid2):
    """Calculate statistics about differences between two grids."""
    lats1, lons1 = grid1
    lats2, lons2 = grid2
    
    # Find closest point distances
    distances = []
    for lat1, lon1 in zip(lats1, lons1):
        min_dist = float('inf')
        for lat2, lon2 in zip(lats2, lons2):
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    return {
        'min_dist': np.min(distances),
        'max_dist': np.max(distances),
        'avg_dist': np.mean(distances),
        'std_dist': np.std(distances)
    }

def compare_model_grids(harbor):
    """Compare grid points between all models for a given harbor."""
    base_path = f"/home/ubuntu/data/ML/training-data/OCEANIDS/"
    
    # List of all models
    models = [
        'cnrm_cerfacs_cm5-cnrm_aladin63',
        'cnrm_cerfacs_cm5-knmi_racmo22e',
        'mohc_hadgem2_es-dmi_hirham5',
        'mohc_hadgem2_es-knmi_racmo22e',
        'mohc_hadgem2_es-smhi_rca4',
        'ncc_noresm1_m-dmi_hirham5',
        'ncc_noresm1_m-smhi_rca4'
    ]
    
    # Store coordinates for each model
    model_coords = {}
    
    # Load coordinates for each model
    for model in models:
        filename = f"{base_path}/training_data_oceanids_{model}_{harbor}_WG_PT24H_MAX_2013-2024.csv"
        try:
            df = pd.read_csv(filename)
            # Get lat/lon columns
            lats = df[[col for col in df.columns if col.startswith('lat-')]].values[0]
            lons = df[[col for col in df.columns if col.startswith('lon-')]].values[0]
            model_coords[model] = (lats, lons)
        except FileNotFoundError:
            print(f"Warning: File not found for model {model}")
            continue

    # Compare all pairs of models
    print(f"\nGrid comparison for {harbor}:")
    print("-" * 50)
    
    for model1, model2 in combinations(model_coords.keys(), 2):
        grid1 = model_coords[model1]
        grid2 = model_coords[model2]
        
        # Check if grids are identical
        if (np.array_equal(grid1[0], grid2[0]) and 
            np.array_equal(grid1[1], grid2[1])):
            print(f"\n{model1} and {model2} have IDENTICAL grids")
        else:
            # Calculate differences
            diff_stats = calculate_grid_differences(grid1, grid2)
            print(f"\n{model1} vs {model2}:")
            print(f"  Minimum distance between points: {diff_stats['min_dist']:.2f} km")
            print(f"  Maximum distance between points: {diff_stats['max_dist']:.2f} km")
            print(f"  Average distance between points: {diff_stats['avg_dist']:.2f} km")
            print(f"  Standard deviation: {diff_stats['std_dist']:.2f} km")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python check-matching-grids.py <harbor_name>")
        sys.exit(1)
        
    compare_model_grids(sys.argv[1])
