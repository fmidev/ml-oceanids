import sys
import os
import glob
import pandas as pd
import sklearn.metrics
import numpy as np

# Only location is required as input now
if len(sys.argv) < 2:
    print("Usage: python calculate-metrics-oceanids.py <location>")
    sys.exit(1)

location = sys.argv[1]

# Folder where prediction files are stored
pred_folder = f'/home/ubuntu/data/ML/results/OCEANIDS/cordex/{location}/'

# Glob for all predictions files in that folder
pred_files = glob.glob(os.path.join(pred_folder, "prediction_*.csv"))
print(f"Found {len(pred_files)} prediction files in {pred_folder}")
if len(pred_files) == 0:
    print(f"Warning: No prediction files found in {pred_folder}")

results_list = []
for file_path in pred_files:
    file_name = os.path.basename(file_path)
    print(f"\nProcessing file: {file_name}")
    parts = file_name.split('_')
    
    # Parse filename: prediction_cordex_rcp45_Malaga_TA_PT24H_MAX_cnrm_cerfacs_cm5-cnrm_aladin63.csv
    if parts[0] == "prediction" and len(parts) >= 5:
        # Extract scenario from the 3rd part (index 2)
        scenario = parts[2]
        
        # Extract the location - should be the 4th part (index 3)
        file_location = parts[3]
        
        if file_location != location:
            print(f"  Skipping: File location {file_location} does not match requested location {location}")
            continue
            
        print(f"  Detected scenario: {scenario}")
        
        # Extract variable name (like TA_PT24H_MAX) - should be in position 4 and may include 5 if it has underscores
        variable = parts[4]
        if len(parts) > 5 and parts[5] in ["PT24H", "MAX", "MIN", "AVG", "ACC"]:
            variable = f"{variable}_{parts[5]}"
            if len(parts) > 6 and parts[6] in ["MAX", "MIN", "AVG", "ACC"]:
                variable = f"{variable}_{parts[6]}"
                model_start_idx = 7
            else:
                model_start_idx = 6
        else:
            model_start_idx = 5
            
        # Extract model information - all remaining parts
        model = '_'.join(parts[model_start_idx:]).replace('.csv', '')
        
        print(f"  Parsed: Location={file_location}, Variable={variable}, Model={model}")
        
        # Read and process the data
        data = pd.read_csv(file_path).dropna()
        print(f"  Data loaded with {len(data)} rows and columns: {data.columns.tolist()}")
        
        # For this data format, the observed column is likely the variable itself (e.g., TA_PT24H_MIN)
        observed_col = variable
        
        # If the exact variable name isn't a column, look for it as part of column names
        if observed_col not in data.columns:
            for col in data.columns:
                if variable in col:
                    observed_col = col
                    break
        
        # If we still couldn't find it, try the second column (in the example, TA_PT24H_MIN is column 1)
        if observed_col not in data.columns and len(data.columns) > 1:
            observed_col = data.columns[1]  # Try second column as fallback
        
        # Make sure the Predicted column is properly detected (case sensitive)
        pred_col = 'Predicted'  # Your sample shows capital P
        
        print(f"  Using columns: Observed={observed_col}, Predicted={pred_col}")
        
        # Skip files with missing columns
        if observed_col not in data.columns:
            print(f"Warning: Could not find observed column for {variable} in {file_path}. Skipping.")
            continue
            
        if pred_col not in data.columns:
            print(f"Warning: Could not find Predicted column in {file_path}. Skipping.")
            continue
            
        # Calculate error metrics
        data['error'] = data[pred_col] - data[observed_col]
        
        print(f"  Sample data - Observed: {data[observed_col].head(3).tolist()}, Predicted: {data[pred_col].head(3).tolist()}")
        print(f"  Sample errors: {data['error'].head(3).tolist()}")
        
        mean_error = data['error'].mean()
        mae = sklearn.metrics.mean_absolute_error(data[observed_col], data[pred_col])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(data[observed_col], data[pred_col]))
        r2 = sklearn.metrics.r2_score(data[observed_col], data[pred_col])
        variance = sklearn.metrics.explained_variance_score(data[observed_col], data[pred_col])
        
        print(f"  Metrics calculated - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        results_list.append({
            'Location': location,
            'Scenario': scenario,
            'Variable': variable,
            'Model': model,
            'Mean_Error': mean_error,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Explained_Variance': variance
        })

print(f"\nCollected metrics for {len(results_list)} files")

# Create a DataFrame with the metrics for all predictions
results = pd.DataFrame(results_list)

# Sort the results by Scenario first, then Variable, and reset the index
results = results.sort_values(['Scenario', 'Variable', 'Model']).reset_index(drop=True)

# Create metrics directory if it doesn't exist
os.makedirs('/home/ubuntu/data/ML/results/OCEANIDS/metrics/', exist_ok=True)

# Save the combined results
output_file = f'/home/ubuntu/data/ML/results/OCEANIDS/metrics/{location}-cordex-metrics.csv'
results.to_csv(output_file, index=False)

print(results)
print(f'Results saved to {output_file}')