import sys
import os
import glob
import pandas as pd
import sklearn.metrics

# Input is now the location name
location = sys.argv[1]

# Folder where prediction files are stored
pred_folder = f'/home/ubuntu/data/ML/results/OCEANIDS/{location}/'

# Glob for all predictions files in that folder
pred_files = glob.glob(os.path.join(pred_folder, "predictions_*.csv"))

results_list = []
for file_path in pred_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    # Expected format: predictions_<location>_<pred_name>_<...>.csv
    # Verify location in file name
    if parts[0] == "predictions" and parts[1] == location:
        # Combine all parts from index 2 up to the part before extension for the prediction name
        pred_name = '_'.join(parts[2:-1])
        
        # Read and process the data
        data = pd.read_csv(file_path).dropna()
        data['error'] = data['predicted'] - data['observed']
        
        mean_error = data['error'].mean()
        mae = sklearn.metrics.mean_absolute_error(data['observed'], data['predicted'])
        r2 = sklearn.metrics.r2_score(data['observed'], data['predicted'])
        variance = sklearn.metrics.explained_variance_score(data['observed'], data['predicted'])
        
        results_list.append({
            'Location': location,
            'Prediction': pred_name,
            'Mean_Error': mean_error,
            'MAE': mae,
            'R2': r2,
            'Explained_Variance': variance
        })

# Create a DataFrame with the metrics for all predictions
results = pd.DataFrame(results_list)

# Save the combined results
output_file = f'/home/ubuntu/data/ML/results/OCEANIDS/metrics/{location}-metrics.csv'
results.to_csv(output_file, index=False)

print(results)
print(f'Results saved to {output_file}')