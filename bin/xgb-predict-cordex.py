import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import matplotlib.pyplot as plt
import time, warnings, json, os, commentjson
from datetime import datetime
warnings.filterwarnings("ignore")

startTime = time.time()

# Get command line arguments
if len(sys.argv) < 3:
    print("Usage: python xgb-predict-cordex.py <harbor_name> <pred> [model_name]")
    sys.exit(1)

scenario= sys.argv[1]
harbor_name = sys.argv[2]
pred = sys.argv[3]
model = sys.argv[4]

# Define mapping between new and old variable names
var_name_map = {
    'TA_PT24H_MIN': 'TN_PT24H_MIN',
    'TA_PT24H_MAX': 'TX_PT24H_MAX',
    'TP_PT24H_ACC': 'TP_PT24H_SUM'
}

# Get old variable name if needed for the model filename
old_pred = var_name_map.get(pred, pred)

# Load harbor config
with open('harbors_config.json', 'r') as file:
    config = json.load(file)
harbor = config.get(harbor_name, {})
starty = harbor.get('start')
prediction_endy = harbor.get('end')

# Define directories
mod_dir = f'/home/ubuntu/data/ML/models/OCEANIDS/cordex/{harbor_name}/'
pred_dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{harbor_name}/'
out_dir = f'/home/ubuntu/data/ML/results/OCEANIDS/cordex/{harbor_name}/'

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Define filenames
input_file = f'prediction_data_oceanids_{harbor_name}_cordex_{scenario}_{model}.csv'
# Use old variable name in model filename
mdl_name = f'mdl_{harbor_name}_{pred}_{model}_xgb_cordex_{scenario}_oceanids-QE.json'

print(f"Harbor: {harbor_name}, Prediction: {pred} (Model uses: {old_pred}), Model: {model}")
print(f"Loading prediction data from: {input_file}")
print(f"Using model: {mdl_name}")
print(f"XGBoost version: {xgb.__version__}")  # Print XGBoost version for debugging

# Load the prediction data (which already contains observations)
df_data = pd.read_csv(pred_dir + input_file, parse_dates=['utctime'])

# Check if file exists
model_path = mod_dir + mdl_name
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

# Create result dataframe with timestamp and observations
df_result = pd.DataFrame({'utctime': df_data['utctime']})
df_result[pred] = df_data[pred]  # Observations are already in the file

if pred == 'WG_PT24H_MAX':
    df_result["maxWind_sum_mean"] = df_data["maxWind_sum_mean"]
    df_result["maxWind_sum_max"] = df_data["maxWind_sum_max"]
    df_result["maxWind_sum_min"] = df_data["maxWind_sum_min"] 
elif pred == 'WS_PT24H_AVG':
    df_result["sfcWind_sum_mean"] = df_data["sfcWind_sum_mean"]
    df_result["sfcWind_sum_max"] = df_data["sfcWind_sum_max"]
    df_result["sfcWind_sum_min"] = df_data["sfcWind_sum_min"]
elif pred == 'TA_PT24H_MIN':
    df_result["tasmin_sum_mean"] = df_data["tasmin_sum_mean"] - 273.15
    df_result["tasmin_sum_max"] = df_data["tasmin_sum_max"] - 273.15
    df_result["tasmin_sum_min"] = df_data["tasmin_sum_min"] - 273.15
elif pred == 'TA_PT24H_MAX':
    df_result["tasmax_sum_mean"] = df_data["tasmax_sum_mean"] - 273.15
    df_result["tasmax_sum_max"] = df_data["tasmax_sum_max"] - 273.15
    df_result["tasmax_sum_min"] = df_data["tasmax_sum_min"] - 273.15
elif pred == 'TP_PT24H_ACC':
    df_result["pr_sum_mean"] = df_data["pr_sum_mean"] * 86400
    df_result["pr_sum_max"] = df_data["pr_sum_max"] * 86400
    df_result["pr_sum_min"] = df_data["pr_sum_min"] * 86400
else:
    raise ValueError("Invalid predictor")
    
# Load the model with proper handling of gzipped files
fitted_mdl = xgb.XGBRegressor()
try:
    if model_path.endswith('.txt.gz'):
        import gzip
        import tempfile
        
        print(f"Loading gzipped model from {model_path}")
        # Create a temporary file to store decompressed model (keeping .txt extension)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_path = temp_file.name
            
        # Decompress the gzipped file
        with gzip.open(model_path, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                f_out.write(f_in.read())
                
        # Load from the decompressed temporary file
        fitted_mdl.load_model(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        print(f"Successfully loaded model after decompression")
    else:
        # Regular model loading
        fitted_mdl.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("This might be due to compression issues or model format incompatibility")
    sys.exit(1)

# Rename columns to match old naming convention expected by the model
if pred in var_name_map:
    # Create a copy of the DataFrame to avoid modifying the original
    df_pred = df_data.copy()
    
    # Replace new variable name with old in column names
    old_name = var_name_map[pred]
    rename_dict = {}
    
    # Check for columns that need renaming
    for col in df_pred.columns:
        if pred in col:
            new_col = col.replace(pred, old_name)
            rename_dict[col] = new_col
    
    # Apply renaming if needed
    if rename_dict:
        print(f"Renaming columns to match model expectations: {rename_dict}")
        df_pred = df_pred.rename(columns=rename_dict)
else:
    df_pred = df_data

# Ensure the DataFrame has the correct columns
required_columns = fitted_mdl.get_booster().feature_names
if required_columns is None:
    # Manually specify the feature names if they are not available
    required_columns = df_pred.columns.tolist()
    print("Feature names not found in the model. Using DataFrame columns as feature names.")
else:
    print("Required columns:", required_columns)

# Check for dayofyear vs dayOfYear naming inconsistency
if 'dayofyear' in required_columns and 'dayofyear' not in df_pred.columns and 'dayOfYear' in df_pred.columns:
    print("Detected column name mismatch: 'dayofyear' required but 'dayOfYear' found. Renaming column.")
    df_pred = df_pred.rename(columns={'dayOfYear': 'dayofyear'})

# Make sure the DataFrame has all required columns
missing_cols = [col for col in required_columns if col not in df_pred.columns]
if missing_cols:
    print(f"Warning: Missing required columns: {missing_cols}")
    print("Available columns:", df_pred.columns.tolist())
    sys.exit(1)

df_pred = df_pred[required_columns]

# XGBoost predict without DMatrix
result = fitted_mdl.predict(df_pred)
result = result.tolist()
df_result['Predicted'] = result

# Use the new variable name in the output filename for consistency
df_result.to_csv(f'{out_dir}prediction_cordex_{scenario}_{harbor_name}_{pred}_{model}.csv', index=False)

print(df_result)
