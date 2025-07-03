import time, warnings, sys, json, os, commentjson
import pandas as pd
import xgboost as xgb
from datetime import datetime
warnings.filterwarnings("ignore")

startTime = time.time()

# Get command line arguments
harbor_name = sys.argv[1]
pred = sys.argv[2]

# Load harbor config
with open('harbors_config.json', 'r') as file1:
    config = json.load(file1)
harbor = config.get(harbor_name, {})
start = harbor.get('start')
end = harbor.get('end')

# Define directories
data_dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
mod_dir = f'/home/ubuntu/data/ML/models/OCEANIDS/{harbor_name}/'
out_dir = f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/'

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Define filenames
train_fname = f'training_data_oceanids_{harbor_name}-sf-addpreds.csv.gz'
mod_name = f'mdl_{harbor_name}_{pred}_xgb_era5_oceanids-QE.json'
out_fname = f'predictions_{harbor_name}_{pred}_era5.csv'

# Load predictand mappings
with open('predictand_mappings.json', 'r') as f:
    mappings = json.load(f)
predictand_mappings = {key: mappings[key]["parameter"] for key in mappings}

# Get columns to drop
selected_value = predictand_mappings[pred]
keys_to_drop = [key for key in predictand_mappings if key != pred]
values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]

# Load training data config to select proper training columns
with open('training_data_config.json', 'r') as file:
    config_data = commentjson.load(file)
columns = config_data['training_columns']
#print("Training config columns:", columns)

# Filter the columns for predictor and related variables:
filtered_columns = []
for col in columns:
    drop = any(drop_key in col for drop_key in keys_to_drop)
    drop = drop or any(drop_val in col for drop_val in values_to_drop)
    if not drop:
        filtered_columns.append(col)

# Read training data using the filtered columns
df = pd.read_csv(data_dir + train_fname, usecols=filtered_columns)
df = df.dropna(axis=1, how='all')

# Print CSV headers to debug
print("CSV columns:", df.columns.tolist())

# Prepare predictors (drop metadata and observed target)
preds_headers = [col for col in df.columns if col not in ['utctime', 'name', 'latitude', 'longitude', pred]]
X = df[preds_headers]

# Load model
model = xgb.XGBRegressor()
model.load_model(f'{mod_dir}{mod_name}')

# Retrieve expected features from the model
model_features = model.get_booster().feature_names
print("Model expected features:", model_features)

# Ensure X only contains features that were in the training data
missing_features = [col for col in X.columns if col not in model_features]
if missing_features:
    print(f"Removing features that weren't in training data: {missing_features}")
    X = X[[col for col in X.columns if col in model_features]]

# Add any missing features from training data with zeros
missing_from_X = [col for col in model_features if col not in X.columns]
if missing_from_X:
    print(f"Adding missing features to data: {missing_from_X}")
    for col in missing_from_X:
        X[col] = 0

# Ensure column order matches exactly what the model expects
X = X[model_features]

# Make predictions
predictions = model.predict(X)

# Create results dataframe
results_df = pd.DataFrame({
    'utctime': df['utctime'],
    'observed': df[pred],
    'predicted': predictions
})

# Save to CSV
results_df.to_csv(f'{out_dir}{out_fname}', index=False)

print(f"Predictions saved to {out_dir}{out_fname}")
executionTime = (time.time() - startTime)
print('Execution time in minutes: %.2f' % (executionTime / 60))