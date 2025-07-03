import xgboost as xgb # type: ignore
import time,sys,os,json,shap,commentjson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
# SHAP analysis for XGBoost model
# Compares variables from 4 grid points (-1, -2, -3, -4) for feature importance
warnings.filterwarnings("ignore")

startTime=time.time()

harbor_name=sys.argv[1]
pred=sys.argv[2]

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/' # training data
mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/{harbor_name}/' # saved mdl
res_dir=f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/' # SHAP pic
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

fname=f'training_data_oceanids_{harbor_name}-sf-addpreds.csv.gz'
mod_name=f'mdl_{harbor_name}_{pred}_xgb_era5_oceanids-QE.json'
shappic=f'shap_comparison_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png'

# Load train/validation years split config file
with open(f'{mod_dir}{harbor_name}_{pred}_best_split.json', 'r') as file3:
    split_dict = json.load(file3)

test_y=split_dict.get('test_years')
train_y=split_dict.get('train_years')

# predictand mappings from JSON file
with open('predictand_mappings.json', 'r') as f:
    mappings = json.load(f)
predictand_mappings = { key: mappings[key]["parameter"] for key in mappings }

selected_value = predictand_mappings[pred]
keys_to_drop = [key for key in predictand_mappings if key != pred]
values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]

# Load training data config file
with open(f'training_data_config.json', 'r') as file:
    config = commentjson.load(file) # commentjson to allow comments syntax in json files
columns = config['training_columns']

# Filter the columns for predictor and related variables, but always include ewss columns
filtered_columns = []
for col in columns:
    drop = any(drop_key in col for drop_key in keys_to_drop)
    drop = drop or any(drop_val in col for drop_val in values_to_drop)
    if not drop:
        filtered_columns.append(col)

# Load data and preprocess
df=pd.read_csv(data_dir+fname,usecols=filtered_columns)
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
headers=list(df) # list column headers

# Split to train and test by years, KFold for best split (k=5)
# we need the validation (test) set for SHAP analysis
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# Prepare validation dataset from training data set (as in fitting)
preds_headers=list(df[headers].drop(['utctime','name','latitude', 'longitude',pred], axis=1))
X_val=test_stations[preds_headers]
y_val=test_stations[pred]

# Load the model   
fitted_mdl = xgb.XGBRegressor()
fitted_mdl.load_model(mod_dir + mod_name)

# Get model information
booster = fitted_mdl.get_booster()
model_feature_count = booster.num_features()
print(f"Model expects {model_feature_count} features")

# Get feature names from the model
model_features = fitted_mdl.get_booster().feature_names  # For XGBoost

# Ensure X_val only contains features that were in the training data
missing_features = [col for col in X_val.columns if col not in model_features]
if missing_features:
    print(f"Removing features that weren't in training data: {missing_features}")
    X_val = X_val[[col for col in X_val.columns if col in model_features]]

# Add any missing features from training data with zeros
missing_from_val = [col for col in model_features if col not in X_val.columns]
if missing_from_val:
    print(f"Adding missing features to validation data: {missing_from_val}")
    for col in missing_from_val:
        X_val[col] = 0

# Ensure column order matches exactly
X_val = X_val[model_features]

# Calculate SHAP values - using full validation dataset to match original scripts
try:
    # Try to use Explainer with model's predict function
    print(f"Calculating SHAP values for full validation dataset ({X_val.shape[0]} rows)...")
    explainer = shap.Explainer(fitted_mdl.predict, X_val)
    shap_values_val = explainer(X_val)
    
    # Extract values for traditional SHAP plots
    if hasattr(shap_values_val, "values"):
        shap_values_val = shap_values_val.values
    
    print("SHAP values calculated successfully")
except Exception as e:
    print(f"Primary SHAP calculation method failed: {e}")
    print("Trying fallback method with TreeExplainer...")
    
    explainer = shap.TreeExplainer(booster)
    shap_values_val = explainer.shap_values(X_val)
    print("Successfully calculated SHAP values using TreeExplainer.")


# Group features by grid point (-1, -2, -3, -4)
grid_point_patterns = {
    'Grid Point 1': r'-1$',
    'Grid Point 2': r'-2$',
    'Grid Point 3': r'-3$',
    'Grid Point 4': r'-4$'
}

# Function to get grid point for a feature
def get_grid_point(feature_name):
    for grid_name, pattern in grid_point_patterns.items():
        if re.search(pattern, feature_name):
            return grid_name
    return "Other"  # For variables that don't have grid point suffix

# Function to get base variable name (without grid point suffix)
def get_base_name(feature_name):
    return re.sub(r'-[1-4]$', '', feature_name)

# Group features by grid point
grouped_features = {}
for i, feature in enumerate(preds_headers):
    grid_point = get_grid_point(feature)
    if grid_point not in grouped_features:
        grouped_features[grid_point] = []
    
    # Store index, name and mean absolute SHAP value
    mean_abs_shap = np.mean(np.abs(shap_values_val[:, i]))
    grouped_features[grid_point].append((i, feature, mean_abs_shap))

# Sort features by importance within each group
for grid_point in grouped_features:
    grouped_features[grid_point].sort(key=lambda x: x[2], reverse=True)

# Group features by base variable name (across all grid points)
base_variable_features = {}
for i, feature in enumerate(preds_headers):
    base_name = get_base_name(feature)
    if base_name not in base_variable_features:
        base_variable_features[base_name] = []
    
    # Store index, name and mean absolute SHAP value
    mean_abs_shap = np.mean(np.abs(shap_values_val[:, i]))
    base_variable_features[base_name].append((i, feature, mean_abs_shap))

#--------------------------------------------------------------------------
# VISUALIZATION 1: COMPARING GRID POINTS (AGGREGATE ALL VARIABLES BY POINT)
#--------------------------------------------------------------------------

# Calculate total importance for each grid point
grid_point_importance = {}
for grid_point, features in grouped_features.items():
    if grid_point == "Other":
        continue
    total_importance = sum(shap_val for _, _, shap_val in features)
    grid_point_importance[grid_point] = total_importance

# Create bar chart comparing grid points
plt.figure(figsize=(15, 10))
grid_points = list(grid_point_importance.keys())
importances = [grid_point_importance[gp] for gp in grid_points]

plt.bar(grid_points, importances, color=['red', 'blue', 'green', 'purple'])
plt.title(f'Aggregated SHAP Importance by Grid Point - {harbor_name} {pred}', fontsize=16)
plt.xlabel('Grid Point', fontsize=14)
plt.ylabel('Total |SHAP Value|', fontsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig(res_dir + f'grid_point_comparison_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png', dpi=200)
plt.close()

# Create beeswarm plot for grid point comparison
# We'll create a new dataset with one feature per grid point, where each feature
# is the sum of all SHAP values for that grid point
rows = X_val.shape[0]
grid_point_shap_values = np.zeros((rows, 4))
grid_point_features = np.zeros((rows, 4))

for i, grid_point in enumerate(sorted(grid_points)):
    indices = [idx for idx, _, _ in grouped_features[grid_point]]
    # Sum SHAP values across all features for this grid point
    for j in indices:
        grid_point_shap_values[:, i] += shap_values_val[:, j]
    
    # For the feature values, use the average value of all features for this grid point
    feature_values = X_val.iloc[:, indices].mean(axis=1)
    grid_point_features[:, i] = feature_values

# Create a DataFrame for the beeswarm plot
grid_point_df = pd.DataFrame(grid_point_features, columns=sorted(grid_points))

plt.figure(figsize=(15, 10))
shap.summary_plot(
    grid_point_shap_values,
    grid_point_df,
    plot_type="dot",
    show=False
)
plt.title(f'SHAP Beeswarm for Grid Points - {harbor_name} {pred}', fontsize=16)
plt.tight_layout()
plt.savefig(res_dir + f'grid_point_beeswarm_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png', dpi=200)
plt.close()

#-----------------------------------------------------------------------------
# VISUALIZATION 2: COMPARING VARIABLES (AGGREGATE ACROSS ALL GRID POINTS)
#-----------------------------------------------------------------------------

# Calculate total importance for each base variable
base_var_importance = {}
for base_name, features in base_variable_features.items():
    total_importance = sum(shap_val for _, _, shap_val in features)
    base_var_importance[base_name] = total_importance

# Sort base variables by importance
sorted_base_vars = sorted(base_var_importance.items(), key=lambda x: x[1], reverse=True)
top_base_vars = sorted_base_vars[:20]  # Take top 20 for visualization

# Create bar chart comparing base variables
plt.figure(figsize=(20, 15))
base_vars = [var for var, _ in top_base_vars]
importances = [imp for _, imp in top_base_vars]

plt.barh(base_vars, importances, color='skyblue')
plt.title(f'Aggregated SHAP Importance by Variable Type - {harbor_name} {pred}', fontsize=16)
plt.xlabel('Total |SHAP Value|', fontsize=14)
plt.ylabel('Variable', fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(res_dir + f'variable_comparison_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png', dpi=200)
plt.close()

# Create beeswarm plot for variable comparison
# We need to aggregate SHAP values by base variable name
top_base_var_names = [var for var, _ in top_base_vars]
rows = X_val.shape[0]
base_var_shap_values = np.zeros((rows, len(top_base_var_names)))
base_var_features = np.zeros((rows, len(top_base_var_names)))

for i, base_name in enumerate(top_base_var_names):
    if base_name in base_variable_features:
        indices = [idx for idx, _, _ in base_variable_features[base_name]]
        
        # Sum SHAP values across all grid points for this variable
        for j in indices:
            base_var_shap_values[:, i] += shap_values_val[:, j]
        
        # For the feature values, use the average value across all grid points
        feature_values = X_val.iloc[:, indices].mean(axis=1)
        base_var_features[:, i] = feature_values

# Create a DataFrame for the beeswarm plot
base_var_df = pd.DataFrame(base_var_features, columns=top_base_var_names)

plt.figure(figsize=(20, 15))
shap.summary_plot(
    base_var_shap_values,
    base_var_df,
    plot_type="dot",
    show=False
)
plt.title(f'SHAP Beeswarm for Variable Types - {harbor_name} {pred}', fontsize=16)
plt.tight_layout()
plt.savefig(res_dir + f'variable_beeswarm_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png', dpi=200)
plt.close()

# Save the aggregated data for reference
grid_point_data = pd.DataFrame({
    'Grid Point': list(grid_point_importance.keys()),
    'Total SHAP Value': list(grid_point_importance.values())
}).sort_values('Total SHAP Value', ascending=False)

base_var_data = pd.DataFrame({
    'Variable': [var for var, _ in sorted_base_vars],
    'Total SHAP Value': [imp for _, imp in sorted_base_vars]
})

grid_point_data.to_csv(res_dir + f'grid_point_importance_{harbor_name}_{pred}.csv', index=False)
base_var_data.to_csv(res_dir + f'variable_importance_{harbor_name}_{pred}.csv', index=False)

# Print summary information
print("Grid point importance summary:")
print(grid_point_data)
print("\nVariable importance summary:")
print(base_var_data.head(20))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
