import xgboost as xgb # type: ignore
import time,sys,os,json,shap,commentjson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# SHAP analysis for XGBoost model with CORDEX data
# Mean absolute SHAP values for predictors from validation data set
warnings.filterwarnings("ignore")

startTime=time.time()

# Get command line arguments
harbor_name=sys.argv[1]
pred=sys.argv[2]
scenario=sys.argv[3]  # e.g., rcp45
model=sys.argv[4]     # e.g., mohc_hadgem2_es-smhi_rca4

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{harbor_name}/' # training data
mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/cordex/{harbor_name}/' # saved mdl
res_dir=f'/home/ubuntu/data/ML/results/OCEANIDS/cordex/{harbor_name}/' # SHAP pic
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Update file naming to include scenario and model
fname=f'training_data_oceanids_{harbor_name}_cordex_{scenario}_{model}.csv'
mod_name=f'mdl_{harbor_name}_{pred}_{model}_xgb_cordex_rcp45_oceanids-QE.json'
shappic=f'shap_{harbor_name}_{pred}_{model}_xgb_cordex_{scenario}_oceanids.png'

# Load train/validation years split config file
with open(f'{mod_dir}cordex_rcp45_{harbor_name}_{pred}_{model}_best_split.json', 'r') as file3:
    yconfig = json.load(file3)
test_y=yconfig.get('test_years')
train_y=yconfig.get('train_years')

# predictand mappings from JSON file
with open('cordex_predictand_mappings.json', 'r') as f:
    mappings = json.load(f)
predictand_mappings = { key: mappings[key]["parameter"] for key in mappings }

selected_value = predictand_mappings[pred]
keys_to_drop = [key for key in predictand_mappings if key != pred]
values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]

# Load training data config file
with open(f'cordex_training_data_config.json', 'r') as file:
    config = commentjson.load(file) # commentjson to allow comments syntax in json files
columns = config['training_columns']
#print(columns)

# Always include essential columns
essential_columns = ['utctime', 'name', 'latitude', 'longitude', pred]

# Filter the columns for predictor and related variables with more precise logic:
filtered_columns = []
for col in columns:
    # Always include essential columns
    if any(col == essential for essential in essential_columns):
        filtered_columns.append(col)
        continue
        
    # More precise filtering: check for exact matches or specific patterns
    should_drop = False
    for key in keys_to_drop:
        # Check if key is the column name or starts with key + specific delimiter
        if col == key or col.startswith(f"{key}_") or col.startswith(f"{key}."):
            should_drop = True
            break
            
    if not should_drop:
        for val in values_to_drop:
            # Check if value is in column name with some context (not just any substring)
            if f"_{val}" in col or f".{val}" in col or col.endswith(f"_{val}") or col.endswith(f".{val}"):
                should_drop = True
                break
    
    if not should_drop:
        filtered_columns.append(col)

# Debug information
print(f"Number of filtered columns: {len(filtered_columns)}")
print(f"First few filtered columns: {filtered_columns[:5] if filtered_columns else 'None'}")

# Read only the columns we need
if not filtered_columns:
    print("WARNING: No columns passed filtering! Using all available columns.")
    df = pd.read_csv(data_dir+fname)
else:
    df = pd.read_csv(data_dir+fname, usecols=filtered_columns)

print(f"DataFrame shape after loading: {df.shape}")
print(df.columns.tolist())

# drop NaN values
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

# Load the model   
fitted_mdl = xgb.XGBRegressor()
fitted_mdl.load_model(mod_dir + mod_name)

# SHAP analysis
explainer = shap.TreeExplainer(fitted_mdl, X_val)
shap_values_val = explainer.shap_values(X_val)
mean_abs_shap_val = np.mean(np.abs(shap_values_val), axis=0)
mean_abs_shap_df = pd.DataFrame({
    'predictor': preds_headers,
    'mean_abs_shap': mean_abs_shap_val
}).sort_values('mean_abs_shap')
    
print("Mean absolute SHAP values (computed on the validation sample):")
print(mean_abs_shap_df)

# Generate and save bar plot
plt.figure(figsize=(30, 10))
shap.summary_plot(shap_values_val, X_val, max_display=len(preds_headers), plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(res_dir+shappic, dpi=200)
plt.clf()
plt.close('all')

# Generate and save beeswarm plot with updated naming
beeswarm_pic = f'beeswarm_{harbor_name}_{pred}_{model}_xgb_cordex_{scenario}_oceanids-QE.png'
plt.figure(figsize=(30, 10))
shap.summary_plot(shap_values_val, X_val, max_display=len(preds_headers), show=False)
plt.tight_layout()
plt.savefig(res_dir + beeswarm_pic, dpi=200)
plt.clf()
plt.close('all')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
