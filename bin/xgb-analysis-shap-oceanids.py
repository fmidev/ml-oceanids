import xgboost as xgb # type: ignore
import time,sys,os,json,shap,commentjson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# SHAP analysis for XGBoost model
# Mean absolute SHAP values for predictors from validation data set
warnings.filterwarnings("ignore")

startTime=time.time()

harbor_name=sys.argv[1]
pred=sys.argv[2]

'''# Load harbor config file
with open('harbors_config.json', 'r') as file:
    config = json.load(file)

# Access info for a specific harbor
harbor = config.get(harbor_name, {})
start = harbor.get('start')
end=harbor.get('end')
'''

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/' # training data
mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/{harbor_name}/' # saved mdl
res_dir=f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/' # SHAP pic
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

fname=f'training_data_oceanids_{harbor_name}-sf-addpreds.csv'
mod_name=f'mdl_{harbor_name}_{pred}_xgb_era5_oceanids-QE.json'
shappic=f'shap_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png'

# Load train/validation years split config file
with open(f'{mod_dir}{harbor_name}_{pred}_best_split.json', 'r') as file3:
    yconfig = json.load(file3)
test_y=yconfig.get('test_years')
train_y=yconfig.get('train_years')

# Define the predictand mappings
predictand_mappings={
    'WG_PT24H_MAX': 'fg10',
    'TA_PT24H_MAX': 'mx2t',
    'TA_PT24H_MIN': 'mn2t',
    'TP_PT24H_ACC': 'tp'
    }
selected_value = predictand_mappings[pred]
keys_to_drop = [key for key in predictand_mappings if key != pred]
values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]

# Load training data config file
with open(f'training_data_config.json', 'r') as file:
    config = commentjson.load(file) # commentjson to allow comments syntax in json files
columns = config['training_columns']
#print(columns)

# Filter the columns for predictor and related variables:
filtered_columns = []
for col in columns:
    drop = any(drop_key in col for drop_key in keys_to_drop)
    drop = drop or any(drop_val in col for drop_val in values_to_drop)
    if not drop:
        filtered_columns.append(col)
df=pd.read_csv(data_dir+fname,usecols=filtered_columns)
print(df.columns.tolist())

# drop NaN values
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
headers=list(df) # list column headers
#print(df)

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

explainer = shap.TreeExplainer(fitted_mdl, X_val)
shap_values_val = explainer.shap_values(X_val)
mean_abs_shap_val = np.mean(np.abs(shap_values_val), axis=0)
mean_abs_shap_df = pd.DataFrame({
    'predictor': preds_headers,
    'mean_abs_shap': mean_abs_shap_val
}).sort_values('mean_abs_shap')
    
print("Mean absolute SHAP values (computed on the validation sample):")
print(mean_abs_shap_df)
plt.figure(figsize=(30, 10))
shap.summary_plot(shap_values_val, X_val, max_display=len(preds_headers), plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(res_dir+shappic, dpi=200)
plt.clf()
plt.close('all')

#######################
# SHAP Analysis
'''training_data_file = data_dir+fname
if os.path.exists(training_data_file):
    X_train = pd.read_csv(training_data_file, usecols=preds)
    
    # Create a TreeExplainer and compute SHAP values
    explainer = shap.TreeExplainer(fitted_mdl)
    shap_values = explainer.shap_values(X_train)
    
    # Generate and save a SHAP summary plot (bar plot)
    plt.figure(figsize=(30, 10))
    shap.summary_plot(shap_values, X_train,max_display=X_train.shape[1], plot_type="bar", show=False)
    shap_plot_file = os.path.join(res_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(shap_plot_file, dpi=200)
    plt.clf()
    plt.close('all')
'''
###
'''
## SHAP Analysis and Visualization
training_data_file = data_dir+fname
if os.path.exists(training_data_file):
    X_train = pd.read_csv(training_data_file, usecols=preds)
    
    # Compute SHAP values for the entire training set
    explainer = shap.TreeExplainer(fitted_mdl)
    shap_values = explainer.shap_values(X_train)
    
    # (Optional) Create a SHAP Explanation object for improved API compatibility
    explanation = shap.Explanation(values=shap_values, data=X_train, feature_names=preds)
    
    print('plot beeswarm')
    # ----- 1. Beeswarm (Summary) Plot -----
    # This plot shows the distribution of SHAP values for each feature.
    plt.figure(figsize=(40, 20))  # make the plot wider if needed
    shap.summary_plot(shap_values, X_train, max_display=len(preds))
    plt.savefig(os.path.join(res_dir, "shap_beeswarm.png"), dpi=200)
    plt.clf()
    plt.close('all')
    
    print('plot bar')
    # ----- 2. Bar Plot -----
    # A bar plot of mean absolute SHAP values to rank features globally.
    plt.figure(figsize=(40, 20))
    shap.plots.bar(explanation, max_display=len(preds))
    plt.savefig(os.path.join(res_dir, "shap_bar.png"), dpi=200)
    plt.clf()
    plt.close('all')
    
    print('plot dependence')
    # ----- 3. Dependence Plot -----
    # Shows how the SHAP value for a single feature varies with its value.
    # (Using the first predictor as an example; change as needed.)
    if preds:
        feature_for_dependence = preds[0]
        plt.figure(figsize=(40, 20))
        shap.dependence_plot(feature_for_dependence, shap_values, X_train)
        plt.savefig(os.path.join(res_dir, "shap_dependence.png"), dpi=200)
        plt.clf()
        plt.close('all')
    
    print('plot force')
    # ----- 4. Force Plot -----
    # A local explanation for an individual instance.
    # (For static output, we use matplotlib; interactive force plots are typically rendered in a browser.)
    plt.figure(figsize=(40, 20))
    # For a single instance (here, the first row)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :], matplotlib=True)
    plt.savefig(os.path.join(res_dir, "shap_force.png"), dpi=200)
    plt.clf()
    plt.close('all')
    
    print('plot waterfall')
    # ----- 5. Waterfall Plot -----
    # Another local explanation style for a single prediction.
    plt.figure(figsize=(40, 20))
    shap.plots.waterfall(shap.Explanation(values=shap_values[0, :],
                                          base_values=explainer.expected_value,
                                          data=X_train.iloc[0, :],
                                          feature_names=preds))
    plt.savefig(os.path.join(res_dir, "shap_waterfall.png"), dpi=200)
    plt.clf()
    plt.close('all')
    
    print('plot decision')
    # ----- 6. Decision Plot -----
    # Shows the cumulative effect of features for each prediction.
    plt.figure(figsize=(40, 20))
    shap.decision_plot(explainer.expected_value, shap_values, X_train, feature_names=preds)
    plt.savefig(os.path.join(res_dir, "shap_decision.png"), dpi=200)
    plt.clf()
    plt.close('all')

'''
executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))