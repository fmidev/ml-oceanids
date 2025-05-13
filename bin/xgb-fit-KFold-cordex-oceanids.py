import os,time,json,sys,commentjson
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
### XGBoost with KFold for OCEANIDS with Eurocordex data
# original script (AK 2025)
# adapted to cordex data (RS)
#warnings.filterwarnings("ignore")
startTime=time.time()

scenario=sys.argv[1]
harbor_name=sys.argv[2]
pred=sys.argv[3]
model=sys.argv[4]

# Load harbor config file
with open('harbors_config.json', 'r') as file:
    config = json.load(file)

# Access info for a specific harbor: start and end years needed for KFold split
harbor = config.get(harbor_name, {})
start = harbor.get('start')
end = harbor.get('end')
starty,endy = start[0:4],end[0:4]

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{harbor_name}/' # training data
mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/cordex/{harbor_name}/' # saved mdl
if not os.path.exists(mod_dir):
    os.makedirs(mod_dir)

fname=f'training_data_oceanids_{harbor_name}_cordex_{scenario}_{model}.csv'
mod_name=f'mdl_{harbor_name}_{pred}_{model}_xgb_cordex_{scenario}_oceanids-KFold'
best_split_file = f'cordex_{scenario}_{harbor_name}_{pred}_{model}_best_split.json'

# predictand mappings from JSON file and extract the parameter values
with open('cordex_predictand_mappings.json', 'r') as f:
    mappings = json.load(f)
predictand_mappings = { key: mapping["parameter"] for key, mapping in mappings.items() }

selected_value = predictand_mappings[pred]
keys_to_drop = [key for key in predictand_mappings if key != pred]
values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]

### Read in 2D tabular training data
# Load training data config file
with open(f'cordex_training_data_config.json', 'r') as file:
    config = commentjson.load(file) # commentjson to allow comments syntax in json files
columns = config['training_columns']

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
print(df)
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print(df)
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
headers=list(df) # list column headers
#print(df)

# Read predictor (preds) and predictand (var) data
var_headers=list(df[[pred]])
preds_headers=list(df[headers].drop(['utctime','name','latitude', 'longitude',pred], axis=1))
var=df[[pred]]
preds=df[preds_headers]

# Define hyperparameters for XGBoost
nstm=645
lrte=0.067
max_depth=10
subsample=0.29
colsample_bytree=0.56
#colsample_bynode=1
num_parallel_tree=10
eval_met='rmse'
a=0.54

# KFold cross-validation; splitting to train/test sets by years
y1,y2=int(starty),int(endy)
available_years = np.sort(df['utctime'].dt.year.unique())
allyears = np.array([year for year in available_years if y1 <= year <= y2])
print("Years for KFold splitting:", allyears)

kf=KFold(5,shuffle=True,random_state=20)
fold=0
mdls=[]
best_rmse = float('inf')
best_split = None

for train_idx, test_idx in kf.split(allyears):
    fold+=1
    train_years=allyears[train_idx]
    test_years=allyears[test_idx]
    # Create boolean masks based on available years in the DataFrame
    train_mask = np.isin(df['utctime'].dt.year, train_years)
    test_mask = np.isin(df['utctime'].dt.year, test_years)
    train_set = df[train_mask].reset_index(drop=True)
    test_set = df[test_mask].reset_index(drop=True)
    # Skip fold if test set is empty
    if test_set.empty or train_set.empty:
        print(f"Fold {fold}: Test or train set is empty for test years {test_years} or train years {train_years}. Skipping this fold.")
        continue   
    
    # Split to predictors and target variable
    preds_train=train_set[preds_headers]
    preds_test=test_set[preds_headers]
    var_train=train_set[var_headers]
    var_test=test_set[var_headers]
    
    # Define the model        
    xgbr=xgb.XGBRegressor(
            objective='reg:squarederror', # 'count:poisson'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            alpha=0.01, #gamma=0.01
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            #colsample_bynode=colsample_bynode,
            random_state=99,
            eval_metric=eval_met,
            early_stopping_rounds=50
            )
    
    # Train the model
    eval_set=[(preds_test,var_test)]
    fitted_mdl=xgbr.fit(
            preds_train,var_train,
            eval_set=eval_set,
            verbose=False #True
            )

    # Predict var and compare with test
    var_pred=fitted_mdl.predict(preds_test)
    mse=mean_squared_error(var_test,var_pred)
    rmse = mse**(1/2.0)
    
    print("Fold: %s RMSE: %.2f" % (fold, rmse))
    print('Train: ',train_years,'Test: ',test_years)
    mdls.append(fitted_mdl)
    
    # Check if this is the best model
    if rmse < best_rmse:
        best_rmse = rmse
        best_split = {
            'train_years': train_years.tolist(),
            'test_years': test_years.tolist()
        }

# Save XGB models
for i,mdl in enumerate(mdls):
    mdl.save_model(mod_dir+mod_name+f'_{i+1}.json')

# Save best train and test split to JSON file
with open(mod_dir+best_split_file, "w") as outfile:
    json.dump(best_split, outfile, indent=4)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))
