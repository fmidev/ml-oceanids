import os,optuna,time,warnings,json,sys,commentjson
import sklearn.metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd
import xgboost as xgb
import numpy as np
# XGBoost with Optuna hyperparameter tuning in OCEANIDS project for ERA5/ERA5D training data
# give the name of the harbor and predictand as cmd argument
# note: does not save trained mdl
# (AK 2025)
# adapted to cordex data (RS)
warnings.filterwarnings("ignore")

startTime = time.time()

# Give harbor name and predictand as cmd arguments
scenario=sys.argv[1]
harbor_name=sys.argv[2]
pred=sys.argv[3]
model=sys.argv[4]

# Optuna objective
def objective(trial):
    # hyperparameters
    param = {
        "objective":"reg:quantileerror",#reg:squarederror
        "num_parallel_tree":1,#trial.suggest_int("number_parallel_tree", 1, 10), # 1 for quantileerror
        "max_depth":trial.suggest_int("max_depth",3,18),
        "subsample":trial.suggest_float("subsample",0.01,1),
        "learning_rate":trial.suggest_float("learning_rate",0.01,0.7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
        "n_estimators":trial.suggest_int("n_estimators",50,1000),
        "min_child_weight":1,
        "quantile_alpha":0.95,
        #"alpha":trial.suggest_float("alpha", 0.000000001, 1.0),
        "n_jobs":64,
        "random_state":99,
        "early_stopping_rounds":50,
        "eval_metric":'mae'#"rmse"
    }
    eval_set=[(valid_x,valid_y)]

    xgbr=xgb.XGBRegressor(**param)
    bst = xgbr.fit(train_x,train_y,eval_set=eval_set)
    preds = bst.predict(valid_x)
    #accuracy = np.sqrt(mean_squared_error(valid_y,preds))
    accuracy = mean_absolute_error(valid_y,preds)
    print("MAE: "+str(accuracy))

    return accuracy

data_dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{harbor_name}/'  # training data
mod_dir = f'/home/ubuntu/data/ML/models/OCEANIDS/cordex/{harbor_name}/'  # save hyperparameters
optuna_dir = '/home/ubuntu/data/ML/'  # optuna storage

fname=f'training_data_oceanids_{harbor_name}_cordex_{scenario}_{model}.csv'
hfname=f'hyperparameters_cordex_{scenario}_{harbor_name}_{pred}_{model}.json'
study=f'cordex_{scenario}_{harbor_name}_{pred}_{model}_qe'

# Load train/validation years split config file
with open(f'{mod_dir}cordex_{scenario}_{harbor_name}_{pred}_{model}_best_split.json', 'r') as file3:
    yconfig = json.load(file3)
test_y=yconfig.get('test_years')
train_y=yconfig.get('train_years')

# predictand mappings from JSON file and determine qa (quantile_alpha) for the given pred
with open('cordex_predictand_mappings.json', 'r') as f:
    mappings = json.load(f)
predictand_mappings = { key: mappings[key]["parameter"] for key in mappings }
qa = mappings[pred]["quantile_alpha"]

selected_value = predictand_mappings[pred]
keys_to_drop = [key for key in predictand_mappings if key != pred]
values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]

# Load training data config file
with open(f'cordex_training_data_config.json', 'r') as file:
    config = commentjson.load(file) # commentjson to allow comments syntax in json files
columns = config['training_columns']
print(columns)

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

# Read in training data
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
#print(df)

# Split to train and test by years, KFold for best split (k=5)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# Split to predictors (preds) and predictand (var) data
var_headers=list(df[[pred]])
preds_headers=list(df[headers].drop(['utctime','name','latitude', 'longitude',pred], axis=1))
train_x=train_stations[preds_headers] 
valid_x=test_stations[preds_headers]
train_y=train_stations[var_headers]
valid_y=test_stations[var_headers]
    
### Optuna trials

# move to correct dir for optuna study 
os.chdir(optuna_dir)
print(os.getcwd())

study = optuna.create_study(storage="sqlite:///MLexperiments.sqlite3",study_name=study,direction="minimize",load_if_exists=True)
study.optimize(objective, n_trials=50, timeout=432000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Save best hyperparameters to json
best_params = {
    "best_rmse": trial.value,
    "max_depth": trial.params["max_depth"],
    "subsample": trial.params["subsample"],
    "learning_rate": trial.params["learning_rate"],
    "colsample_bytree": trial.params["colsample_bytree"],
    "n_estimators": trial.params["n_estimators"],
    #"alpha": trial.params["alpha"],
    "quantile_alpha":qa,
    "num_parallel_tree": 1
}
with open(mod_dir+hfname, 'w') as f:
    json.dump(best_params, f, indent=4)

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))