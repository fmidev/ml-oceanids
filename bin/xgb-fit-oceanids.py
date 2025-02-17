import time,warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import importlib
import sys
import json

warnings.filterwarnings("ignore")
### XGBoost for OCEANIDS

startTime=time.time()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='XGBoost with Optuna hyperparameter tuning for OCEANIDS')
parser.add_argument('module_name', type=str, help='Name of the module to import')
parser.add_argument('--pred', type=str, required=True, help='Prediction variable')
parser.add_argument('--model', type=str, required=False, help='Model name (optional, for eurocordex models)', default=None)
args = parser.parse_args()

module_name = args.module_name
pred = args.pred
model = args.model

try:
    module = importlib.import_module(module_name)
    print(f"Successfully imported {module_name}")
    if args.model:
        module.set_variables(pred, model)
    else:
        module.set_variables(pred)
except ImportError:
    print(f"Failed to import {module_name}")
    sys.exit(1)

print(f"Module: {module_name}, Prediction: {pred}, Model: {model}")

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mod_dir='/home/ubuntu/data/ML/models/OCEANIDS' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS'

### Read in 2D tabular training data
print(module.fname)
df=pd.read_csv(data_dir+module.fname,usecols=module.cols_own)

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
print('test ',module.test_y,' train ',module.train_y)
train_stations,test_stations=pd.DataFrame(),pd.DataFrame()
for y in module.train_y:
        train_stations=pd.concat([train_stations,df[df['utctime'].dt.year == y]],ignore_index=True)
for y in module.test_y:
        test_stations=pd.concat([test_stations,df[df['utctime'].dt.year == y]],ignore_index=True)

# Split to predictors (preds) and predictand (var) data
var_headers=list(df[[pred]])
preds_headers=list(df[headers].drop(['utctime',pred], axis=1))
preds_train=train_stations[preds_headers] 
preds_test=test_stations[preds_headers]
var_train=train_stations[var_headers]
var_test=test_stations[var_headers]

### XGBoost
# Load hyperparameters from JSON (Optuna tuned) 
params_file = f"{mod_dir}/{module.mdl_name[:-4]}_params.json"
with open(params_file, 'r') as f:
    params = json.load(f)

# Get parameters from JSON
max_depth = params["max_depth"]
subsample = params["subsample"]
lrte = params["learning_rate"]
colsample_bytree = params["colsample_bytree"]
nstm = params["n_estimators"]
a = params["alpha"]
num_parallel_tree = params["num_parallel_tree"]
qa=module.quantile_alpha

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:quantileerror',#'count:poisson','reg:squarederror'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            quantile_alpha=qa,#gamma=0.01
            alpha=a,#gamma=0.01
            min_child_weight=1,
            #num_boost_round=500,
            num_parallel_tree=num_parallel_tree,
            n_jobs=64,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            #colsample_bynode=colsample_bynode,
            eval_metric='mae', #'rmse', 
            random_state=99,
            early_stopping_rounds=50
            )

# train model 
eval_set=[(preds_test,var_test)]
xgbr.fit(
        preds_train,var_train,
        eval_set=eval_set)
print(xgbr)

# predict var and compare with test
var_pred=xgbr.predict(preds_test)
mse=mean_squared_error(var_test,var_pred)
mae=mean_absolute_error(var_test,var_pred)

# save model 
xgbr.save_model(mod_dir+'/'+module.mdl_name)

print("RMSE: %.5f" % (mse**(1/2.0)))
print("MAE: %.5f" % (mae))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))