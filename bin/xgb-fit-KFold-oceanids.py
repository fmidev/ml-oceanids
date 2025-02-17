import os, time, datetime, random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import importlib
import sys
import json

### XGBoost with KFold for OCEANIDS
startTime=time.time()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='XGBoost with Optuna hyperparameter tuning for OCEANIDS')
parser.add_argument('module_name', type=str, help='Name of the module to import')
parser.add_argument('--pred', type=str, required=True, help='Prediction variable')
parser.add_argument('--model', type=str, required=False, help='Model name (optional)', default=None)
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
mod_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS/'

### Read in 2D tabular training data
#df=pd.read_csv(data_dir+fname,usecols=cols_own)
df=pd.read_csv(data_dir+module.fname,usecols=module.cols_own)

# drop NaN values and columns
df=df.dropna(axis=1, how='all') 
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')
df['utctime']= pd.to_datetime(df['utctime'])
print(df)
headers=list(df) # list column headers

# Read predictor (preds) and predictand (var) data
var=df[[pred]]
preds=df[headers].drop(['utctime',pred], axis=1)
var_headers=list(var) 
preds_headers=list(preds)

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
y1,y2=int(module.starty),int(module.endy)
print(y1,y2)
allyears=np.arange(y1,y2+1).astype(int)

kf=KFold(5,shuffle=True,random_state=20)
fold=0
mdls=[]
best_rmse = float('inf')
best_split = None

for train_idx, test_idx in kf.split(allyears):
    fold+=1
    train_years=allyears[train_idx]
    test_years=allyears[test_idx]
    train_idx=np.isin(df['utctime'].dt.year,train_years)
    test_idx=np.isin(df['utctime'].dt.year,test_years)
    train_set=df[train_idx].reset_index(drop=True)
    test_set=df[test_idx].reset_index(drop=True)
   
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
            n_jobs=24,
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
    mdl.save_model(mod_dir+f'KFold-mdl_{pred}_{module.harbor}_{module.starty}-{module.endy}_{i+1}.json')

# Save the best split
if best_split:
    with open(mod_dir+f'best_split_{pred}_{module.harbor}_{module.starty}-{module.endy}.json', 'w') as f:
        json.dump(best_split, f)
        print(F"Best split saved to {mod_dir}best_split_{pred}_{module.harbor}_{module.starty}-{module.endy}.json")

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))