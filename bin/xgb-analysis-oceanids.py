import xgboost as xgb # type: ignore
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import importlib
import sys

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


startTime=time.time()

def filter_points(df,lat,lon,nro,name):
    df0=df.copy()
    filter1 = df0['latitude'] == lat
    filter2 = df0['longitude'] == lon
    df0.where(filter1 & filter2, inplace=True)
    df0.columns=['lat-'+str(nro),'lon-'+str(nro),name+'-'+str(nro)] # change headers      
    df0=df0.dropna()
    return df0

data_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data
mdls_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
res_dir='/home/ubuntu/data/ML/results/OCEANIDS/'

# read in predictors in the fitted model from training data file
print(module.fname) # type: ignore
df=pd.read_csv(data_dir+module.fname,usecols=module.cols_own) # type: ignore
df=df.dropna(axis=1, how='all')
s1=df.shape[0]
df=df.dropna(axis=0,how='any')
s2=df.shape[0]
print('From '+str(s1)+' rows dropped '+str(s1-s2)+', apprx. '+str(round(100-s2/s1*100,1))+' %')

df['utctime']= pd.to_datetime(df['utctime'])
#headers=list(df) # list column headers
#preds=list(df[headers].drop(droplist, axis=1))
#print(preds)
preds=list(df.drop(['utctime',module.pred], axis=1)) # type: ignore
print(preds)

## F-score
print("start fscore")
mdl=mdls_dir+module.mdl_name # type: ignore
models=[]
fitted_mdl=xgb.XGBRegressor()
fitted_mdl.load_model(mdl)
models.append(fitted_mdl)

all_scores=pd.DataFrame(columns=['Model','predictor','meangain'])
row=0
for i,mdl in enumerate(models):
    mdl.get_booster().feature_names = list(preds) # predictor column headers
    bst=mdl.get_booster() # get the underlying xgboost Booster of model
    gains=np.array(list(bst.get_score(importance_type='gain').values()))
    features=np.array(list(bst.get_fscore().keys()))
    '''
    get_fscore uses get_score with importance_type equal to weight
    weight: the number of times a feature is used to split the data across all trees
    gain: the average gain across all splits the feature is used in
    '''
    for feat,gain in zip(features,gains):
        all_scores.loc[row]=(i+1,feat,gain); row+=1
all_scores=all_scores.drop(columns=['Model'])
mean_scores=all_scores.groupby('predictor').mean().sort_values('meangain')
print(mean_scores)

f, ax = plt.subplots(1,1,figsize=(6, 10))
mean_scores.plot.barh(ax=ax, legend=False)
ax.set_xlabel('F score')
ax.set_title(module.mdl_name) # type: ignore
ax.set_xscale('log')
plt.tight_layout()
f.savefig(res_dir+module.fscorepic, dpi=200) # type: ignore
#plt.show()
plt.clf(); plt.close('all')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))



