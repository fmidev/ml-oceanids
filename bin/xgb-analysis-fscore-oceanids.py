import xgboost as xgb # type: ignore
import time,sys,os,json,commentjson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

startTime=time.time()

harbor_name=sys.argv[1]
pred=sys.argv[2]

mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/{harbor_name}/' # saved mdl
res_dir=f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/' # F score pic
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

mod_name=f'mdl_{harbor_name}_{pred}_xgb_era5_oceanids-QE.json'
fscorepic=f'fscore_{harbor_name}_{pred}_xgb_era5_oceanids-QE.png'

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
print(columns)

# Filter the columns for predictor and related variables:
filtered_columns = []
for col in columns:
    drop = any(drop_key in col for drop_key in keys_to_drop)
    drop = drop or any(drop_val in col for drop_val in values_to_drop)
    if not drop:
        filtered_columns.append(col)
#print(filtered_columns)

drop_list=['utctime','name','latitude', 'longitude',pred]
preds = [item for item in filtered_columns if item not in drop_list]
#print(preds)

## F-score
print("start fscore")
mdl=mod_dir+mod_name
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

f, ax = plt.subplots(1,1,figsize=(10, 35))
mean_scores.plot.barh(ax=ax, legend=False)
ax.set_xlabel('F score')
ax.set_title(mod_name)
ax.set_xscale('log')
plt.tight_layout()
f.savefig(res_dir+fscorepic, dpi=200)
#plt.show()
plt.clf(); plt.close('all')

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))