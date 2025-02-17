import time,warnings,sys,json,os,commentjson
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
### XGBoost for fitting a model with ERA5/ERA5D data in OCEANIDS
# (AK 2025) 
warnings.filterwarnings("ignore")

startTime=time.time()

harbor_name=sys.argv[1]
pred=sys.argv[2]

# Load harbor config file
with open('harbors_config.json', 'r') as file1:
    config = json.load(file1)
harbor = config.get(harbor_name, {})
start = harbor.get('start')
end=harbor.get('end')

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/' # training data
mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/{harbor_name}/' # saved mdl

fname=f'training_data_oceanids_{harbor_name}-sf-addpreds.csv'
mod_name=f'mdl_{harbor_name}_{pred}_xgb_era5_oceanids-QE.json'

# Load hyperparameters config file
with open(f'{mod_dir}hyperparameters_{harbor_name}_{pred}.json', 'r') as file2:
    hpconfig = json.load(file2)
max_depth=hpconfig.get('max_depth')
subsample=hpconfig.get('subsample')
lrte=hpconfig.get('learning_rate')
colsample_bytree=hpconfig.get('colsample_bytree')
nstm=hpconfig.get('n_estimators')
qa=hpconfig.get('quantile_alpha')
#al=hpconfig.get('alpha')
num_parallel_tree = hpconfig.get('num_parallel_tree')

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
print(columns)

# Filter the columns for predictor and related variables:
filtered_columns = []
for col in columns:
    drop = any(drop_key in col for drop_key in keys_to_drop)
    drop = drop or any(drop_val in col for drop_val in values_to_drop)
    if not drop:
        filtered_columns.append(col)
df=pd.read_csv(data_dir+fname,usecols=filtered_columns)
#print(df)

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
preds_train=train_stations[preds_headers] 
preds_test=test_stations[preds_headers]
var_train=train_stations[var_headers]
var_test=test_stations[var_headers]

### XGBoost fitting

# initialize and tune model
xgbr=xgb.XGBRegressor(
            objective= 'reg:quantileerror',#'reg:squarederror'
            n_estimators=nstm,
            learning_rate=lrte,
            max_depth=max_depth,
            quantile_alpha=qa,#gamma=0.01
            #alpha=al,#gamma=0.01
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
xgbr.save_model(f'{mod_dir}{mod_name}')

print("RMSE: %.5f" % (mse**(1/2.0)))
print("MAE: %.5f" % (mae))

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))