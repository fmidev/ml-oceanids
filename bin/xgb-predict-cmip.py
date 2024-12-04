import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import matplotlib.pyplot as plt
#from Sitia_023330_ece3 import *
from Bremerhaven_004885_ece3 import *

mod_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
pred_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data



# Load the prediction data
df_fin = pd.read_csv(pred_dir + f"prediction_data_oceanids_ece3-Bremerhaven-{pred}-2000-2100.csv", parse_dates=['utctime'])
df_obs = pd.read_csv(pred_dir + f"ece3-Bremerhaven-{pred}.csv", parse_dates=['utctime'])
df_result = pd.DataFrame(df_fin['utctime'])
df_result[pred] = df_obs[pred]

# Set values to NaN for dates after 2023-08-31
cutoff_date = pd.Timestamp('2023-08-31')
df_result.loc[df_result['utctime'] > cutoff_date, pred] = np.nan


if pred == 'WG_PT24H_MAX' or pred == 'WS_PT24H_AVG':
    df_result["sfcWind_sum_mean"] = df_fin["sfcWind_sum_mean"]
    df_result["sfcWind_sum_max"] = df_fin["sfcWind_sum_max"]
    df_result["sfcWind_sum_min"] = df_fin["sfcWind_sum_min"]
elif pred == 'TN_PT24H_MIN':
    df_result["tasmin_sum_mean"] = df_fin["tasmin_sum_mean"] - 273.15
    df_result["tasmin_sum_max"] = df_fin["tasmin_sum_max"] - 273.15
    df_result["tasmin_sum_min"] = df_fin["tasmin_sum_min"] - 273.15
elif pred == 'TX_PT24H_MAX':
    df_result["tasmax_sum_mean"] = df_fin["tasmax_sum_mean"] - 273.15
    df_result["tasmax_sum_max"] = df_fin["tasmax_sum_max"] - 273.15
    df_result["tasmax_sum_min"] = df_fin["tasmax_sum_min"] - 273.15
else:
    df_result["pr_sum_mean"] = df_fin["pr_sum_mean"] * 68400
    df_result["pr_sum_max"] = df_fin["pr_sum_max"] * 68400
    df_result["pr_sum_min"] = df_fin["pr_sum_min"] * 68400
    

# Load the model
fitted_mdl = xgb.XGBRegressor()
fitted_mdl.load_model(mod_dir + mdl_name)

# Ensure the DataFrame has the correct columns
required_columns = fitted_mdl.get_booster().feature_names
if required_columns is None:
    # Manually specify the feature names if they are not available
    required_columns = df_fin.columns.tolist()
    print("Feature names not found in the model. Using DataFrame columns as feature names.")
else:
    print("Required columns:", required_columns)

df_fin = df_fin[required_columns]

# XGBoost predict without DMatrix
result = fitted_mdl.predict(df_fin)
result = result.tolist()
df_result['Predicted'] = result

df_result.to_csv(f'/home/ubuntu/data/ML/results/OCEANIDS/ece3-{harbor}-{pred}-quantileerror-predictions-2000-2100.csv', index=False)

print(df_result)
