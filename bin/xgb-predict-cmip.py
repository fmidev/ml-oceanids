import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import matplotlib.pyplot as plt
import argparse
import importlib

mod_dir='/home/ubuntu/data/ML/models/OCEANIDS/' # saved mdl
pred_dir='/home/ubuntu/data/ML/training-data/OCEANIDS/' # training data

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



# Load the prediction data
df_fin = pd.read_csv(pred_dir + module.pname, parse_dates=['utctime'])
#print(df_fin)
df_obs = pd.read_csv(pred_dir + module.fname, parse_dates=['utctime'])
#print(df_obs)
df_result = pd.DataFrame(df_fin['utctime'])
df_result[pred] = df_obs[pred]

if pred == 'WG_PT24H_MAX':
    df_result["maxWind_sum_mean"] = df_fin["maxWind_sum_mean"]
    df_result["maxWind_sum_max"] = df_fin["maxWind_sum_max"]
    df_result["maxWind_sum_min"] = df_fin["maxWind_sum_min"] 
elif pred == 'WS_PT24H_AVG':
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
elif pred == 'TP_PT24H_SUM':
    df_result["pr_sum_mean"] = df_fin["pr_sum_mean"] * 86400
    df_result["pr_sum_max"] = df_fin["pr_sum_max"] * 86400
    df_result["pr_sum_min"] = df_fin["pr_sum_min"] * 86400
else:
    raise ValueError("Invalid predictor")
    

# Load the model
fitted_mdl = xgb.XGBRegressor()
fitted_mdl.load_model(mod_dir + module.mdl_name)


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

df_result.to_csv(f'/home/ubuntu/data/ML/results/OCEANIDS/cordex-{model}-{module.harbor}-{pred}-quantileerror-predictions-{module.starty}-{module.prediction_endy}.csv', index=False)

print(df_result)
