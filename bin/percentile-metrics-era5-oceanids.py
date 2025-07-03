#!/usr/bin/env python3
"""
Script to calculate 95th percentile values for observations and predictions
for all harbor and predictor combinations.
"""

import time, warnings, sys, json, os, commentjson
import pandas as pd
import numpy as np
from datetime import datetime
warnings.filterwarnings("ignore")

# List of predictors to process (same as in create-shap-slides.py)
PREDICTORS = [
    'TA_PT24H_MAX',
    'TA_PT24H_MIN',
    'WS_PT24H_AVG',
    'RH_PT24H_AVG',
    'TP_PT24H_ACC',
    'WG_PT24H_MAX'
]

# Predictors whose values are in Kelvin and need conversion to Celsius
temp_predictors = ['TA_PT24H_MAX', 'TA_PT24H_MIN']

# Mapping of predictor codes to human-readable descriptions
PREDICTOR_DESCRIPTIONS = {
    'TA_PT24H_MAX': 'Maximum Temperature',
    'TA_PT24H_MIN': 'Minimum Temperature',
    'WS_PT24H_AVG': 'Average Wind Speed',
    'RH_PT24H_AVG': 'Average Relative Humidity',
    'TP_PT24H_ACC': 'Total Precipitation',
    'WG_PT24H_MAX': 'Maximum Wind Gust'
}

def get_harbors_from_config(config_path):
    """Get list of harbors from the configuration file."""
    with open(config_path, 'r') as f:
        harbors_config = json.load(f)
    return list(harbors_config.keys())

def calculate_percentiles_for_harbor_predictor(harbor_name, pred):
    """Calculate 95th percentile for observations and predictions for a specific harbor and predictor."""
    
    # Define directories
    data_dir = f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
    out_dir = f'/home/ubuntu/data/ML/results/OCEANIDS/{harbor_name}/'
    
    # Define filenames
    train_fname = f'training_data_oceanids_{harbor_name}-sf-addpreds.csv.gz'
    pred_fname = f'predictions_{harbor_name}_{pred}_era5.csv'
    
    results = {
        'harbor': harbor_name,
        'predictor': pred,
        'description': PREDICTOR_DESCRIPTIONS.get(pred, pred),
        'obs_95th_percentile': None,
        'pred_95th_percentile': None,
        'error': None
    }
    
    try:
        # Load predictand mappings
        with open('predictand_mappings.json', 'r') as f:
            mappings = json.load(f)
        predictand_mappings = {key: mappings[key]["parameter"] for key in mappings}
        
        # Get columns to drop
        selected_value = predictand_mappings[pred]
        keys_to_drop = [key for key in predictand_mappings if key != pred]
        values_to_drop = [val for key, val in predictand_mappings.items() if key != pred]
        
        # Load training data config to select proper training columns
        with open('training_data_config.json', 'r') as file:
            config_data = commentjson.load(file)
        columns = config_data['training_columns']
        
        # Filter the columns for predictor and related variables
        filtered_columns = []
        for col in columns:
            drop = any(drop_key in col for drop_key in keys_to_drop)
            drop = drop or any(drop_val in col for drop_val in values_to_drop)
            if not drop:
                filtered_columns.append(col)
        
        # Read training data using the filtered columns
        training_data_path = data_dir + train_fname
        if not os.path.exists(training_data_path):
            results['error'] = f"Training data not found: {training_data_path}"
            return results
            
        df_train = pd.read_csv(training_data_path, usecols=filtered_columns)
        df_train = df_train.dropna(axis=1, how='all')
        
        # Calculate 95th percentile for observations from training data
        if pred in df_train.columns:
            obs_data = df_train[pred].dropna()
            # Convert Kelvin to Celsius for temperature predictors if values in Kelvin
            if pred in temp_predictors and obs_data.max() > 200:
                obs_data = obs_data - 273.15
            if len(obs_data) > 0:
                results['obs_95th_percentile'] = np.percentile(obs_data, 95)
            else:
                results['error'] = f"No valid observation data for {pred}"
                return results
        else:
            results['error'] = f"Predictor {pred} not found in training data columns"
            return results
        
        # Load predictions data
        pred_data_path = out_dir + pred_fname
        if not os.path.exists(pred_data_path):
            results['error'] = f"Predictions data not found: {pred_data_path}"
            return results
            
        df_pred = pd.read_csv(pred_data_path)
        
        # Calculate 95th percentile for predictions
        if 'predicted' in df_pred.columns:
            pred_data = df_pred['predicted'].dropna()
            # Convert Kelvin to Celsius for temperature predictors if values in Kelvin
            if pred in temp_predictors and pred_data.max() > 200:
                pred_data = pred_data - 273.15
            if len(pred_data) > 0:
                results['pred_95th_percentile'] = np.percentile(pred_data, 95)
            else:
                results['error'] = f"No valid prediction data"
                return results
        else:
            results['error'] = f"'predicted' column not found in predictions data"
            return results
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

def main():
    """Main function to process all harbors and predictors."""
    startTime = time.time()
    
    # Get list of harbors from config
    harbors_config_path = '/home/ubuntu/ml-oceanids/bin/harbors_config.json'
    harbors = get_harbors_from_config(harbors_config_path)
    
    # Store all results
    all_results = []
    
    print("Calculating 95th percentile values for all harbor-predictor combinations...")
    print("=" * 80)
    
    # Process each harbor and predictor combination
    for harbor in harbors:
        print(f"\nProcessing harbor: {harbor}")
        print("-" * 40)
        
        for pred in PREDICTORS:
            print(f"  Processing predictor: {pred} ({PREDICTOR_DESCRIPTIONS.get(pred, pred)})")
            
            results = calculate_percentiles_for_harbor_predictor(harbor, pred)
            all_results.append(results)
            
            if results['error']:
                print(f"    ERROR: {results['error']}")
            else:
                print(f"    Observations 95th percentile: {results['obs_95th_percentile']:.2f}")
                print(f"    Predictions 95th percentile: {results['pred_95th_percentile']:.2f}")
    
    # Save results to CSV file
    output_file = '/home/ubuntu/ml-oceanids/bin/95th-percentile-values-era5.csv'
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful_results = results_df[results_df['error'].isnull()]
    failed_results = results_df[results_df['error'].notnull()]
    
    print(f"Total combinations processed: {len(all_results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if len(successful_results) > 0:
        print(f"\nResults saved to: {output_file}")
        print("\nSample of successful results:")
        print(successful_results[['harbor', 'predictor', 'obs_95th_percentile', 'pred_95th_percentile']].head(10).to_string(index=False))
    
    if len(failed_results) > 0:
        print(f"\nFailed combinations:")
        for _, row in failed_results.iterrows():
            print(f"  {row['harbor']} - {row['predictor']}: {row['error']}")
    
    executionTime = (time.time() - startTime)
    print(f'\nExecution time: {executionTime:.2f} seconds ({executionTime/60:.2f} minutes)')

if __name__ == "__main__":
    main()