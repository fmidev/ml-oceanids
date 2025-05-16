#!/bin/bash

# Record start time
start_time=$(date +%s)

scenario=$1
harbor=$2
predictand=$3
model=$4


if [ $# -ne 4 ]; then
    echo "Usage: $0 <scenario> <harbor> <model> <predictand>"
    echo "Example: $0 rcp85 Vuosaari TA_PT24H_MIN cnrm_cerfacs_cm5-cnrm_aladin63"
    exit 1
fi

cd /home/ubuntu/ml-oceanids/bin

# Define directories for files
data_dir="/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/${harbor}/"
model_dir="/home/ubuntu/data/ML/models/OCEANIDS/cordex/${harbor}/"
results_dir="/home/ubuntu/data/ML/results/OCEANIDS/cordex/${harbor}/"
mkdir -p "$data_dir" "$model_dir" "$results_dir"

# Step 1: Format CORDEX CSV
! [ -s "/home/ubuntu/data/cordex/${scenario}/cordex_${scenario}_${harbor}_${model}.csv" ] && python cordex-csv-format.py $scenario $harbor $model || echo "CORDEX formatted file already exists"

# Step 2: Transform CORDEX CSV
! [ -s "${data_dir}cordex_${scenario}_${model}_${harbor}.csv" ] && python cordex-csv-transform.py $scenario $harbor $model || echo "CORDEX transformed file already exists"

# Step 3: Add training data
! [ -s "${data_dir}training_data_oceanids_${harbor}_cordex_${scenario}_${model}.csv" ] && python cordex-training-additions.py $scenario $harbor $model || echo "Training data already exists"

# For RCP85 scenario, skip model creation steps
if [ "$scenario" != "rcp85" ]; then
    # Step 4: Fit KFold model to determine best train/test split
    ! [ -s "${model_dir}cordex_${scenario}_${harbor}_${predictand}_${model}_best_split.json" ] && python xgb-fit-KFold-cordex-oceanids.py $scenario $harbor $predictand $model || echo "Already done KFold best split"
    
    # Step 5: Optimize hyperparameters with Optuna
    ! [ -s "${model_dir}hyperparameters_cordex_${scenario}_${harbor}_${predictand}_${model}.json" ] && python xgb-fit-optuna-cordex-oceanids.py $scenario $harbor $predictand $model || echo "Already done Optuna"
    
    # Step 6: Fit final model
    ! [ -s "${model_dir}mdl_${harbor}_${predictand}_${model}_xgb_cordex_${scenario}_oceanids-QE.json" ] && python xgb-fit-cordex-oceanids.py $scenario $harbor $predictand $model || echo "Already fitted a model"
else
    echo "Scenario is rcp85, skipping model creation steps (KFold, Optuna, fitting)"
fi

# Step 7: Run prediction
! [ -s "${results_dir}prediction_cordex_${scenario}_${harbor}_${predictand}_${model}.csv" ] && python xgb-predict-cordex.py $scenario $harbor $predictand $model || echo "Prediction already exists"

# Calculate elapsed time
end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))

# Convert seconds to hours, minutes, seconds
hours=$((elapsed_seconds / 3600))
minutes=$(((elapsed_seconds % 3600) / 60))
seconds=$((elapsed_seconds % 60))

echo "Pipeline complete!"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"