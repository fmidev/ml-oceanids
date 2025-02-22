#!/bin/bash

harbor=$1
predictand=$2
cd /home/ubuntu/ml-oceanids/bin

# Trainind data preprocessing
! [ -s /home/ubuntu/data/ML/training-data/OCEANIDS/${harbor}/training_data_oceanids_${harbor}-sf-addpreds.csv ] && ! [ -s /home/ubuntu/data/ML/training-data/OCEANIDS/${harbor}/training_data_oceanids_${harbor}-sf_2020-clim.csv ] && python ts-era5-oceanids.py $harbor && ./join-training-data.sh $harbor && python add-predictors-oceanids.py $harbor || echo "training data files already done"

# KFold
! [ -s /home/ubuntu/data/ML/models/OCEANIDS/${harbor}/${harbor}_${predictand}_best_split.json ] && python xgb-fit-KFold-era5-oceanids.py $harbor $predictand || echo "Already done KFold best split"

# Optuna
! [ -s /home/ubuntu/data/ML/models/OCEANIDS/${harbor}/hyperparameters_${harbor}_${predictand}.json ] && python xgb-fit-optuna-era5-oceanids.py $harbor $predictand || echo "Already done Optuna"

# XGBoost fitting
! [ -s /home/ubuntu/data/ML/models/OCEANIDS/${harbor}/mdl_${harbor}_${predictand}_xgb_era5_oceanids-QE.json ] && python xgb-fit-era5-oceanids.py $harbor $predictand || echo "Already fitted a model"

# Shap analysis
#! [ -s /home/ubuntu/data/ML/results/OCEANIDS/${harbor}/shap_${harbor}_${predictand}_xgb_era5_oceanids-QE.png ] && 
python xgb-analysis-shap-oceanids.py $harbor $predictand

# Fscore analysis
#! [ -s /home/ubuntu/data/ML/results/OCEANIDS/${harbor}/fscore_${harbor}_${predictand}_xgb_era5_oceanids-QE.png ] && 
python xgb-analysis-fscore-oceanids.py $harbor $predictand
