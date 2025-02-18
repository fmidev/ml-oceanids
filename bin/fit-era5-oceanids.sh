#!/bin/bash

harbor=$1
predictand=$2

! [ -s /home/ubuntu/data/ML/training-data/OCEANIDS/${harbor}/training_data_oceanids_${harbor}-sf-addpreds.csv ] && ! [ -s /home/ubuntu/data/ML/training-data/OCEANIDS/${harbor}/training_data_oceanids_${harbor}-sf_2020-clim.csv ] && python ts-era5-oceanids.py $harbor && ./join-training-data.sh $harbor && python add-predictors-oceanids.py $harbor || echo "training data files already done"

python xgb-fit-KFold-era5-oceanids.py $harbor $predictand
python xgb-fit-optuna-era5-oceanids.py $harbor $predictand
python xgb-fit-era5-oceanids.py $harbor $predictand
python xgb-analysis-shap-oceanids.py $harbor $predictand
python xgb-analysis-fscore-oceanids.py $harbor $predictand