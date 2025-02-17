import json
import os
import sys

def load_harbor_config(harbor_name):
    with open('/home/ubuntu/ml-oceanids/bin/harbors_config.json', 'r') as f:
        configs = json.load(f)
    if harbor_name not in configs:
        raise ValueError(f"Harbor {harbor_name} not found in configuration")
    return configs[harbor_name]

def get_bbox(lat, lon, margin=0.25):
    return f"{lon-margin},{lat+margin},{lon+margin},{lat-margin}"

# CORDEX-specific mappings
pred_correl_map = {
    'WG_PT24H_MAX': 'maxWind',
    'TX_PT24H_MAX': 'tasmax',
    'TN_PT24H_MIN': 'tasmin',
    'TP_PT24H_SUM': 'pr',
    'RH_PT24H_AVG': 'r'
}

qa_correl_map = {
    'WG_PT24H_MAX': 0.95,
    'TX_PT24H_MAX': 0.95,
    'TN_PT24H_MIN': 0.05,
    'TP_PT24H_SUM': 0.95,
    'RH_PT24H_AVG': 0.95
}

class HarborPredictorCordex:
    def __init__(self, harbor_name):
        self.harbor = harbor_name
        config = load_harbor_config(harbor_name)
        self.lat = config['latitude']
        self.lon = config['longitude']
        self.start = config['start']
        self.end = config['end']
        self.bbox = get_bbox(self.lat, self.lon)
        self.starty = self.start[0:4]
        self.endy = self.end[0:4]
        
        # Initialize variables that will be set later
        self.pred = None
        self.model = None
        self.correl_pred = None
        self.quantile_alpha = None
        self.fname = None
        self.pname = None
        self.mdl_name = None
        self.fscorepic = None
        self.xgbstudy = None
        self.obsfile = None
        self.cols_own = None

        # Default test/train years for CORDEX
        self.test_y = [2019, 2024]
        self.train_y = [2013, 2014, 2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023]

    def set_variables(self, pred_arg, model_arg):
        self.pred = pred_arg
        self.model = model_arg
        self.correl_pred = pred_correl_map[self.pred]
        self.quantile_alpha = qa_correl_map[self.pred]
        
        # Set CORDEX-specific filenames
        self.fname = f'training_data_oceanids_{self.model}_{self.harbor}_{self.pred}_{self.starty}-{self.endy}.csv'
        self.pname = f'prediction_data_oceanids_{self.model}-{self.harbor}_{self.pred}_{self.starty}-2100.csv'
        self.mdl_name = f'mdl_{self.model}_{self.pred}_{self.starty}-{self.endy}_cordex_{self.harbor}-quantileerror.txt'
        self.fscorepic = f'Fscore_{self.model}_{self.pred}-cordex-{self.harbor}-quantileerror.png'
        self.xgbstudy = f'xgb-{self.model}-{self.pred}-cordex-{self.harbor}-quantileerror'
        self.obsfile = f'obs-oceanids-{self.start}-{self.end}-{self.model}-{self.pred}-{self.harbor}-cordex-quantileerror-daymax.csv'

        # Set CORDEX-specific columns
        self.cols_own = ['utctime',
                        'pr-1', 'pr-2', 'pr-3', 'pr-4',
                        'sfcWind-1', 'sfcWind-2', 'sfcWind-3', 'sfcWind-4',
                        'maxWind-1', 'maxWind-2', 'maxWind-3', 'maxWind-4',
                        'tasmax-1', 'tasmax-2', 'tasmax-3', 'tasmax-4',
                        'tasmin-1', 'tasmin-2', 'tasmin-3', 'tasmin-4',
                        self.pred, 'dayofyear', 'year', 'month',
                        f'{self.correl_pred}_sum', f'{self.correl_pred}_sum_mean', 
                        f'{self.correl_pred}_sum_min', f'{self.correl_pred}_sum_max',
                        f'{self.pred}_mean', f'{self.pred}_min', f'{self.pred}_max',
                        f'{self.correl_pred}_{self.pred}_diff_mean', 
                        f'{self.correl_pred}_{self.pred}_diff_min', 
                        f'{self.correl_pred}_{self.pred}_diff_max']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python harbor_predictor_cordex.py <harbor_name>")
        sys.exit(1)
    
    harbor_name = sys.argv[1]
    predictor = HarborPredictorCordex(harbor_name)
    # Use predictor.set_variables(pred_type, model_name) to initialize for specific prediction
