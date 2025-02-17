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

# Maps remain the same for all harbors
qa_correl_map = {
    'WG_PT24H_MAX': 0.95,
    'TX_PT24H_MAX': 0.95,
    'TN_PT24H_MIN': 0.05,
    'TP_PT24H_SUM': 0.95,
    'RH_PT24H_AVG': 0.95
}

pred_correl_map = {
    'WG_PT24H_MAX': 'fg10',
    'TX_PT24H_MAX': 'mx2t',
    'TN_PT24H_MIN': 'mn2t',
    'TP_PT24H_SUM': 'tp',
    'RH_PT24H_AVG': 'r'
}

class HarborPredictor:
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
        
        self.pred = None
        self.correl_pred = None
        self.quantile_alpha = None
        self.qpred = None
        self.fname = None
        self.obsfile = None
        self.fscorepic = None
        self.mdl_name = None
        self.xgbstudy = None
        self.cols_own = None
        self.test_y = None
        self.train_y = None


    def set_variables(self, pred_arg):
        self.pred = pred_arg
        self.correl_pred = pred_correl_map[pred_arg]
        self.quantile_alpha = qa_correl_map[pred_arg]
        self.qpred = f'max_t({pred_arg}/24h/0h)'
        self.fname = f'training_data_oceanids_{self.harbor}-sf_{self.starty}-2023-{pred_arg}.csv'
        self.obsfile = f'obs-oceanids-{self.start}-{self.end}-all-{self.harbor}-quantileerror-fe-daymean-daymax.csv'
        self.fscorepic = f'Fscore_{pred_arg}-sf-{self.harbor}-quantileerror-fe.png'
        self.mdl_name = f'mdl_{pred_arg}_{self.starty}-{self.endy}_sf_{self.harbor}_quantileerror-fe.txt'
        self.xgbstudy = f'xgb-{pred_arg}-{self.harbor}-sf-quantileerror-fe'

        # Set columns list (same as original)
        self.cols_own = ['utctime', self.pred, 'dayOfYear']
        # ... add all the columns as in the original script ...
        
        # Load best split if it exists
        split_file = f'/home/ubuntu/data/ML/models/OCEANIDS/best_split_{self.pred}_{self.harbor}_{self.starty}-{self.endy}.json'
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                best_split = json.load(f)
                self.train_y = best_split['train_years']
                self.test_y = best_split['test_years']
        else:
            self.test_y = [2015, 2019]
            self.train_y = [2013, 2014, 2016, 2017, 2018, 2020, 2021, 2022, 2023]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python harbor_predictor.py <harbor_name>")
        sys.exit(1)
    
    harbor_name = sys.argv[1]
    predictor = HarborPredictor(harbor_name)
    # Use predictor.set_variables(pred_type) to initialize for specific prediction
