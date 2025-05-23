import sys

bbox='24.9459,60.45867,25.4459,59.95867'
harbor='Vuosaari'
FMISID='151028'
lat=60.20867
lon=25.1959

pred_correl_map= {  
    'WG_PT24H_MAX': 'maxWind',
    'TX_PT24H_MAX': 'tasmax',
    'TN_PT24H_MIN': 'tasmin',
    'TP_PT24H_SUM': 'pr'
}
qa_correl_map= {
    'WG_PT24H_MAX': 0.95,
    'TX_PT24H_MAX': 0.95,
    'TN_PT24H_MIN': 0.05,
    'TP_PT24H_SUM': 0.95
}

# pred and model will be set by the main script
pred = None
model = None

# These variables will be updated by the main script
correl_pred = None
quantile_alpha = None
qpred = None
start = '20130628T000000Z'
end = '20241201T000000Z'
starty = start[0:4]
endy = end[0:4]

fname = None # training data filepath
pname = None # prediction data filepath
mdl_name = None # model filepath
fscorepic = None
xgbstudy = None
obsfile = None
test_y = [2019, 2024]
train_y = [2013, 2014, 2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023]

cols_own = None

def set_variables(pred_arg, model_arg):
    global pred, model, correl_pred, quantile_alpha, fname, pname, mdl_name, fscorepic, xgbstudy, obsfile, cols_own
    pred = pred_arg
    model = model_arg
    correl_pred = pred_correl_map[pred]
    quantile_alpha = qa_correl_map[pred]
    fname = f'training_data_oceanids_{model}_{harbor}_{pred}_{starty}-{endy}.csv'
    pname = f'prediction_data_oceanids_{model}-{harbor}_{pred}_{starty}-2100.csv'
    mdl_name = f'mdl_{model}_{pred}_{starty}-{endy}_cordex_{harbor}-quantileerror.txt'
    fscorepic = f'Fscore_{model}_{pred}-cordex-{harbor}-quantileerror.png'
    xgbstudy = f'xgb-{model}-{pred}-cordex-{harbor}-quantileerror'
    obsfile = f'obs-oceanids-{start}-{end}-{model}-{pred}-{harbor}-cordex-quantileerror-daymax.csv'
    cols_own = ['utctime',
                'pr-1', 'pr-2', 'pr-3', 'pr-4',
                'sfcWind-1', 'sfcWind-2', 'sfcWind-3', 'sfcWind-4',
                'maxWind-1', 'maxWind-2', 'maxWind-3', 'maxWind-4',
                'tasmax-1', 'tasmax-2', 'tasmax-3', 'tasmax-4',
                'tasmin-1', 'tasmin-2', 'tasmin-3', 'tasmin-4',
                pred, 'dayofyear', 'year', 'month',
                f'{correl_pred}_sum', f'{correl_pred}_sum_mean', f'{correl_pred}_sum_min', f'{correl_pred}_sum_max',
                f'{pred}_mean', f'{pred}_min', f'{pred}_max',
                f'{correl_pred}_{pred}_diff_mean', f'{correl_pred}_{pred}_diff_min', f'{correl_pred}_{pred}_diff_max'
    ]