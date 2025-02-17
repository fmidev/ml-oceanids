import pandas as pd
import sys

#fmisid = '101061'
#timeframe = '2010-2024'
loc = 'Bremerhaven'
models = ['cnrm_cerfacs_cm5-cnrm_aladin63', 'cnrm_cerfacs_cm5-knmi_racmo22e', 'mohc_hadgem2_es-dmi_hirham5', 'mohc_hadgem2_es-knmi_racmo22e',
            'mohc_hadgem2_es-smhi_rca4', 'ncc_noresm1_m-dmi_hirham5','ncc_noresm1_m-smhi_rca4']


for model in models:
    df1 = pd.read_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{model}-{loc}-cordex-with-obs.csv')
    df2 = pd.read_csv(f'/home/ubuntu/data/synop/fmisid115948_Bremerhaven_2010-2024_rh_24h.csv')

    df2.rename(columns={'date': 'utctime'}, inplace=True)
    df2['utctime'] = pd.to_datetime(df2['utctime'])
    df1['utctime'] = pd.to_datetime(df1['utctime'])
    merged_df = df1.merge(df2, on='utctime', how='left')

    merged_df.to_csv(f'/home/ubuntu/data/ML/training-data/OCEANIDS/cordex/{model}-{loc}-cordex-with-obs.csv', index=False)
    print(f'{model}-{loc} merged successfully.')