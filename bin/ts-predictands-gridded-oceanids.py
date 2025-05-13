#!/usr/bin/env python3
import time, warnings,requests,json,sys,os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch EOBS/CERRA/ERA5/ERA5-Land predictand data for ML OCEANIDS
# and compare against synop observations
# (AK 2025)

harbor_name=sys.argv[1]
# Load harbor config file
with open('harbors_config.json', 'r') as file:
    config = json.load(file)

def ts_query(source,latlon,value,start,end,hour,name):
    df=pd.DataFrame()
    query=f'http://{source}/timeseries?latlon={latlon}&param=utctime,latitude,longitude,{value}&starttime={start}&endtime={end}&hour={hour}&format=json&precision=full&tz=utc&timeformat=sql'
    print(query)
    response=requests.get(url=query)
    results_json=json.loads(response.content)
    #print(results_json)
    for i in range(len(results_json)):
        res1=results_json[i]
        for key,val in res1.items():
            if key!='utctime':   
                res1[key]=str(val).strip('[]').split()
    df=pd.DataFrame(results_json)  
    df.columns=['utctime','latitude','longitude',name] # change headers      
    expl_cols=['latitude','longitude',name]
    df=df.explode(expl_cols)
    df.set_index('utctime',inplace=True)
    return(df)


# Access info for a specific harbor
harbor = config.get(harbor_name, {})
latitude = harbor.get('latitude')
longitude = harbor.get('longitude')
latlon = f'{latitude},{longitude}'

start = '20000101T000000Z'
end = '20241231T000000Z'

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

name_file=f'obs-{harbor_name}'

predictand_dict=[
    {'TP_PT24H_ACC_EOBS':'RR-M:EOBS:5074:1:0:1'},       # Total daily amount of rain (mm)
    {'RH_PT24H_AVG_EOBS':'RH-PRCNT:EOBS:5074:1:0:1'},   # Daily mean relative humidity (%)
    {'WS_PT24H_AVG_EOBS':'FF10AVG-MS:EOBS:5074:1:0:1'}, # Daily mean wind speed (m/s)
    {'TA_PT24H_MAX_EOBS':'SUM{TMAX-24-K:EOBS:5074:1:0:1;273.15}'},  # Daily maximum temperature (C -> K)
    {'TA_PT24H_MIN_EOBS':'SUM{TMIN-24-K:EOBS:5074:1:0:1;273.15}'},  # Daily minimum temperature (C -> K)
    {'TP_PT24H_ACC_ERA5D':'MUL{RR-M:ERA5D:5021:1:0:1;1000}'}, # m -> mm
    {'WG_PT24H_MAX_ERA5D':'FFG-MS:ERA5D:5021:1:0:1'},
    {'TA_PT24H_MAX_ERA5D':'TMAX-K:ERA5D:5021:1:0:1'},
    {'TA_PT24H_MIN_ERA5D':'TMIN-K:ERA5D:5021:1:0:1'},
]

source='desm.harvesterseasons.com:8080' # server for timeseries query

for pardict in predictand_dict:
    key, value = list(pardict.items())[0]
    name = key
    csv_path = f'{data_dir}{name_file}-{name}.csv'
    if not os.path.exists(csv_path):
        # Query one year at a time and accumulate the results
        frames = []
        for year in range(2000, 2025):  # from 2000 to 2024 inclusive
            year_start = f"{year}0101T000000Z"
            year_end = f"{year}1231T000000Z"
            df_year = ts_query(source, latlon, value, year_start, year_end, '0', name)
            frames.append(df_year)
            print(df_year)
        df_full = pd.concat(frames)
        df_full = df_full.reset_index()
        print(df_full)
        df_full.to_csv(csv_path, index=False)
    else:
        print(f'File already exists. Skipping ts_query for {name}')
