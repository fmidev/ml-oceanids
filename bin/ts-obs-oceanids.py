#!/usr/bin/env python3
import time, warnings,requests,json,sys,os
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries query to fetch predictand data for ML from Finnish observation stations
# remember to: conda activate xgb2 and give harbor name as cmd
# (AK 2025) 

startTime=time.time()

harbor_name=sys.argv[1]

# Load harbor config file
with open('harbors_config.json', 'r') as file:
    config = json.load(file)

# Access info for a specific harbor
harbor = config.get(harbor_name, {})
latitude = harbor.get('latitude')
longitude = harbor.get('longitude')
start = harbor.get('start')
end=harbor.get('end')

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

obs_file=f'obs-oceanids-{harbor_name}.csv'

# read in fmi-apikey from file
f=open("fmi-apikey","r")
lines=f.readlines()
apikey=lines[0]
f.close()

predictand_dict={
    'WG_PT24H_MAX': 'max_t(WG_PT1H_MAX/24h/0h)',
    'TA_PT24H_MAX': 'max_t(TA_PT1H_MAX/24h/0h)',
    'TA_PT24H_MIN': 'min_t(TA_PT1H_MIN/24h/0h)',
    'TP_PT24H_ACC': 'sum_t(PRAO_PT1H_ACC/24h/0h)'
    }
preds = ",".join(predictand_dict.values())
names = list(predictand_dict.keys())

# Timeseries query
source='data.fmi.fi'
query=f'http://{source}/fmi-apikey/{apikey}/timeseries?latlon={latitude},{longitude}&producer=observations_fmi&precision=double&timeformat=sql&tz=utc&starttime={start}&endtime={end}&hour=0&format=json&param=utctime,latitude,longitude,name,{preds}'
print(query)
#print(query.replace(apikey, 'you-need-fmiapikey-here'))
response=requests.get(url=query)
results_json=json.loads(response.content)
#print(results_json)    
df=pd.DataFrame(results_json)  
df.columns=['utctime','latitude','longitude','name']+names # change headers      
print(df)

# save dataframe as csv
df.to_csv(data_dir+obs_file,index=False) 

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))