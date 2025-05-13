#!/usr/bin/env python3
import time, warnings,requests,json,sys,os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# SmarMet-server timeseries (ts) query to fetch ERA5/ERA5D training data for ML OCEANIDS
# ts queries for four grid points around harbor location
# remember to first: conda activate xgb2 and give harbor name as cmd
# (AK 2025)

startTime=time.time()

harbor_name=sys.argv[1]

def filter_points(df,lat,lon,nro,name):
    df0=df.copy()
    filter1 = df0['latitude'] == lat
    filter2 = df0['longitude'] == lon
    df0.where(filter1 & filter2, inplace=True)
    df0.columns=['lat-'+str(nro),'lon-'+str(nro),name+'-'+str(nro)] # change headers      
    df0=df0.dropna()
    return df0

def filter_dataframe(df,grid_points,name):
    df_new=pd.DataFrame()
    for i in range(1, 5):  # For lat1, lon1, ..., lat4, lon4
        lat = grid_points[f"lat-{i}"]
        lon = grid_points[f"lon-{i}"]
        df_filtered = filter_points(df, lat, lon, i, name)
        if df_new.empty:
            df_new = df_filtered
        else:
            df_new = pd.concat([df_new,df_filtered],axis=1,sort=False)
    df_new=df_new.reset_index()
    return df_new

def get_bbox(lat, lon,loc, buffer_degrees):
    #Generate a bounding box around a given lat/lon point
    min_lat = lat - buffer_degrees
    max_lat = lat + buffer_degrees
    min_lon = lon - buffer_degrees
    max_lon = lon + buffer_degrees
    
    # write to config file
    bbox = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon
    }
    mod_dir=f'/home/ubuntu/data/ML/models/OCEANIDS/{loc}/'
    if not os.path.exists(mod_dir):
        os.makedirs(mod_dir)
    bbox_config_file = f'{mod_dir}{loc}_bbox_config.json'
    with open(bbox_config_file, "w") as f:
        json.dump(bbox, f, indent=4)
    
    return f'{min_lon},{min_lat},{max_lon},{max_lat}'

def ts_query(source,bbox,value,start,end,hour,name):
    df=pd.DataFrame()
    query=f'http://{source}/timeseries?bbox={bbox}&param=utctime,latitude,longitude,{value}&starttime={start}&endtime={end}&hour={hour}&format=json&precision=full&tz=utc&timeformat=sql'
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

# Load harbor config file
with open('harbors_config.json', 'r') as file:
    config = json.load(file)

# Access info for a specific harbor
harbor = config.get(harbor_name, {})
latitude = harbor.get('latitude')
longitude = harbor.get('longitude')
start = harbor.get('start')
end = harbor.get('end')

print(f'ERA5 Timeseries queries for {harbor_name}')

data_dir=f'/home/ubuntu/data/ML/training-data/OCEANIDS/{harbor_name}/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# get bbox and save to config file 
bbox = get_bbox(latitude, longitude,harbor_name, buffer_degrees=0.25)

# static, 00 UTC, and 24h aggregation predictors
predictors_00 = [
    #{'anor': 'ANOR-RAD:ERA5:5021:1:0:1:0'}, # Angle of sub-gridscale orography
    #{'z': 'Z-M2S2:ERA5:5021:1:0:1:0'}, # geopotential in m2 s-2
    #{'lsm': 'LC-0TO1:ERA5:5021:1:0:1:0'}, # Land sea mask: 1=land, 0=sea
    #{'sdor': 'SDOR-M:ERA5:5021:1:0:1:0'}, # Standard deviation of orography
    #{'slor': 'SLOR:ERA5:5021:1:0:1:0'}, # Slope of sub-gridscale orography
    #{'tclw':'TCLW-KGM2:ERA5:5021:1:0:1:0'}, # total column cloud liquid water (24h instantanous) 
    #{'tcwv':'TOTCWV-KGM2:ERA5:5021:1:0:1:0'}, # total column water vapor 
    #{'swvl1':'SOILWET-M3M3:ERA5:5021:9:7:1:0'}, # volumetric soil water layer 1 (0-7cm) (24h instantanous)
    #{'swvl2':'SWVL2-M3M3:ERA5:5021:9:1820:1:0'}, # volumetric soil water layer 2 (7-28cm) (24h instantanous)
    #{'swvl3':'SWVL3-M3M3:ERA5:5021:9:7268:1:0'}, # volumetric soil water layer 3 (28-100cm) (24h instantanous)
    #{'swvl4':'SWVL4-M3M3:ERA5:5021:9:25855:1:0'}, # volumetric soil water layer 4 (100-289cm) (24h instantanous)
    ##OLD{'ewss':'sum_t(EWSS-NM2S:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Eastward turbulent surface stress
    ##OLD{'e':'sum_t(EVAP-M:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Evaporation
    ##OLD{'nsss':'sum_t(NSSS-NM2S:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Northward turbulent surface stress
    ##OLD{'slhf':'sum_t(FLLAT-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Surface latent heat flux
    ##OLD{'ssr':'sum_t(RNETSWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Surface net solar radiation
    ##OLD{'str':'sum_t(RNETLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Surface net thermal radiation
    ##OLD{'sshf':'sum_t(FLSEN-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Surface sensible heat flux
    ##OLD{'ssrd':'sum_t(RADGLOA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Surface solar radiation downwards
    ##OLD{'strd':'sum_t(RADLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Surface thermal radiation downwards
    ##OLD{'tp':'sum_t(RR-M:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Total precipitation
    ##OLD{'ttr':'sum_t(RTOPLWA-JM2:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day sum Top net thermal radiation
    ##OLD{'fg10':'max_t(FFG-MS:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day maximum 10m wind gust since previous post-processing (24h aggregation: max value of previous day)
    ##OLD{'mx2t':'max_t(TMAX-K:ERA5:5021:1:0:1:0/24h/0h)'}, # Previous day maximum Maximum temperature in the last 24h
    ##OLD{'mn2t':'min_t(TMIN-K:ERA5:5021:1:0:1:0/24h/0h)'} # Previous day minimum Minimum temperature in the last 24h
]

# era5d predictors
predictors_era5d = [    
    {'ewss':'EWSS-NM2S:ERA5D:5021:1:0:1'}, # Previous day sum Eastward turbulent surface stress
    {'e':'EVAP-M:ERA5D:5021:1:0:1'}, # Previous day sum Evaporation
    {'nsss':'NSSS-NM2S:ERA5D:5021:1:0:1'}, # Previous day sum Northward turbulent surface stress
    {'slhf':'FLLAT-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface latent heat flux
    {'ssr':'RNETSWA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface net solar radiation
    {'str':'RNETLWA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface net thermal radiation
    {'sshf':'FLSEN-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface sensible heat flux
    {'ssrd':'RADGLOA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface solar radiation downwards
    {'strd':'RADLWA-JM2:ERA5D:5021:1:0:0'}, # Previous day sum Surface thermal radiation downwards
    {'tp':'RR-M:ERA5D:5021:1:0:1'}, # Previous day sum Total precipitation
    {'ttr':'RTOPLWA-JM2:ERA5D:5021:1:0:1'}, # Previous day sum Top net thermal radiation
    {'fg10':'FFG-MS:ERA5D:5021:1:0:1'}, # Previous day maximum 10m wind gust since previous post-processing (24h aggregation: max value of previous day)
    {'mx2t':'TMAX-K:ERA5D:5021:1:0:1'}, # Previous day maximum Maximum temperature in the last 24h
    {'mn2t':'TMIN-K:ERA5D:5021:1:0:1'} # Previous day minimum Minimum temperature in the last 24h
]

# 00 and 12 UTC predictors
predictors_0012 = [
    {'u10':'U10-MS:ERA5:5021:1:0:1:0'}, # 10m u-component of wind (6h instantanous)
    {'v10':'V10-MS:ERA5:5021:1:0:1:0'}, # 10m v-component of wind (6h instantanous)
    {'td2':'TD2-K:ERA5:5021:1:0:1:0'}, # 2m dewpoint temperature (6h instantanous)
    {'t2':'T2-K:ERA5:5021:1:0:1:0'}, # 2m temperature (6h instantanous)
    {'msl':'PSEA-HPA:ERA5:5021:1:0:1:0'}, # mean sea level pressure (6h instantanous)
    {'tsea':'TSEA-K:ERA5:5021:1:0:1'}, # sea surface temperature (6h instantanous)
    {'tcc':'N-0TO1:ERA5:5021:1:0:1:0'}, # total cloud cover (6h instantanous)
    {'kx': 'KX:ERA5:5021:1:0:0'}, # K index
    {'t850': 'T-K:ERA5:5021:2:850:1:0'}, # temperature in K, pressure levels 500-850 hPa       
    {'t700': 'T-K:ERA5:5021:2:700:1:0'},  
    {'t500': 'T-K:ERA5:5021:2:500:1:0'},
    {'q850': 'Q-KGKG:ERA5:5021:2:850:1:0'}, # specific humidity in kg/kg, pressure levels 500-850 hPa
    {'q700': 'Q-KGKG:ERA5:5021:2:700:1:0'},
    {'q500': 'Q-KGKG:ERA5:5021:2:500:1:0'},
    {'u850': 'U-MS:ERA5:5021:2:850:1:0'}, # U comp of wind in m/s, pressure levels 500-850 hPa
    {'u700': 'U-MS:ERA5:5021:2:700:1:0'},
    {'u500': 'U-MS:ERA5:5021:2:500:1:0'},
    {'v850': 'V-MS:ERA5:5021:2:850:1:0'}, # V comp of wind in m/s, pressure levels 500-850 hPa
    {'v700': 'V-MS:ERA5:5021:2:700:1:0'},
    {'v500': 'V-MS:ERA5:5021:2:500:1:0'},
    {'z850': 'Z-M2S2:ERA5:5021:2:850:1:0'}, # geopotential in m2 s-2, pressure levels 500-850 hPa
    {'z700': 'Z-M2S2:ERA5:5021:2:700:1:0'},
    {'z500': 'Z-M2S2:ERA5:5021:2:500:1:0'},   
]

source='desm.harvesterseasons.com:8080' # server for timeseries query

# get grid point lats lons 
query=f'http://{source}/timeseries?bbox={bbox}&param=utctime,latitude,longitude,U10-MS:ERA5:5021:1:0:1:0&starttime={start}&endtime={start}&hour=0&format=json&precision=full&tz=utc&timeformat=sql'
response=requests.get(url=query)
results_json=json.loads(response.content)
data = results_json[0]
lats = data.get('latitude', '').strip('[]').split()
lons = data.get('longitude', '').strip('[]').split()
grid_points = {f'lat-{i+1}': lat for i, lat in enumerate(lats)}
grid_points.update({f'lon-{i+1}': lon for i, lon in enumerate(lons)})

# static and 00 parameters
for pardict in predictors_00:
    key, value = list(pardict.items())[0]
    name=key
    # hour 00 
    df_00=ts_query(source,bbox,value,start,end,'0',name)
    df_00_fin=filter_dataframe(df_00,grid_points,name)
    print(df_00_fin)
    df_00_fin.to_csv(f'{data_dir}era5_oceanids_{name}_{harbor_name}.csv',index=False) 

# era5d parameters
for pardict in predictors_era5d:
    key, value = list(pardict.items())[0]
    name=key
    df_era5d=ts_query(source,bbox,value,start,end,'0',name)
    df_era5d_fin=filter_dataframe(df_era5d,grid_points,name)
    cols_to_shift=[name+'-1',name+'-2',name+'-3',name+'-4']
    df_era5d_fin[cols_to_shift]=df_era5d_fin[cols_to_shift].shift(1) # shif data to correct dates
    print(df_era5d_fin)
    df_era5d_fin.to_csv(f'{data_dir}era5_oceanids_{name}_{harbor_name}.csv',index=False) 

# 00 and 12 UTC parameters
for pardict in predictors_0012:
    key, value = list(pardict.items())[0]
    name=key
    name1 = key + '-00'
    name2 = key + '-12'
    # hour 00 
    df_00=ts_query(source,bbox,value,start,end,'0',name1)
    df_00_fin=filter_dataframe(df_00,grid_points,name1)
    print(df_00_fin)
    df_00_fin.to_csv(f'{data_dir}era5_oceanids_{name1}_{harbor_name}.csv',index=False) 
    # hour 12
    df_12=ts_query(source,bbox,value,start,end,'12',name2)
    df_12_fin=filter_dataframe(df_12,grid_points,name2)
    print(df_12_fin)
    df_12_fin.to_csv(f'{data_dir}era5_oceanids_{name2}_{harbor_name}.csv',index=False) 

executionTime=(time.time()-startTime)
print('Execution time in minutes: %.2f'%(executionTime/60))