bbox='7.8261,52.7832,9.3261,54.2832'
harbor='Bremerhaven'
FMISID='004885'
lat=53.53321
lon=8.576089
pred='TP_PT24H_SUM'
correl_pred='sfcWind'
qpred='max_t('+pred+'/24h/0h)'
start='20000101T000000Z'
end='20230831T000000Z' 
starty=start[0:4]
endy=end[0:4]

fname ='ece3-Bremerhaven-'+pred+'.csv' # training input data file
mdl_name='mdl_'+pred+'_2000-2023_ece3_Bremerhaven-quantileerror.txt'
fscorepic='Fscore_'+pred+'-ece3-Bremerhaven-quantileerror.png'
xgbstudy='xgb-'+pred+'-ece3-Bremerhaven-quantileerror'
obsfile='obs-oceanids-'+start+'-'+end+'-'+pred+'-'+harbor+'-ece3-quantileerror-daymax.csv'
test_y=[2014, 2016, 2018, 2021, 2022]
train_y= [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2015, 2017, 2019, 2020, 2023]

cols_own=['utctime',
          #'lat-1','lon-1','lat-2','lon-2','lat-3','lon-3','lat-4','lon-4',
          'pr-1','pr-2','pr-3','pr-4',
          'sfcWind-1','sfcWind-2','sfcWind-3','sfcWind-4',
          'tasmax-1','tasmax-2','tasmax-3','tasmax-4',
          'tasmin-1','tasmin-2','tasmin-3','tasmin-4',
          pred,'dayofyear','year','month',
          f'{correl_pred}_sum',f'{correl_pred}_sum_mean',f'{correl_pred}_sum_min',f'{correl_pred}_sum_max',
          f'{pred}_mean',f'{pred}_min',f'{pred}_max',
          f'{correl_pred}_{pred}_diff_mean',f'{correl_pred}_{pred}_diff_min',f'{correl_pred}_{pred}_diff_max'
]
