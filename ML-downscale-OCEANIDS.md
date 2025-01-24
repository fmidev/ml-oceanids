# ML-downscale ​seasonal/climate forecasts for OCEANIDS

# Predictands

| Predictand | Units | Producer |Temporal resolution|  ML name |
| :------------- |:---|:-----------|:--|:-|
| Average wind speed, 1 hour| m/s |daily mean from hourly data|WS_PT1H_AVG |
|Maximum wind speed, 1 hour |m/s|previous day maximum value from hourly data|WG_PT1H_MAX|


# Predictors

## ERA5 and seasonal forecasts

ERA5 used for fitting, seasonal forecast in predicting


ERA5 data available hourly. 
| Predictor | Units | Producer | Spatial resolution |ML Temporal resolution | ML name |
| :------------- |:---|:-------------|:--|:-|:-|
| 10m u-component of wind | m/s |ERA5|0.25° x 0.25°|00 UTC|u10|
| 10m v-component of wind | m/s |ERA5|0.25° x 0.25°|00 UTC|v10|
| 10m wind gust since previous post-processing  | m/s |ERA5|0.25° x 0.25°|previous day maximum value|fg10|
|2m dewpoint temperature|K|ERA5|0.25° x 0.25°|00 UTC|td2|
|2m temperature|K|ERA5|0.25° x 0.25°|00 UTC|t2|
|Eastward turbulent surface stress|N m-2 s|ERA5|0.25° x 0.25°|previous day 24h sums|ewss|
|Evaporation|m of water equivalent|ERA5|0.25° x 0.25°|previous day 24h sums|e|
|Land-sea mask|-|ERA5|0.25° x 0.25°|static|lsm|
|Mean sea level pressure|Pa|ERA5|0.25° x 0.25°|00 UTC|msl|
|Northward turbulent surface stress|N m-2 s|ERA5|0.25° x 0.25°|previous day 24h sums|nsss|
|Sea surface temperature|K|ERA5|0.25° x 0.25°|00 UTC|tsea|
|Surface latent heat flux|W m-2|ERA5|0.25° x 0.25°|previous day 24h sums|slhf|
|Surface net solar radiation|W m-2|ERA5|0.25° x 0.25°|previous day 24h sums|ssr|
|Surface net thermal radiation|W m-2|ERA5|0.25° x 0.25°|previous day 24h sums|str|
|Surface sensible heat flux|W m-2|ERA5|0.25° x 0.25°|previous day 24h sums|sshf|
|Surface solar radiation downwards|W m-2|ERA5|0.25° x 0.25°|previous day 24h sums|ssrd|
|Surface thermal radiation downwards|W m-2|ERA5|0.25° x 0.25°|previous day 24h sums|strd|
|Total cloud cover|0 to 1|ERA5|0.25° x 0.25°|00 UTC|tcc|
|Total column cloud liquid water|kg m-2|ERA5|0.25° x 0.25°|00 UTC|tclw|
|Total precipitation|m|ERA5|0.25° x 0.25°|previous day 24h sums|tp|

### Predictors for predicting with seasonal forecast

| Predictor | Units | Producer | Spatial resolution | ML Temporal resolution (available SF resolution) | ML name |
| :------------- |:---|:-------------| :--|:-|:-|
| 10m u-component of wind | m/s |||00 UTC (6h instantaneous)|u10|
| 10m v-component of wind | m/s |||00 UTC 6h instantaneous|v10|
| 10m wind gust since previous post-processing  | m/s |||previous day maximum value (24h aggregation)|fg10|
|2m dewpoint temperature|K|||00 UTC (6h instantaneous)|td2|
|2m temperature|K|||00 UTC (6h instantaneous)|t2|
|Eastward turbulent surface stress|N m-2 s|||previous day 24h sums (24h aggregation since beginning of forecast)|ewss|
|Evaporation|m of water equivalent|||previous day 24h sums (24h aggregation since beginning of forecast)|e|
|Land-sea mask|-|||static|lsm|
|Mean sea level pressure|Pa|||00 UTC (6h instantaneous)|msl|
|Northward turbulent surface stress|N m-2 s|||previous day 24h sums (24h aggregation since beginning of forecast)|nsss|
|Sea surface temperature|K|||00 UTC (6h instantaneous)|tsea|
|Surface latent heat flux|W m-2|||previous day 24h sums (24h aggregation since beginning of forecast)|slhf|
|Surface net solar radiation|W m-2|||previous day 24h sums (24h aggregation since beginning of forecast)|ssr|
|Surface net thermal radiation|W m-2|||previous day 24h sums (24h aggregation since beginning of forecast)|str|
|Surface sensible heat flux|W m-2|||previous day 24h sums (24h aggregation since beginning of forecast)|sshf|
|Surface solar radiation downwards|W m-2|||previous day 24h sums (24h aggregation since beginning of forecast)|ssrd|
|Surface thermal radiation downwards|W m-2|||previous day 24h sums (24h aggregation since beginning of forecast)|strd|
|Total cloud cover|0 to 1|||00 UTC (6h instantaneous)|tcc|
|Total column cloud liquid water|kg m-2|||00 UTC (24h instantaneous)|tlwc|
|Total precipitation|m|||previous day 24h sums (24h aggregation since beginning of forecast)|tp|



# Vuosaari

Closest to harbor ERA5 grid points in Table 1. vuosaari
|lat name|lon name|lat deg|lon deg|
|:-:|:-:|:--:|:--:|
|lat-1 |lon-1 |60.00 |25.00|
lat-2 |lon-2 |60.25 |25.25|
lat-3 |lon-3 |60.25 |25.00|
lat-4 |lon-4 |60.00| 25.25|

# Raahe

# Rauma

# Bremerhaven
