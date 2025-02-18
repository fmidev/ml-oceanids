# ml-oceanids
Machine learning actions for OCEANIDS Horizon Europe project - downscaling climate and seasonal predictions with station data. See below for more details.

# Training models to forecast several target parameters with gradient boosting for harbors in the OCEANIDS project

The code presented here reproduces the data and model training and prediction workflows used in the OCEANIDS project to predict several target parameters for selected locations. The models are trained with ERA5 reanalysis data while seasonal forecasts are used in prediction.

## System requirements
Python version 3.12.7 in the UNIX/Linux environment was used in this project.

The time it takes to download training data or run the model training dependes f.ex. on the number of locations, number of predictors, selected hyperparameters, etc. For 4 locations, XX predictors and hyperparameters used, it took approximately XX hours to train the model, with 64 CPU cores and 228G memory. With the fitted model, predicting target parameter from seasonal forecast data takes around XX hours with a similar setup.

## Dependencies
To create xgb2 environment used in this project, check out the `xgb2.yml` file.

To download the seasonal forecast data etc from the Climate Data Store, the CDS API client needs to be installed https://cds.climate.copernicus.eu/how-to-api. You will need to register for an ECMWF account to download data from CDS.

Instructions for Optuna and Optuna Dashboard at https://optuna.org.

For each step it is adviced to use the GNU Screen, downloading the data and running the model training/prediction takes time.

The Climate Data Operator (CDO) software is used in predicting the target parameters for handling the input/output grib files https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-30001.1.

We use the GNU parallel: Tange, O., 2018. GNU Parallel 2018. Available at: https://doi.org/10.5281/zenodo.1146014

## Downloading the predictors and predictand data

To train an XGBoost model, observational data is required as the predictand (target parameter) in fitting, such as wind gust, temperature or precipitation. We use ERA5 and ERA5D (daily statistics) reanalysis data and derived features as predictors (input variables): time series data from the four ERA5 grid points closest to the observation site is retrieved from our Smartmet-server at https://desm.harvesterseasons.com/grid-gui with its Timeseries API. Details of all the predictands and predictors available are provided in Tables 1, 2, and 3 at the end of this file.

For training the model you will need a table of the predictand and all predictors in the nearest four grid points around chosen location for the whole time period as input. We have several time series scripts in Python that use the request module to make http-requests to our SmartMet server (https://desm.harvesterseasons.com/grid-gui) Time Series API (https://github.com/fmidev/smartmet-plugin-timeseries). Use these scripts to get time series from ERA5 and ERA5D, and target parameter observations. To run the time series (ts) scripts, you will need to define `harbors_config.json` with name of location, corresponding latitude/longitude, and observation period start and end (note: these should follow the Smartmet-server Timeseries query formatting). Output is a csv file for each parameter. Check the directory structures defined in the scripts.

You need to download the predictand data aka the observation time series for your selected location and save them in a csv file (file name should follow: `obs-oceanids-{harbor_name}.csv`). We run the `ts-obs-oceanids.py` script to ts query observations for Finnish stations but this needs fmi-apikey which is not shared outside organisation.

### OPTION 1
The following steps can be run from script fit-era5-oceanis.sh, or separately as described in Option 2. Example usage for several locations and predictands: `parallel -j1 ./fit-era5-oceanids.sh {\1} {\2} ::: Raahe Rauma Vuosaari ::: WG_PT24H_MAX TA_PT24H_MAX TA_PT24H_MIN TP_PT24H_ACC`

### OPTION 2
To download the ERA5 and ERA5D predictor data, run the `ts-era5-oceanids.py`. It fetches the static, 24h accumulated/max/min, and 00 and 12 UTC hourly time series data, saves them per predictor as csv files. Example usage: `python ts-era5-oceanids.py Vuosaari`.

To combine all predictor CSV files into a single training data input file, run the script `join-training-data.sh.` Example usage: `./join-training-data.sh Vuosaari`.

To get the ERA5/ERA5D derived or other additional predictors, run `add-predictors-oceanids.py`. Example usage: `python add-predictors-oceanids.py Vuosaari`. 

To plot the location and four nearest grid points on map, run `plot-era5-oceanids.py`. Example usage: `python plot-era5-oceanids.py Vuosaari`. 

![Training locations](Raahe_training-locs.png)
Figure 1 Example: Training locations 1 to 4, along with the Raahe observation site (red).

## Training the model

These scripts use config files `harbors_config.json` and `training_data_config.json` where latter defines the column headers for predictors used in training the model. Also, KFold run creates location-specific config files for best train/validation dataset split (by years) and Optuna run creates location-specific config files for hyperparameters.

First, to perform the K-Fold cross-validation (split input dataset to optimal training and testing sets by years), run `xgb-fit-KFold-era5-oceanids.py`. Result is printed to terminal and best split is written to location specific config file. Example usage: `python xgb-fit-KFold-era5-oceanids.py Vuosaari WG_PT24H_MAX`. 

Second, to perform the Optuna hyperparameter tuning (https://optuna.org/), run `xgb-fit-optuna-era5-oceanids.py`. Check the results on your Optuna Dashboard view. Example usage: `python xgb-fit-optuna-era5-oceanids.py Vuosaari WG_PT24H_MAX`.

And then, to train the model with tuned hyperparameters, run `xgb-fit-era5-oceanids.py`. The fitted model is saved as a json file. Example usage: `python xgb-fit-era5-oceanids.py Vuosaari WG_PT24H_MAX`. If you didn't use KFold or Optuna, you need to specify config files for hyperparameters and train/validation years split before running this script. 

<!--To create the cross-correlation matrix and correlation bar chart figures, run `cross-correlation-swi2.py`.-->

To create the F-score (feature importance) figure, run `xgb-analysis-fscore-oceanids.py`. To create the mean absolute SHAP values figure, run `xgb-analysis-shap-oceanids.py`.

## Predicting target parameters
Scripts for predicting target parameters can be found from https://github.com/fmidev/harvesterseasons-smartmet/tree/destine/bin. Predicting target parameters requires first downloading the data and pre-processing as all input data must be re-gridded. `get-seasonal.sh` downloads (latest or user-specified start month+year) the seasonal forecast (https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview) and other necessary data for the European area. The script performs also statistical bias-adjusting and downscaling for several parameters. Preprocessing uses the GNU parallel and CDO.

Predicting with the trained model happens with `run-xgb-predict-oceanids.sh`, and `xgb-predict-oceanids.py`, with input gribs remapped to ERA5 grid and selecting the four grid points used in training. The Python script uses Xarray to join different input grids into one data frame that includes all time steps for each input in the target grid. Then prediction for target parameter is made with XGBoost predict with the previously trained model. Ready csv file is then returned to the bash script that combines the results for all the 51 ensemble members to a single csv output file. 

## Predictands

|Predictand|ML name|Data|Units|Description|
|:-|:-|:-|:-|:-|
|Daily greatest wind gust speed|WG_PT24H_MAX|WG_PT1H_MAX|m s-1|Previous day 24h maximum value from 1 hour greatest wind gust speed|
||TP_PT24H_SUM||||
||TX_PT24H_MAX||||
||TN_PT24H_MIN||||

Table 1 Predictands used in this project.

## Predictors ERA5, ERA5D and seasonal forecast

All available predictors listed in Tables 2 (training the model) and 3 (predicting target parametes), with those used in training bolded.

### ERA5, ERA5D, and derived predictors

| ML name |Predictor | Units | Producer | Spatial resolution |ML Temporal resolution |  Smartmet name |
| :-- |:---|:-------------|:--|:-|:-|:-|
|anor| Angle of sub-gridscale orography| |ERA5|0.25° x 0.25°|static (00 UTC)|ANOR-RAD|
|z|Geopotential|m2 s-2  |ERA5|0.25° x 0.25°|static (00 UTC)|Z-M2S2|
|lsm| Land sea mask  |1=land, 0=sea |ERA5|0.25° x 0.25°|static (00 UTC)|LC-0TO1|
|sdor|Standard deviation of orography  |  |ERA5|0.25° x 0.25°|static (00 UTC)|SDOR-M|
|slor| Slope of sub-gridscale orography |  |ERA5|0.25° x 0.25°|static (00 UTC)|SLOR|
|tclw|Total column cloud liquid water |  |ERA5|0.25° x 0.25°|00 UTC|TCLW-KGM2|
|tcwv|Total column water vapour |  |ERA5|0.25° x 0.25°|00 UTC|TOTCWV-KGM2|
|swvl1|Volumetric soil water layer 1 (0-7cm) |  |ERA5|0.25° x 0.25°|00 UTC|SOILWET-M3M3|
|swvl2|Volumetric soil water layer 2 (7-28cm) |  |ERA5|0.25° x 0.25°|00 UTC|SWVL2-M3M3|
|swvl3|Volumetric soil water layer 3 (28-100cm) |  |ERA5|0.25° x 0.25°|00 UTC|SWVL3-M3M3|
|swvl4| Volumetric soil water layer 4 (100-289cm)|  |ERA5|0.25° x 0.25°|00 UTC|SWVL4-M3M3|
|ewss|Eastward turbulent surface stress|N m-2 s|ERA5D|0.25° x 0.25°|previous day 24h sums|EWSS-NM2S|
|e|Evaporation|m of water equivalent|ERA5D|0.25° x 0.25°|previous day 24h sums|EVAP-M|
|nsss|Northward turbulent surface stress|N m-2 s|ERA5D|0.25° x 0.25°|previous day 24h sums|NSSS-NM2S|
|slhf|Surface latent heat flux|W m-2|ERA5D|0.25° x 0.25°|previous day 24h sums|FLLAT-JM2|
|ssr|Surface net solar radiation|W m-2|ERA5D|0.25° x 0.25°|previous day 24h sums|RNETSWA-JM2|
|str|Surface net thermal radiation|W m-2|ERA5D|0.25° x 0.25°|previous day 24h sums|RNETLWA-JM2|
|sshf|Surface sensible heat flux|W m-2|ERA5D|0.25° x 0.25°|previous day 24h sums|FLSEN-JM2|
|ssrd|Surface solar radiation downwards|W m-2|ERA5D|0.25° x 0.25°|previous day 24h sums|RADGLOA-JM2|
|strd|Surface thermal radiation downwards|W m-2|ERA5D|0.25° x 0.25°|previous day 24h sums|RADLWA-JM2|
|tp|Total precipitation|m|ERA5D|0.25° x 0.25°|previous day 24h sums|RR-M|
|ttr|Top net thermal radiation||ERA5D|0.25° x 0.25°|previous day 24h sums|RTOPLWA-JM2|
|fg10| 10m wind gust since previous post-processing  | m s-1 |ERA5D|0.25° x 0.25°|previous day 24h maximum value|FFG-MS|
|mx2t| Maximum 2m temperature since previous post-processing | K |ERA5D|0.25° x 0.25°|previous day 24h maximum value|TMAX-K|
|mn2t|Minimum 2m temperature since previous post-processing | K |ERA5D|0.25° x 0.25°|previous day 24h minimum value|TMIN-K|
|u10-00, u10-12| 10m u-component of wind | m s-1 |ERA5|0.25° x 0.25°|00 and 12 UTC|U10-MS|
|v10-00, v10-12| 10m v-component of wind | m s-1 |ERA5|0.25° x 0.25°|00 and 12 UTC|V10-MS|
|td2-00, td2-12|2m dewpoint temperature|K|ERA5|0.25° x 0.25°|00 and 12 UTC|TD2-K|
|t2-00, t2-12| 2m temperature|K|ERA5|0.25° x 0.25°|00 and 12 UTC|T2-K|
|msl-00, msl-12| Mean sea level pressure|Pa|ERA5|0.25° x 0.25°|00 and 12 UTC|PSEA-HPA|
|tsea-00, tsea-12| Sea surface temperature|K|ERA5|0.25° x 0.25°|00 and 12 UTC|TSEA-K|
|tcc-00, tcc-12|Total cloud cover|0 to 1|ERA5|0.25° x 0.25°|00 and 12 UTC|N-0TO1|
|kx-00, kx-12|K index||ERA5|0.25° x 0.25°|00 and 12 UTC|KX|
|t850-00, t850-12|Temperature at 850 hPa|K|ERA5|0.25° x 0.25°|00 and 12 UTC|T-K|
|t700-00, t700-12|Temperature at 700 hPa|K|ERA5|0.25° x 0.25°|00 and 12 UTC|T-K|
|t500-00, t500-12|Temperature at 500 hPa|K|ERA5|0.25° x 0.25°|00 and 12 UTC|T-K|
|q850-00, q850-12|Specific humidity at 850 hPa|kg kg-1|ERA5|0.25° x 0.25°|00 and 12 UTC|Q-KGKG|
|q700-00, q700-12|Specific humidity at 700 hPa|kg kg-1|ERA5|0.25° x 0.25°|00 and 12 UTC|Q-KGKG|
|q500-00, q500-12|Specific humidity at 500 hPa|kg kg-1|ERA5|0.25° x 0.25°|00 and 12 UTC|Q-KGKG|
|u850-00, u850-12|U-component of wind at 850 hPa|m s-1|ERA5|0.25° x 0.25°|00 and 12 UTC|U-MS|
|u700-00, u700-12|U-component of wind at 700 hPa|m s-1|ERA5|0.25° x 0.25°|00 and 12 UTC|U-MS|
|u500-00, u500-12|U-component of wind at 500 hPa|m s-1|ERA5|0.25° x 0.25°|00 and 12 UTC|U-MS|
|v850-00, v850-12|V-component of wind at 850 hPa|m s-1|ERA5|0.25° x 0.25°|00 and 12 UTC|V-MS|
|v700-00, v700-12|V-component of wind at 700 hPa|m s-1|ERA5|0.25° x 0.25°|00 and 12 UTC|V-MS|
|v500-00, v500-12|V-component of wind at 500 hPa|m s-1|ERA5|0.25° x 0.25°|00 and 12 UTC|V-MS|
|z850-00, z850-12|Geopotential at 850 hPa|m2 s-2|ERA5|0.25° x 0.25°|00 and 12 UTC|Z-M2S2|
|z700-00, z700-12|Geopotential at 700 hPa|m2 s-2|ERA5|0.25° x 0.25°|00 and 12 UTC|Z-M2S2|
|z500-00, z500-12|Geopotential at 500 hPa|m2 s-2|ERA5|0.25° x 0.25°|00 and 12 UTC|Z-M2S2|

Table 2 ERA5/ERA5D and derived predictors for training the models.
### Seasonal forecast and derived predictors 

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

Table 3. Seasonal forecast and other predictors used in predicting target parameters. 