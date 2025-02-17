#!/bin/bash
# Join all the data files into a single training data file for specific harbor
# Usage: ./join-training-data.sh <harbor>

harbor=$1

cd /home/ubuntu/data/ML/training-data/OCEANIDS/$harbor

paste -d ',' obs-oceanids-$harbor.csv \
<(cut -d ',' -f2,3,5,6,8,9,11,12 era5_oceanids_anor_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_anor_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_lsm_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_sdor_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_slor_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tclw_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tcwv_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_swvl1_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_swvl2_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_swvl3_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_swvl4_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_ewss_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_e_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_nsss_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_slhf_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_ssr_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_str_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_sshf_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_ssrd_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_strd_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tp_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_ttr_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_fg10_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_mx2t_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_mn2t_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u10-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u10-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v10-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v10-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_td2-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_td2-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t2-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t2-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_msl-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_msl-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tsea-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tsea-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tcc-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_tcc-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_kx-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_kx-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t850-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t850-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t700-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t700-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t500-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_t500-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_q850-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_q850-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_q700-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_q700-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_q500-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_q500-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u850-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u850-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u700-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u700-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u500-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_u500-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v850-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v850-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v700-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v700-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v500-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_v500-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z850-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z850-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z700-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z700-12_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z500-00_$harbor.csv) \
<(cut -d ',' -f4,7,10,13 era5_oceanids_z500-12_$harbor.csv) \
> training_data_oceanids_$harbor-sf.csv
