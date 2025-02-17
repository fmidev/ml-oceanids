#!/bin/env bash
# Make dataset out of cordex data
# Usage: ./make-cordex-data.sh location

LOCATION=$1
MODEL=$2

echo "Adjusting cordex data"
parallel 'sed "s/ \+/,/g" {} > '$MODEL'-'$LOCATION'.csv' ::: pr sfcWind maxWind tasmax tasmin

