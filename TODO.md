## New plots
- Geopotential/temperature/humidity 925 hPa
- Geopotential/temperature/humidity 850 hPa
- Convergence ?
- Surface moisture flux divergence
- Soil moisture saturation ?
- Add graupel to prec_clouds plot
- Surface radiation

## Features
- Verify that cache works among different processes opening the same file concurrently
- Add option to prescribe an aggregation function when adding the values on the map so that we can instead use min/max functions.
- Add a progress bar when downloading data
- Download and add proper grib tables, and force `cfgrib` to use them so that we get always the right names! -> For the moment we have to  install the tables from DWD and set ECCODES_DEFINITION_PATH to include them