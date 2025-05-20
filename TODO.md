## New plots
- Geopotential/temperature/humidity 925 hPa
- Geopotential/temperature/humidity 850 hPa
- Thunderstorm parameters
    - Bulk shear (WSHEAR_U, WSHEAR_V already avail as parameters)
    - Maximum amplitude of updraft helicity (UH_MAX)
- Convergence ?
- Surface moisture flux divergence
- Soil moisture saturation ?
- Add graupel to prec_clouds plot
- Surface radiation

## Features
- Verify that cache works among different processes opening the same file concurrently
- Add option to prescribe an aggregation function when adding the values on the map so that we can instead use min/max functions.
- For plots with multiple colorbars use the same strategy to first create an axis below the main plot and then divide into two equal parts and place the colorbars inside. The current strategy with divide_axis_for_cbar is not really the best...
- Add input argument to every plot script to give the possibility to only plot a single timestep or a range of timesteps
- Add a progress bar when downloading data
- Download and add proper grib tables, and force `cfgrib` to use them so that we get always the right names! -> For the moment we have to  install the tables from DWD and set ECCODES_DEFINITION_PATH to include them