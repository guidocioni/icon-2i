## New plots
- Geopotential/temperature/humidity 925 hPa
- Geopotential/temperature/humidity 850 hPa
- Geopotential/temperature/humidity 850 hPa
- ThetaE ?
- Thunderstorm parameters ?
    - Storm Relative Helicity
    - Bulk shear (WSHEAR already avail as parameter)
    - Supercell composite
    - Lightning potential index? (LPI)
    - Supercell Detection Index (SDI_2)
- Convergence ?
- Surface moisture flux divergence
- Soil moisture saturation ?
- Add graupel to prec_clouds plot
- SST
- Surface radiation


## Feature
- Verify that cache works among different processes opening the same file concurrently
- Add option to prescribe an aggregation function when adding the values on the map so that we can instead use min/max functions.
- For plots with multiple colorbars use the same strategy to first create an axis below the main plot and then divide into two equal parts and place the colorbars inside. The current strategy with divide_axis_for_cbar is not really the best...
- Add input argument to every plot script to give the possibility to only plot a single timestep or a range of timesteps
- Add a progress bar when downloading data

## Common