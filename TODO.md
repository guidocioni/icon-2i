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
    - Supercell Detection Index (SPI_2)
- Convergence ?
- Surface moisture flux divergence
- Soil moisture saturation ?
- Add graupel to prec_clouds plot
- SST
- Surface radiation


## Feature
- Verify that cache works among different processes opening the same file concurrently
- Add run as argument parsed in every plotting script and then pass it to get_files_sfc. Set it as default to the closest run (make a function for that)
- Add option to prescribe an aggregation function when adding the values on the map so that we can instead use min/max functions.
- For plots with multiple colorbars use the same strategy to first create an axis below the main plot and then divide into two equal parts and place the colorbars inside. The current strategy with divide_axis_for_cbar is not really the best...

## Common