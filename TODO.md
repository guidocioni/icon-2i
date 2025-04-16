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

## Common