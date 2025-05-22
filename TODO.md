## New plots
- Convergence ?
- Surface moisture flux divergence
- Add graupel to prec_clouds plot
- Surface radiation

## Features
- Separate the downloading part of the file into a new module so that we have more control on that instead of just using fsspec. We also need to make that parallelize and with a progress bar. Then once the files are downloaded pass them to xarray.
- Verify that cache works among different processes opening the same file concurrently
- Add option to prescribe an aggregation function when adding the values on the map so that we can instead use min/max functions.
- Download and add proper grib tables, and force `cfgrib` to use them so that we get always the right names! -> For the moment we have to  install the tables from DWD and set ECCODES_DEFINITION_PATH to include them