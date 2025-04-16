from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import xarray as xr

import utils
from definitions import (
    chunks_size,
    figsize_x,
    figsize_y,
    logging,
    options_savefig,
    processes,
)

args = utils.parse_arguments()
debug = args.debug
projection = args.projection
variable_name = "prec_clouds"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["RAIN_GSP", "RAIN_CON", "SNOW_GSP", "SNOW_CON", "PMSL"],
        projection=projection,
    )
    # We need to parse cloud layers separately for the moment
    dset_high_clouds = utils.get_files_levels(["CLCH"], projection=projection)
    dset_low_clouds = utils.get_files_levels(["CLCL"], projection=projection)
    dset['clc_h'] = dset_high_clouds.to_dataarray().squeeze()
    dset['clc_l'] = dset_low_clouds.to_dataarray().squeeze()
    #
    rain_gsp_cf_name = utils.find_variable_by_grib_param_id(dset, 500134)
    rain_con_cf_name = utils.find_variable_by_grib_param_id(dset, 500137)
    snow_gsp_cf_name = utils.find_variable_by_long_name(dset, "Large-scale snowfall water equivalent")
    snow_con_cf_name = utils.find_variable_by_long_name(dset, "Convective snowfall water equivalent")
    pmsl_cf_name = utils.find_variable_by_grib_param_id(dset, 500002)
    # De-accumulate precipitation
    rain_acc = dset[rain_gsp_cf_name] + dset[rain_con_cf_name]
    snow_acc = dset[snow_gsp_cf_name] + dset[snow_con_cf_name]
    rain = rain_acc.differentiate(coord="step", datetime_unit="h")
    snow = snow_acc.differentiate(coord="step", datetime_unit="h")
    rain = xr.DataArray(rain, name="rain_rate")
    snow = xr.DataArray(snow, name="snow_rate")
    dset = xr.merge([dset.drop_vars([rain_gsp_cf_name, rain_con_cf_name, snow_gsp_cf_name, snow_con_cf_name]), rain, snow])

    # Convert units
    dset[pmsl_cf_name] = dset[pmsl_cf_name].metpy.convert_units("hPa").metpy.dequantify()

    levels_rain = (
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        60.0,
        80.0,
        100.0,
        120.0,
    )
    levels_snow = (
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5,
        7.5,
        10.0,
        15.0,
    )
    levels_clouds = (30, 50, 80, 90, 100)
    levels_mslp = np.arange(
        dset[pmsl_cf_name].min().astype("int"), dset[pmsl_cf_name].max().astype("int"), 3.0
    )

    cmap_snow, norm_snow = utils.get_colormap_norm('snow', levels_snow, extend='max')
    cmap_rain, norm_rain = utils.get_colormap_norm("prec", levels_rain, extend='max')
    cmap_low_clouds, norm_low_clouds = utils.get_colormap_norm("clouds", levels_clouds, extend='max')
    cmap_high_clouds, norm_high_clouds = utils.get_colormap_norm("clouds_orange", levels_clouds, extend='max')

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)
    m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        levels_mslp=levels_mslp,
        levels_rain=levels_rain,
        levels_snow=levels_snow,
        levels_clouds=levels_clouds,
        cmap_rain=cmap_rain,
        cmap_snow=cmap_snow,
        cmap_low_clouds=cmap_low_clouds,
        norm_rain=norm_rain,
        norm_snow=norm_snow,
        norm_low_clouds=norm_low_clouds,
        cmap_high_clouds=cmap_high_clouds,
        norm_high_clouds=norm_high_clouds
    )

    logging.info("Pre-processing finished, launching plotting scripts")
    if debug:
        plot_files(dset.isel(step=slice(0, 2)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and utils.processes
        dss = utils.chunks_dataset(dset, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(processes)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    first = True
    for step in dss["step"]:
        data = dss.sel(step=step).copy()
        pmsl_cf_name = utils.find_variable_by_grib_param_id(data, 500002)
        data[pmsl_cf_name].values = mpcalc.smooth_n_point(data[pmsl_cf_name].values, n=9, passes=10)
        cum_hour = int(
            ((data["valid_time"] - data["time"]).dt.total_seconds() / 3600).item()
        )
        run = data["time"].to_pandas()
        # Build the name of the output image
        filename = utils.find_image_filename(
            projection=projection, variable_name=variable_name, forecast_hour=cum_hour
        )

        cs_rain = args["ax"].contourf(
            args["x"],
            args["y"],
            data["rain_rate"],
            extend="max",
            cmap=args["cmap_rain"],
            norm=args["norm_rain"],
            levels=args["levels_rain"],
            zorder=4,
            antialiased=True,
        )
        cs_snow = args["ax"].contourf(
            args["x"],
            args["y"],
            data["snow_rate"],
            extend="max",
            cmap=args["cmap_snow"],
            norm=args["norm_snow"],
            levels=args["levels_snow"],
            zorder=5,
        )
        cs_clouds = args["ax"].contourf(
            args["x"],
            args["y"],
            data['clc_l'],
            extend="max",
            cmap=args["cmap_low_clouds"],
            norm=args["norm_low_clouds"],
            levels=args["levels_clouds"],
            zorder=3,
            antialiased=True
        )
        cs_high_clouds = args["ax"].contourf(
            args["x"],
            args["y"],
            data['clc_h'],
            extend="max",
            cmap=args["cmap_high_clouds"],
            norm=args["norm_high_clouds"],
            levels=args["levels_clouds"],
            zorder=2,
            antialiased=True,
            alpha=0.3
        )

        c = args["ax"].contour(
            args["x"],
            args["y"],
            data[pmsl_cf_name],
            levels=args["levels_mslp"],
            colors="whitesmoke",
            linewidths=1.5,
            zorder=7,
        )

        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=6)

        maxlabels = utils.plot_maxmin_points(
            args["ax"],
            args["x"],
            args["y"],
            data[pmsl_cf_name],
            "max",
            150,
            symbol="H",
            color="royalblue",
            random=True,
        )
        minlabels = utils.plot_maxmin_points(
            args["ax"],
            args["x"],
            args["y"],
            data[pmsl_cf_name],
            "min",
            150,
            symbol="L",
            color="coral",
            random=True,
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "Clouds, rain, snow and MSLP",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            ax_cbar, ax_cbar_2 = utils.divide_axis_for_cbar(args["ax"])
            cbar_snow = plt.gcf().colorbar(
                cs_snow, cax=ax_cbar, orientation="horizontal", label="Snow [cm/hr]"
            )
            cbar_rain = plt.gcf().colorbar(
                cs_rain, cax=ax_cbar_2, orientation="horizontal", label="Rain [mm/hr]"
            )
            cbar_snow.minorticks_off()
            cbar_rain.minorticks_off()

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [
                c,
                cs_rain,
                cs_snow,
                cs_clouds,
                cs_high_clouds,
                labels,
                an_fc,
                an_var,
                an_run,
                maxlabels,
                minlabels,
            ]
        )

        first = False


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
