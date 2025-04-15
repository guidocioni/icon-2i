from functools import partial
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

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
variable_name = "winter"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["H_SNOW", "RAIN_GSP", "RAIN_CON", "SNOWLMT"], projection=projection
    )
    hsnow_var_name = utils.find_variable_by_grib_param_id(dset, 500045)
    rain_gsp_cf_name = utils.find_variable_by_grib_param_id(dset, 500134)
    rain_con_cf_name = utils.find_variable_by_grib_param_id(dset, 500137)
    snowlmt_cf_name = utils.find_variable_by_grib_param_id(dset, 500128)

    # Compute snow change since beginning
    dset[hsnow_var_name] = (
        dset[hsnow_var_name].metpy.convert_units("cm").metpy.dequantify()
    )
    hsnow = dset[hsnow_var_name] - dset[hsnow_var_name].isel(step=0)
    hsnow = hsnow.where((hsnow > 0.25) | (hsnow < -0.25))
    dset["hsnow_change"] = hsnow
    # Compute rain change
    rain_acc = dset[rain_gsp_cf_name] + dset[rain_con_cf_name]
    rain = rain_acc - rain_acc.isel(step=0)
    dset["rain"] = rain

    dset[snowlmt_cf_name] = (
        dset[snowlmt_cf_name].metpy.convert_units("m").metpy.dequantify()
    )

    dset = dset.drop_vars([rain_gsp_cf_name, rain_con_cf_name, hsnow_var_name])

    levels_snow = (.5, 1, 5, 10, 15, 20, 30, 40, 50, 70, 90, 120)
    levels_rain = (.9, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 80, 120)
    levels_snowlmt = np.arange(0.0, 3000.0, 500.0)
    cmap_snow, norm_snow = utils.get_colormap_norm("snow_bergfex", levels_snow, extend='max')
    cmap_rain, norm_rain = utils.get_colormap_norm("rain_bergfex", levels_rain, extend='max')

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection, cities=True)
    m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap_snow=cmap_snow,
        norm_snow=norm_snow,
        cmap_rain=cmap_rain,
        norm_rain=norm_rain,
        levels_snow=levels_snow,
        levels_rain=levels_rain,
        levels_snowlmt=levels_snowlmt,
    )

    logging.info("Pre-processing finished, launching plotting scripts")
    if debug:
        plot_files(dset.isel(step=slice(-2, -1)), **args)
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
        snowlmt_cf_name = utils.find_variable_by_grib_param_id(data, 500128)
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
            data["rain"],
            extend="max",
            cmap=args["cmap_rain"],
            norm=args["norm_rain"],
            levels=args["levels_rain"],
            alpha=0.8,
            antialiased = True
        )

        cs_snow = args["ax"].contourf(
            args["x"],
            args["y"],
            data["hsnow_change"],
            extend="max",
            cmap=args["cmap_snow"],
            norm=args["norm_snow"],
            levels=args["levels_snow"],
        )

        c = args["ax"].contour(
            args["x"],
            args["y"],
            data[snowlmt_cf_name],
            levels=args["levels_snowlmt"],
            colors="red",
            linewidths=0.5,
        )

        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=5)

        density = 13
        if projection == 'nord':
            density = 8
        elif projection == 'sud':
            density = 7
        elif projection == 'centro':
            density = 5
        # vals = utils.add_vals_on_map(
        #     ax=args["ax"],
        #     var=data["hsnow_change"],
        #     x=args["x"],
        #     y=args["y"],
        #     cmap=args['cmap_snow'],
        #     norm=args['norm_snow'],
        #     density=density,
        #     lcolors=False
        # )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "New snow and rain since forecast start",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            ax_cbar, ax_cbar_2 = utils.divide_axis_for_cbar(args['ax'])
            cbar_snow = plt.gcf().colorbar(cs_snow, cax=ax_cbar, orientation='horizontal',
                                           label='Snow [cm]')
            cbar_rain = plt.gcf().colorbar(cs_rain, cax=ax_cbar_2, orientation='horizontal',
                                           label='Rain [mm]')
            cbar_snow.minorticks_off()
            cbar_rain.minorticks_off()

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [
                cs_rain,
                cs_snow,
                c,
                labels,
                # vals,
                an_fc,
                an_var,
                an_run,
            ]
        )

        first = False


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
