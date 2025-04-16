from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

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
variable_name = "snowlmt"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["SNOWLMT"], projection=projection
    )
    snowlmt_cf_name = utils.find_variable_by_grib_param_id(dset, 500128)

    dset[snowlmt_cf_name] = (
        dset[snowlmt_cf_name].metpy.convert_units("m").metpy.dequantify()
    )

    levels_snowlmt = np.arange(0.0, 3000.0, 100.0)

    cmap, norm = utils.get_colormap_norm("snow_acc_wxcharts", levels_snowlmt, extend='max')
    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
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

        cs = args["ax"].contourf(
            args["x"],
            args["y"],
            data[snowlmt_cf_name].fillna(0),
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_snowlmt"],
        )

        c = args["ax"].contour(
            args["x"],
            args["y"],
            data[snowlmt_cf_name],
            levels=[0, 500, 1000, 1500, 2000, 2500],
            colors="white",
            linewidths=1.0,
        )

        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=5)

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "Snow Limit Altitude (above surface)",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            cb = plt.colorbar(
                cs,
                orientation="horizontal",
                label="Snow Limit [m]",
                pad=0.03,
                fraction=0.04,
            )
            cb.minorticks_off()

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [
                cs,
                c,
                labels,
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
