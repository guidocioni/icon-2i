import sys
import os
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import utils
from definitions import (
    IMAGES_DIR,
    chunks_size,
    figsize_x,
    figsize_y,
    logging,
    processes,
    options_savefig
)

debug = False
if not debug:
    import matplotlib

    matplotlib.use("Agg")

# The one employed for the figure name when exported
variable_name = "precip_acc"

logging.info("Starting script to plot " + variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    logging.info("Projection not defined, falling back to default (nord)")
    projection = "nord"
else:
    projection = sys.argv[1]


output_dir = os.path.join(IMAGES_DIR, projection)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created directory: {output_dir}")

def main():
    dset = utils.get_files_sfc(vars=["TOT_PREC"], projection=projection)

    levels_precip = np.concatenate(
        [
            np.arange(1, 51, 1),
            np.arange(52, 102, 2),
            np.arange(110, 210, 10),
            np.arange(230, 530, 30),
            np.arange(550, 1050, 50),
            np.arange(1100, 2100, 100),
        ]
    )
    cmap, norm = utils.get_colormap_norm("prec_acc_wxcharts", levels=levels_precip)

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = utils.get_projection(dset, projection)
    # additional maps adjustment for this map
    m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

    dset = dset.drop_vars(["longitude", "latitude"]).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_precip=levels_precip, cmap=cmap, norm=norm)

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
            data["tp"],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_precip"],
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "Accumulated precipitation and MSLP [hPa]",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            cb = plt.colorbar(
                cs,
                orientation="horizontal",
                label="Accumulated precipitation [mm]",
                pad=0.035,
                fraction=0.035,
                ticks=[1, 5, 10, 15, 25, 35, 50, 100, 200, 500, 1000, 2000]
            )
            cb.minorticks_off()

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [cs, an_fc, an_var, an_run]
        )

        first = False


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
