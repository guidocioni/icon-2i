from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import utils
from args import debug, projection, run
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name = "precip_acc"

if not debug:
    import matplotlib
    matplotlib.use("Agg")


def main():
    logging.info(f"Plotting {variable_name} for projection {projection}.")
    dset = utils.get_files_sfc(vars=["TOT_PREC"], projection=projection, run=run)

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
    cmap, norm = utils.get_colormap_norm(
        "prec_acc_wxcharts", levels=levels_precip, extend="max"
    )

    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background=True, cities=True)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_precip=levels_precip, cmap=cmap, norm=norm)

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
        tp_cf_name = utils.find_variable_by_long_name(
            data,
            [
                "Total Precipitation",
                "Total Precipitation rate (S)",
                "Total Precipitation (Accumulation)",
            ],
        )
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
            data[tp_cf_name],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_precip"],
        )

        density = 17
        fontsize = 6.5
        if projection == "nord":
            density = 10
        elif projection == "sud":
            density = 9
        elif projection == "centro":
            density = 7
        vals = utils.add_vals_on_map(
            ax=args["ax"],
            var=data[tp_cf_name].where(data[tp_cf_name] > 50),
            x=args["x"],
            y=args["y"],
            cmap=args["cmap"],
            norm=args["norm"],
            density=density,
            fontsize=fontsize,
            font_border_color="white",
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "Accumulated precipitation (mm)",
            run,
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    ticks=[1, 5, 10, 15, 25, 35, 50, 75, 100, 200, 500, 1000, 2000]
                ),
            )

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections([cs, an_fc, an_var, an_run, vals])

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
