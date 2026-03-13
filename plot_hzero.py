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

variable_name = "hzero"

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(f"Plotting {variable_name} for projection {projection}.")
    dset = utils.get_files_sfc(vars=["HZEROCL"], projection=projection, run=run)
    hzero_cf_name = utils.find_variable_by_long_name(
        dset, ["Height of 0 degree Celsius isotherm above msl", "Geometrical height above ground"]
    )

    dset[hzero_cf_name] = (
        dset[hzero_cf_name].metpy.convert_units("m").metpy.dequantify()
    )

    levels_hzero = np.concatenate(
        [
            np.arange(0.0, 800.0, 50.0),
            np.arange(800.0, 3000.0, 100.0),
            np.arange(3000.0, 4000.0, 250.0),
            np.arange(4000.0, 6500, 500.0),
        ]
    )

    cmap, norm = utils.get_colormap_norm("temp_mlgx", levels_hzero, extend="max")
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background="Canvas/World_Dark_Gray_Base")

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_hzero=levels_hzero,
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
        hzero_cf_name = utils.find_variable_by_long_name(
            data, ["Height of 0 degree Celsius isotherm above msl", "Geometrical height above ground"]
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
            data[hzero_cf_name].fillna(0),
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_hzero"],
        )

        c = args["ax"].contour(
            args["x"],
            args["y"],
            data[hzero_cf_name],
            levels=[0, 500, 1000, 2000, 3000, 3500, 4000, 5000, 5500, 6000],
            colors="gray",
            linewidths=1.0,
        )

        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=7)

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "Freezing level (above surface, m)",
            run,
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    ticks=args["levels_hzero"][::2]
                ),
            )

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
    utils.run_main_with_timing(main)
