from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt

import utils
from args import debug, projection, run, level
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name = f"rh_{level}"

if not debug:
    import matplotlib
    matplotlib.use("Agg")


def main():
    logging.info(f"Plotting {variable_name} for projection {projection}.")
    dset = utils.get_files_levels(
        vars=["RELHUM"], projection=projection, run=run, lev_sel=level
    ).squeeze()

    levels_rh = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]

    cmap, norm = utils.get_colormap_norm("rh_mlgx", levels_rh, extend="both")
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, cmap=cmap, norm=norm, levels_rh=levels_rh)

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
        rh_cf_name = utils.find_variable_by_long_name(data, ["Relative Humidity", "Relative humidity"])
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
            data[rh_cf_name],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_rh"],
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            f"Relative Humidity at {level} hPa",
            run,
        )
        an_stats = utils.annotation_stats(
            args["ax"],
            data[rh_cf_name],
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    label="Temperature (C)",
                ),
            )

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [
                cs,
                an_fc,
                an_var,
                an_run,
                an_stats
            ]
        )

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
