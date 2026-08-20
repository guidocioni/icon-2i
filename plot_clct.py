from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt

import utils
from args import debug, projection, run, timesteps
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name ='clct'

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(vars=["CLCT"], projection=projection, run=run, timesteps=timesteps)

    levels_clc = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    cmap, norm = utils.get_colormap_norm("cloud_cover_mlgx", levels=levels_clc, extend='both')

    m, x, y, ax = utils.setup_figure_and_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_clc=levels_clc, cmap=cmap, norm=norm)

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
        clc_cf_name = utils.find_variable_by_long_name(data, ["Total Cloud Cover"])
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
            data[clc_cf_name],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_clc"],
        )
    

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "Cloud Cover (%)",
            run
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(ticks=args["levels_clc"])
            )

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [cs, an_fc, an_var, an_run]
        )

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
