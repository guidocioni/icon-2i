from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt

import utils
from args import debug, projection, run
from definitions import (
    chunks_size,
    figsize_x,
    figsize_y,
    logging,
    options_savefig,
    processes,
)

variable_name ='lpi'

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}"
    )
    dset = utils.get_files_sfc(vars=["LPI"], projection=projection, run=run)

    levels_lpi = [0, 1, 2, 5, 10, 20, 30, 50, 100, 200]
    cmap, norm = utils.get_colormap_norm("cape_wxcharts", levels=levels_lpi, extend='max')

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_lpi=levels_lpi, cmap=cmap, norm=norm)

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
        lpi_cf_name = utils.find_variable_by_long_name(data, ["Lightning Potential Index"])
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
            data[lpi_cf_name],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_lpi"],
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "Lighthing Potential Index (flashes per hour)",
            loc="lower left",
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
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
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
