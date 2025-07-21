from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt

import utils
from args import debug, projection, run
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name = "uh_max"

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(f"Plotting {variable_name} for projection {projection}")
    dset = utils.get_files_levels(
        vars=["UH_MAX"], projection=projection, run=run
    ).squeeze()

    levels_uh_max = [
        -200,
        -150,
        -120,
        -100,
        -80,
        -60,
        -40,
        -25,
        -10,
        0,
        20,
        40,
        60,
        80,
        100,
        125,
        150,
        175,
        200,
        250,
        300,
        400,
        500,
        600,
    ]
    cmap, norm = utils.get_colormap_norm(
        "snow_change", levels=levels_uh_max, extend="both"
    )

    m, x, y, ax = utils.setup_figure_and_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_uh_max=levels_uh_max, cmap=cmap, norm=norm)

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
        # uh_max_cf_name = utils.find_variable_by_long_name(
        #     data,
        #     [
        #         "Maximum amplitude (positive or negative) of updraft helicity  (over given time interval)"
        #     ],
        # )
        # As there is only 1 variable in the dataset we avoid problems with grib tables and just
        # choose that one
        uh_max_cf_name = list(data.data_vars.keys())[0]
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
            data[uh_max_cf_name].where(
                (data[uh_max_cf_name] <= -10) | (data[uh_max_cf_name] >= 20)
            ),
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_uh_max"],
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "Maximum amplitude of updraft helicity (m$^2$/s$^2$)",
            run,
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
            )

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections([cs, an_fc, an_var, an_run])

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
