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

variable_name ="h_snow"

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(
        vars=["H_SNOW"], projection=projection, run=run
    )
    cf_var_name = utils.find_variable_by_grib_param_id(dset, 500045)

    dset[cf_var_name] = dset[cf_var_name].metpy.convert_units('cm').metpy.dequantify()
    levels_snow = (.1 , 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200)

    cmap, norm = utils.get_colormap_norm("snow_acc_wxcharts", levels_snow, extend='max')
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background=True, cities=True)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_snow=levels_snow,
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
        cf_var_name = utils.find_variable_by_grib_param_id(data, 500045)
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
            data[cf_var_name],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_snow"],
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "Snow height (cm)",
            run
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    ticks=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100,
                           125, 150, 175, 200],
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
            ]
        )

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
