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

variable_name = "h_snow_change"

if not debug:
    import matplotlib

    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(vars=["H_SNOW"], projection=projection, run=run)
    cf_var_name = utils.find_variable_by_grib_param_id(dset, 500045)
    dset[cf_var_name] = dset[cf_var_name].metpy.convert_units("cm").metpy.dequantify()
    hsnow = dset[cf_var_name] - dset[cf_var_name].isel(step=0)
    hsnow = hsnow.where((hsnow > 0.25) | (hsnow < -0.25))
    dset["hsnow_change"] = hsnow

    levels_snow = (
        -40,
        -30,
        -20,
        -15,
        -10,
        -7,
        -5,
        -3,
        -1,
        1,
        3,
        5,
        7,
        10,
        15,
        20,
        30,
        40,
        50,
        75,
        100,
        150,
        200,
    )

    cmap, norm = utils.get_colormap_norm("snow_change", levels_snow, extend="both")
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background=True)

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
            data["hsnow_change"],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_snow"],
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "Snow depth change [cm] since run beginning",
            run
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
