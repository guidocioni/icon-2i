from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np

import utils
from args import debug, projection, run, level
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name =f"gph_t_{level}"

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_levels(
        vars=["FI", 'T'], projection=projection, run=run, lev_sel=level
    ).squeeze()
    gph_cf_name = utils.find_variable_by_long_name(dset, ["Geopotential"])
    t_cf_name = utils.find_variable_by_long_name(dset, ["Temperature"])
    dset['z'] = mpcalc.geopotential_to_height(dset[gph_cf_name]).metpy.dequantify()
    dset[t_cf_name] = dset[t_cf_name].metpy.convert_units("degC").metpy.dequantify()

    if level == 850:
        levels_gph = np.arange(
            720, 2160, 20
        )
    elif level == 925:
        levels_gph = np.arange(
            220, 1285, 15
        )
    elif level == 1000:
        levels_gph = np.arange(
            -430, 635, 15
        )
    elif level == 700:
        levels_gph = np.arange(
            2300, 3710, 20
        )
    elif level == 500:
        levels_gph = np.arange(
            4690, 6095, 15
        )
    elif level == 250:
        levels_gph = np.arange(
            8080, 10210, 30
        )
    if level in [1000, 925, 850, 700]:
        levels_temp = np.arange(-40, 50, 1)
    elif level == 500:
        levels_temp = np.arange(-58, 10, 1)
    elif level == 250:
        levels_temp = np.arange(-85, -12, 1)

    cmap, norm = utils.get_colormap_norm("temp_mlgx", levels_temp, extend='both')
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_gph=levels_gph,
        levels_temp=levels_temp
    )

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
        t_cf_name = utils.find_variable_by_long_name(data, ["Temperature"])        
        data['z'].values = mpcalc.smooth_n_point(data['z'].values, n=9, passes=10)
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
            data[t_cf_name],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_temp"],
        )
        c = args["ax"].contour(
            args["x"],
            args["y"],
            data['z'],
            levels=args["levels_gph"],
            colors="white",
            linewidths=1.0,
        )
        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=6)

        maxlabels = utils.plot_maxmin_points(
            args["ax"],
            args["x"],
            args["y"],
            data['z'],
            "max",
            170,
            symbol="H",
            color="royalblue",
            random=True,
        )
        minlabels = utils.plot_maxmin_points(
            args["ax"],
            args["x"],
            args["y"],
            data['z'],
            "min",
            170,
            symbol="L",
            color="coral",
            random=True,
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            f"Geopotential (m) and temperature (C) at {level} hPa",
            run
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
                c,
                labels,
                an_fc,
                an_var,
                an_run,
                maxlabels,
                minlabels,
            ]
        )

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
