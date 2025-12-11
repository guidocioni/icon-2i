from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np

import utils
from args import debug, projection, run
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name = "gusts"

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(
        vars=["U_10M", "V_10M", "VMAX_10M", "PMSL"], projection=projection, run=run
    )
    vmax_cf_name = utils.find_variable_by_long_name(
        dset,
        [
            "Maximum 10 metre wind gust since previous post-processing",
            "maximum Wind 10m",
            "Time-maximum 10 metre wind gust"
        ],
    )
    pmsl_cf_name = utils.find_variable_by_grib_param_id(dset, 500002)
    # Convert units
    dset[vmax_cf_name] = (
        dset[vmax_cf_name].metpy.convert_units("kph").metpy.dequantify()
    )
    dset[pmsl_cf_name] = (
        dset[pmsl_cf_name].metpy.convert_units("hPa").metpy.dequantify()
    )
    # Define contour levels
    levels_mslp = np.arange(
        dset[pmsl_cf_name].min().astype("int"),
        dset[pmsl_cf_name].max().astype("int"),
        3.0,
    )
    levels_winds_10m = np.arange(1, 258, 2)
    # Define colormaps and normalization
    cmap, norm = utils.get_colormap_norm(
        "winds_wxcharts", levels_winds_10m, extend="max"
    )
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background=True)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_mslp=levels_mslp,
        levels_winds_10m=levels_winds_10m,
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
        vmax_cf_name = utils.find_variable_by_long_name(
            data,
            [
                "Maximum 10 metre wind gust since previous post-processing",
                "maximum Wind 10m",
                "Time-maximum 10 metre wind gust"
            ],
        )
        pmsl_cf_name = utils.find_variable_by_grib_param_id(data, 500002)
        u10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre U wind component", "U-Component of Wind"]
        )
        v10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre V wind component", "V-Component of Wind"]
        )
        data[pmsl_cf_name].values = mpcalc.smooth_n_point(
            data[pmsl_cf_name].values, n=9, passes=10
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
            data[vmax_cf_name].where(data[vmax_cf_name] > 1),
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_winds_10m"],
            # shading='gouraud' # to test pcolormesh
        )
        c = args["ax"].contour(
            args["x"],
            args["y"],
            data[pmsl_cf_name],
            levels=args["levels_mslp"],
            colors="white",
            linewidths=1.0,
        )
        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=6)

        maxlabels = utils.plot_maxmin_points(
            args["ax"],
            args["x"],
            args["y"],
            data[pmsl_cf_name],
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
            data[pmsl_cf_name],
            "min",
            170,
            symbol="L",
            color="coral",
            random=True,
        )
        
        cv = utils.vector_plot(
            ax=args["ax"],
            x=args["x"],
            y=args["y"],
            data=data,
            u_name=u10m_cf_name,
            v_name=v10m_cf_name,
            projection=projection,
        )

        an_fc, an_var, an_run = utils.add_annotations(
            args["ax"],
            data["valid_time"].to_pandas(),
            "10m winds gusts (km/h) and direction",
            run
        )
        an_stats = utils.annotation_stats(
            args["ax"],
            data[vmax_cf_name],
        )

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    ticks=[
                        5,
                        11,
                        19,
                        28,
                        38,
                        49,
                        61,
                        74,
                        88,
                        102,
                        117,
                        133,
                        149,
                        165,
                        183,
                        200,
                        250,
                    ],
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
                an_stats,
                cv,
                maxlabels,
                minlabels,
            ]
        )

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
