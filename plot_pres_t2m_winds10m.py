from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np

import utils
from definitions import (
    chunks_size,
    figsize_x,
    figsize_y,
    logging,
    options_savefig,
    processes,
)

args = utils.parse_arguments()
debug = args.debug
projection = args.projection
variable_name = "t_v_pres"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["U_10M", "V_10M", "T_2M", "PMSL"], projection=projection
    )
    pmsl_cf_name = utils.find_variable_by_grib_param_id(dset, 500002)
    t2m_cf_name = utils.find_variable_by_grib_param_id(dset, 500011)
    dset[t2m_cf_name] = dset[t2m_cf_name].metpy.convert_units("degC").metpy.dequantify()
    dset[pmsl_cf_name] = dset[pmsl_cf_name].metpy.convert_units("hPa").metpy.dequantify()

    levels_t2m = np.arange(-25, 45, 1)
    levels_mslp = np.arange(
        dset[pmsl_cf_name].min().astype("int"), dset[pmsl_cf_name].max().astype("int"), 3.0
    )

    cmap, norm = utils.get_colormap_norm("temp", levels_t2m)
    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    _, x, y = utils.get_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_t2m=levels_t2m,
        levels_mslp=levels_mslp,
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
        pmsl_cf_name = utils.find_variable_by_grib_param_id(data, 500002)
        u10m_cf_name = utils.find_variable_by_grib_param_id(data, 500027)
        v10m_cf_name = utils.find_variable_by_grib_param_id(data, 500029)
        t2m_cf_name = utils.find_variable_by_grib_param_id(data, 500011)
        data[pmsl_cf_name].values = mpcalc.smooth_n_point(data[pmsl_cf_name].values, n=9, passes=10)
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
            data[t2m_cf_name],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_t2m"],
        )
        cs2 = args["ax"].contour(
            args["x"],
            args["y"],
            data[t2m_cf_name],
            extend="both",
            levels=args["levels_t2m"][::5],
            linewidths=0.3,
            colors="gray",
            alpha=0.7,
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
        labels2 = args["ax"].clabel(
            cs2, cs2.levels, inline=True, fmt="%2.0f", fontsize=7
        )

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
        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        density = 15
        width = 0.0015
        headwidth = 3.5
        min_wind_threshold = 2
        max_wind_threshold = 80
        scale = 5
        if projection == "nord":
            density = 10
        wind_magnitude = np.clip(
            np.sqrt(
                data[u10m_cf_name][::density, ::density] ** 2
                + data[v10m_cf_name][::density, ::density] ** 2
            ),
            min_wind_threshold,
            max_wind_threshold,
        )
        u_norm = data[u10m_cf_name][::density, ::density] / wind_magnitude
        v_norm = data[v10m_cf_name][::density, ::density] / wind_magnitude
        x = args["x"][::density, ::density]
        y = args["y"][::density, ::density]

        cv = args["ax"].quiver(
            x,
            y,
            u_norm,
            v_norm,
            scale=scale,
            alpha=0.6,
            color="gray",
            width=width,
            headwidth=headwidth,
            headlength=4.5,
            scale_units="inches",
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "MSLP [hPa], Winds@10m and Temperature@2m",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            cb = plt.colorbar(
                cs,
                orientation="horizontal",
                label="Temperature [C]",
                pad=0.03,
                fraction=0.04,
            )
            cb.minorticks_off()

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [
                cs,
                cs2,
                c,
                labels,
                labels2,
                an_fc,
                an_var,
                an_run,
                cv,
                maxlabels,
                minlabels,
            ]
        )

        first = False


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
