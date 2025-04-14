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
variable_name = "gusts"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["U_10M", "V_10M", "VMAX_10M", "PMSL"], projection=projection
    )

    dset['fg10'] = dset['fg10'].metpy.convert_units('kph').metpy.dequantify()
    dset["pmsl"] = dset["pmsl"].metpy.convert_units("hPa").metpy.dequantify()

    levels_mslp = np.arange(
        dset["pmsl"].min().astype("int"), dset["pmsl"].max().astype("int"), 3.0
    )
    levels_winds_10m = np.arange(0, 255, 1)

    cmap, norm = utils.get_colormap_norm("winds_wxcharts", levels_winds_10m)
    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_mslp=levels_mslp,
        levels_winds_10m=levels_winds_10m
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
        data["pmsl"].values = mpcalc.smooth_n_point(data["pmsl"].values, n=9, passes=10)
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
            data["fg10"],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_winds_10m"],
        )
        c = args["ax"].contour(
            args["x"],
            args["y"],
            data["pmsl"],
            levels=args["levels_mslp"],
            colors="white",
            linewidths=1.0,
        )
        labels = args["ax"].clabel(c, c.levels, inline=True, fmt="%4.0f", fontsize=6)

        maxlabels = utils.plot_maxmin_points(
            args["ax"],
            args["x"],
            args["y"],
            data["pmsl"],
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
            data["pmsl"],
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
                data["u10"][::density, ::density] ** 2
                + data["v10"][::density, ::density] ** 2
            ),
            min_wind_threshold,
            max_wind_threshold,
        )
        u_norm = data["u10"][::density, ::density] / wind_magnitude
        v_norm = data["v10"][::density, ::density] / wind_magnitude
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
            "10m Winds direction and max. wind gust",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            cb = plt.colorbar(
                cs,
                orientation="horizontal",
                label="Wind gust (km/h)",
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
                c,
                labels,
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
