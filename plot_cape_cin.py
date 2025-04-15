from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
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
variable_name = "cape_cin"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["CAPE_ML", "CIN_ML", "U_10M", "V_10M"], projection=projection
    )
    cape_cf_name = utils.find_variable_by_grib_param_id(dset, 500153)
    dset[cape_cf_name] = dset[cape_cf_name].where(dset[cape_cf_name] >= 100)

    levels_cape = np.concatenate(
        [np.arange(0.0, 3000.0, 100.0), np.arange(3000.0, 7000.0, 200.0)]
    )
    cmap, norm = utils.get_colormap_norm("cape_wxcharts", levels=levels_cape)

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)
    m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, cmap=cmap, norm=norm, levels_cape=levels_cape)

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
        u10m_cf_name = utils.find_variable_by_grib_param_id(data, 500027)
        v10m_cf_name = utils.find_variable_by_grib_param_id(data, 500029)
        cape_cf_name = utils.find_variable_by_grib_param_id(data, 500153)
        cin_cf_name = utils.find_variable_by_grib_param_id(data, 500154)
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
            data[cape_cf_name],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_cape"],
        )
        cr = args["ax"].contourf(
            args["x"],
            args["y"],
            data[cin_cf_name],
            colors="none",
            levels=(50, 100.0),
            hatches=["...", "..."],
            zorder=5,
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
            "CAPE and Winds@10 m, hatches CIN$<-50$ J/kg",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            cb = plt.colorbar(
                cs,
                orientation="horizontal",
                label="CAPE [J/kg]",
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
                cr,
                an_fc,
                an_var,
                an_run,
                cv,
            ]
        )

        first = False


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
