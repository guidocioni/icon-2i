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
run = args.run
variable_name = "sdi"
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(
        vars=["U_10M", "V_10M", "SDI_2"], projection=projection, run=run
    )
    sdi_cf_name = utils.find_variable_by_long_name(
        dset,
        [
            "supercell detection index 2 (only rot. up drafts)",
        ],
    )
    dset[sdi_cf_name] *= 1000.
    # Define contour levels
    levels_sdi = [-4, -3.5, -3, -1.5, -0.5, -0.25, 0, 0.25, 0.5, 1.5, 3, 3.5, 4]
    # Define colormaps and normalization
    cmap, norm = utils.get_colormap_norm(
        "temp_anom", levels_sdi, extend="both"
    )
    # Initialize background figure
    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)
    m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_sdi=levels_sdi,
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
        sdi_cf_name = utils.find_variable_by_long_name(
            data, "supercell detection index 2 (only rot. up drafts)",
        )
        u10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre U wind component", "U-Component of Wind"]
        )
        v10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre V wind component", "V-Component of Wind"]
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
            data[sdi_cf_name],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_sdi"],
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
            alpha=0.8,
            color="gray",
            width=width,
            headwidth=headwidth,
            headlength=4.5,
            scale_units="inches",
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "10m winds gusts (km/h) and direction",
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
            [
                cs,
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
