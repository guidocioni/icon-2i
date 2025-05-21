from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

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

variable_name = "cape_cin"

if not debug:
    import matplotlib
    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(
        vars=["CAPE_ML", "CIN_ML", "U_10M", "V_10M"], projection=projection, run=run
    )
    cape_cf_name = utils.find_variable_by_grib_param_id(dset, 500153)
    dset[cape_cf_name] = dset[cape_cf_name].where(dset[cape_cf_name] >= 100)

    levels_cape = np.concatenate(
        [np.arange(0.0, 3000.0, 100.0), np.arange(3000.0, 7200.0, 200.0)]
    )
    cmap, norm = utils.get_colormap_norm("cape_wxcharts", levels=levels_cape, extend='max')

    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background=True)

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
        u10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre U wind component", "U-Component of Wind"]
        )
        v10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre V wind component", "V-Component of Wind"]
        )
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
        cv = utils.vector_plot(
            ax=args["ax"],
            x=args["x"],
            y=args["y"],
            data=data,
            u_name=u10m_cf_name,
            v_name=v10m_cf_name,
            projection=projection,
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "CAPE (J/Kg), 10m Winds, hatches CIN$<-50$ J/kg",
            loc="lower left",
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    ticks=[100, 500, 1000, 1500, 2000, 2500,
                           3000, 4000, 5000, 6000, 7000],
                ),
            )

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
