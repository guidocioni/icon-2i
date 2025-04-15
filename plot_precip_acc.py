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
variable_name = 'precip_acc'
output_dir = utils.set_output_dir(projection)

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}. Writing images in {output_dir}"
    )
    dset = utils.get_files_sfc(vars=["TOT_PREC"], projection=projection)

    levels_precip = np.concatenate(
        [
            np.arange(1, 51, 1),
            np.arange(52, 102, 2),
            np.arange(110, 210, 10),
            np.arange(230, 530, 30),
            np.arange(550, 1050, 50),
            np.arange(1100, 2100, 100),
        ]
    )
    cmap, norm = utils.get_colormap_norm("prec_acc_wxcharts", levels=levels_precip, extend='max')

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    m, x, y = utils.get_projection(dset, projection)
    m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_precip=levels_precip, cmap=cmap, norm=norm)

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
        tp_cf_name = utils.find_variable_by_grib_param_id(data, 500041)
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
            data[tp_cf_name],
            extend="max",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_precip"],
        )
    
        density = 17
        if projection == 'nord':
            density = 10
        elif projection == 'sud':
            density = 9
        elif projection == 'centro':
            density = 7
        vals = utils.add_vals_on_map(
            ax=args["ax"],
            var=data[tp_cf_name].where(data[tp_cf_name] > 50),
            x=args["x"],
            y=args["y"],
            cmap=args['cmap'],
            norm=args['norm'],
            density=density,
            fontsize=6
        )

        an_fc = utils.annotation_forecast(args["ax"], data["valid_time"].to_pandas())
        an_var = utils.annotation(
            args["ax"],
            "Accumulated precipitation",
            loc="lower left",
            fontsize=6,
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            cb = plt.colorbar(
                cs,
                orientation="horizontal",
                label="Accumulated precipitation [mm]",
                pad=0.035,
                fraction=0.035,
                ticks=[1, 5, 10, 15, 25, 35, 50, 100, 200, 500, 1000, 2000]
            )
            cb.minorticks_off()

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        utils.remove_collections(
            [cs, an_fc, an_var, an_run, vals]
        )

        first = False


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
