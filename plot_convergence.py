from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np

import utils
from args import debug, projection, run, timesteps
from definitions import (
    chunks_size,
    logging,
    options_savefig,
    processes,
)

variable_name = "convergence"

if not debug:
    import matplotlib

    matplotlib.use("Agg")


def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(
        vars=["U_10M", "V_10M"], projection=projection, run=run, timesteps=timesteps
    )
    # Define contour levels for convergence (symmetric around zero)
    # Using coarser levels to avoid showing too much small-scale noise
    levels_convergence = np.arange(-10, 11, 1)

    # Define colormaps and normalization
    cmap, norm = utils.get_colormap_norm(
        "vorticity_mlgx", levels_convergence, extend="both"
    )

    m, x, y, ax = utils.setup_figure_and_projection(dset, projection, background="World_Shaded_Relief")

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_convergence=levels_convergence,
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
        u10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre U wind component", "U-Component of Wind"]
        )
        v10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre V wind component", "V-Component of Wind"]
        )

        # Smooth wind fields before computing convergence to reduce noise
        # Convergence involves spatial derivatives which amplify grid-scale noise
        u_smoothed = data[u10m_cf_name].values.copy()
        v_smoothed = data[v10m_cf_name].values.copy()
        u_smoothed = mpcalc.smooth_n_point(u_smoothed, n=9, passes=3)
        v_smoothed = mpcalc.smooth_n_point(v_smoothed, n=9, passes=3)

        # Assign smoothed values back with metpy units
        data[u10m_cf_name].values = u_smoothed
        data[v10m_cf_name].values = v_smoothed

        # Compute convergence (negative divergence)
        # Drop 'time' coordinate to avoid metpy warning (we only need valid_time)
        u = data[u10m_cf_name].drop_vars('time').metpy.quantify()
        v = data[v10m_cf_name].drop_vars('time').metpy.quantify()
        convergence = - mpcalc.divergence(u, v)

        # Convert to 10^-4 s^-1 for better visualization
        convergence_scaled = (convergence * 1e4).metpy.dequantify()

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
            convergence_scaled,
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_convergence"],
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
            "10m wind convergence (10⁻4 s⁻¹) and direction",
            run
        )
        an_stats = utils.annotation_stats(
            args["ax"],
            convergence_scaled,
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
                an_stats,
                cv,
            ]
        )

        first = False


if __name__ == "__main__":
    utils.run_main_with_timing(main)
