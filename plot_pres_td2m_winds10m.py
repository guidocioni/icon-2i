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

variable_name ="td_v_pres"

if not debug:
    import matplotlib
    matplotlib.use("Agg")

def main():
    logging.info(
        f"Plotting {variable_name} for projection {projection}."
    )
    dset = utils.get_files_sfc(
        vars=["U_10M", "V_10M", "TD_2M", "PMSL"], projection=projection, run=run
    )
    pmsl_cf_name = utils.find_variable_by_grib_param_id(dset, 500002)
    td2m_cf_name = utils.find_variable_by_long_name(dset, ["2 metre dewpoint temperature", "2m Dew Point Temperature"])
    dset[td2m_cf_name] = dset[td2m_cf_name].metpy.convert_units("degC").metpy.dequantify()
    dset[pmsl_cf_name] = dset[pmsl_cf_name].metpy.convert_units("hPa").metpy.dequantify()

    levels_temp = np.arange(-25, 30, 1)
    levels_mslp = np.arange(
        dset[pmsl_cf_name].min().astype("int"), dset[pmsl_cf_name].max().astype("int"), 3.0
    )

    cmap, norm = utils.get_colormap_norm("temp_mlgx", levels_temp, extend='both')
    m, x, y, ax = utils.setup_figure_and_projection(dset, projection)

    # All the arguments that need to be passed to the plotting function
    args = dict(
        x=x,
        y=y,
        ax=ax,
        cmap=cmap,
        norm=norm,
        levels_temp=levels_temp,
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
        u10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre U wind component", "U-Component of Wind"]
        )
        v10m_cf_name = utils.find_variable_by_long_name(
            data, ["10 metre V wind component", "V-Component of Wind"]
        )
        td2m_cf_name = utils.find_variable_by_long_name(data, ["2 metre dewpoint temperature", "2m Dew Point Temperature"])
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
            data[td2m_cf_name],
            extend="both",
            cmap=args["cmap"],
            norm=args["norm"],
            levels=args["levels_temp"],
        )
        cs2 = args["ax"].contour(
            args["x"],
            args["y"],
            data[td2m_cf_name],
            levels=args["levels_temp"][::5],
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
            "MSLP (hPa), 10m Winds and 2m Dewpoint (C)",
            loc="lower left",
        )
        an_run = utils.annotation_run(args["ax"], run)

        if first:
            utils.add_colorbar(
                ax=args["ax"],
                c=cs,
                cbar_kwargs=dict(
                    label="Dewpoint Temperature (C)",
                ),
            )

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
    utils.run_main_with_timing(main)
