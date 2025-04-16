import argparse
import fsspec
import os
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt

from definitions import (
    COLORMAPS_DIR,
    SHAPEFILES_DIR,
    logging,
)
from projections import proj_defs, subfolder_images


def get_files_sfc(
    vars=["T_2M", "U_10M", "V_10M"],
    run=pd.to_datetime("now").strftime("%Y%m%d00"),
    projection=None,
):
    if not isinstance(vars, list):
        vars = [vars]
    valid_vars = [
        "ALB_RAD",
        "ALHFL_S",
        "ASHFL_S",
        "ASOB_S",
        "ASWDIFD_S",
        "ASWDIR_S",
        "ATHB_S",
        "ATHD_S",
        "ATHU_S",
        "AUMFL_S",
        "AVMFL_S",
        "CAPE_CON",
        "CAPE_ML",
        "CIN_ML",
        "CLCT",
        "FR_LAND",
        "GRAU_GSP",
        "HSURF",
        "HZEROCL",
        "H_SNOW",
        "LPI",
        "PMSL",
        "PS",
        "RAIN_CON",
        "RAIN_GSP",
        "SDI_2",
        "SNOWLMT",
        "SNOW_CON",
        "SNOW_GSP",
        "TD_2M",
        "TOT_PREC",
        "TQV",
        "TWATER",
        "T_2M",
        "T_G",
        "U_10M",
        "VMAX_10M",
        "V_10M",
        "WW",
        "W_SNOW",
    ]
    for var in vars:
        if var not in valid_vars:
            raise ValueError(f"Invalid variable {var}. Must be one of {valid_vars}")
    dss = []
    for var in vars:
        surface_mapping = "surface-0"
        if "_2M" in var:
            surface_mapping = "heightAboveGround-2"
        elif "_10M" in var:
            surface_mapping = "heightAboveGround-10"
        elif var in ["SNOWLMT", "HZEROCL"]:
            surface_mapping = "isothermZero-0"
        elif var == "PMSL":
            surface_mapping = "meanSea-0"
        elif var in ["CAPE_ML", "CIN_ML"]:
            surface_mapping = "atmML-0"
        url = f"https://meteohub.mistralportal.it/nwp/ICON-2I_all2km/{run}/{var}/icon_2I_{run}_{surface_mapping}.grib"
        file = fsspec.open_local(
            f"simplecache::{url}", simplecache={"cache_storage": "/tmp/"}
        )
        dss.append(xr.open_dataset(file, engine="cfgrib", decode_timedelta=True))
    dss = xr.merge(dss, compat="override")

    if projection is not None and projection in proj_defs:
        proj = proj_defs[projection]
        dss = dss.sel(
            latitude=slice(proj["llcrnrlat"], proj["urcrnrlat"]),
            longitude=slice(proj["llcrnrlon"], proj["urcrnrlon"]),
        )

    return dss


def get_file_mapping(var):
    # Define valid variables and their corresponding levels and mappings
    pressure_vars = ["U", "V", "T", "QV", "OMEGA", "FI"]
    pressure_levels = [1000, 925, 850, 700, 500, 250]
    soil_vars = ["W_SO", "T_SO"]
    soil_levels = [0, 1, 2, 7]
    shear_vars = ["WSHEAR_U", "WSHEAR_V"]

    mappings = []

    if var in pressure_vars:
        mappings = [(f"isobaricInhPa-{lev}", lev) for lev in pressure_levels]
    elif var in soil_vars:
        mappings = [(f"depthBelowLandLayer-{lev}", lev) for lev in soil_levels]
    elif var in shear_vars:
        mappings = [("heightAboveGroundLayer-6000", 6000)]
    elif var == "CLCH":
        mappings = [("isobaricLayer-0", 0)]
    elif var == "CLCL":
        mappings = [("isobaricLayer-800", 800)]
    elif var == "CLCM":
        mappings = [("isobaricLayer-400", 400)]

    return mappings


def get_files_levels(
    vars=["T", "U", "V"],
    run=pd.to_datetime("now").strftime("%Y%m%d00"),
    projection=None,
):
    if not isinstance(vars, list):
        vars = [vars]

    valid_vars = [
        "U",
        "V",
        "T",
        "QV",
        "OMEGA",
        "FI",
        "W_SO",
        "T_SO",
        "WSHEAR_U",
        "WSHEAR_V",
        "CLCH",
        "CLCL",
        "CLCM",
    ]

    for var in vars:
        if var not in valid_vars:
            raise ValueError(f"Invalid variable {var}. Must be one of {valid_vars}")

    dss = []
    for var in vars:
        mappings = get_file_mapping(var)
        for mapping, _ in mappings:
            url = f"https://meteohub.mistralportal.it/nwp/ICON-2I_all2km/{run}/{var}/icon_2I_{run}_{mapping}.grib"
            file = fsspec.open_local(
                f"simplecache::{url}", simplecache={"cache_storage": "/tmp/"}
            )
            ds = xr.open_dataset(file, engine="cfgrib")
            attrs = next(iter(ds.data_vars.values())).attrs
            if "GRIB_typeOfLevel" in attrs:
                level_type = attrs["GRIB_typeOfLevel"]
                ds = ds.expand_dims(dim=level_type)
            dss.append(ds)

    dss = xr.merge(dss, compat="override")

    if projection is not None and projection in proj_defs:
        proj = proj_defs[projection]
        dss = dss.sel(
            latitude=slice(proj["llcrnrlat"], proj["urcrnrlat"]),
            longitude=slice(proj["llcrnrlon"], proj["urcrnrlon"]),
        )

    return dss


def get_coordinates(ds):
    """Get the lat/lon coordinates from the ds and convert them to degrees.
    Usually this is only used to prepare the plotting."""
    if ("lat" in ds.coords.keys()) and ("lon" in ds.coords.keys()):
        longitude = ds["lon"]
        latitude = ds["lat"]
    elif ("latitude" in ds.coords.keys()) and ("longitude" in ds.coords.keys()):
        longitude = ds["longitude"]
        latitude = ds["latitude"]
    elif ("lat2d" in ds.coords.keys()) and ("lon2d" in ds.coords.keys()):
        longitude = ds["lon2d"]
        latitude = ds["lat2d"]

    if longitude.max() > 180:
        longitude = ((longitude.lon + 180) % 360) - 180

    return np.meshgrid(longitude.values, latitude.values)


def find_variable_by_grib_param_id(dataset, param_id):
    """
    Find the variable name in an xarray.Dataset based on the GRIB_paramId attribute.

    Parameters:
        dataset (xarray.Dataset): The dataset to search.
        param_id (int): The GRIB_paramId to look for.

    Returns:
        str: The variable name corresponding to the given GRIB_paramId.

    Raises:
        ValueError: If no variable with the specified GRIB_paramId is found.
    """
    for var_name, var_data in dataset.data_vars.items():
        if var_data.attrs.get("GRIB_paramId") == param_id:
            return var_name
    raise ValueError(f"No variable with GRIB_paramId {param_id} found in the dataset.")


def find_variable_by_long_name(dataset, long_name):
    for var_name, var_data in dataset.data_vars.items():
        if var_data.attrs.get("long_name") == long_name:
            return var_name
    raise ValueError(f"No variable with long_name {long_name} found in the dataset.")


def get_projection(
    dset=None,
    projection="it",
    countries=True,
    regions=True,
    labels=False,
    cities=False,
    color_borders="black",
):
    from mpl_toolkits.basemap import Basemap

    proj_options = proj_defs[projection]
    m = Basemap(**proj_options)
    if regions:
        m.readshapefile(
            f"{SHAPEFILES_DIR}/ITA_adm/ITA_adm1",
            "ITA_adm1",
            linewidth=0.2,
            color="black",
            zorder=7,
        )
    if labels:
        m.drawparallels(
            np.arange(-80.0, 81.0, 2),
            linewidth=0.2,
            color="white",
            labels=[True, False, False, True],
            fontsize=7,
        )
        m.drawmeridians(
            np.arange(-180.0, 181.0, 2),
            linewidth=0.2,
            color="white",
            labels=[True, False, False, True],
            fontsize=7,
        )

    if cities:
        plot_cities(m)

    m.drawcoastlines(linewidth=0.5, linestyle="solid", color=color_borders, zorder=7)
    if countries:
        m.drawcountries(linewidth=0.5, linestyle="solid", color=color_borders, zorder=7)

    x, y = None, None
    if dset is not None:
        lon2d, lat2d = get_coordinates(dset)
        x, y = m(lon2d, lat2d)

    return m, x, y


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def chunks_dataset(ds, n):
    """Same as 'chunks' but for the time dimension in
    a dataset"""
    for i in range(0, len(ds.step), n):
        yield ds.isel(step=slice(i, i + n))


# Annotation run, model
def annotation_run(ax, time, loc="upper right", fontsize=8):
    """Put annotation of the run obtaining it from the
    time array passed to the function."""
    time = pd.to_datetime(time)
    at = AnchoredText(
        "ICON-2I Run %s" % time.strftime("%Y%m%d %H UTC"),
        prop=dict(size=fontsize),
        frameon=True,
        loc=loc,
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return at


def annotation_forecast(ax, time, loc="upper left", fontsize=8, local=True):
    """Put annotation of the forecast time."""
    time = pd.to_datetime(time)
    if local:  # convert to local time
        time = convert_timezone(time)
        at = AnchoredText(
            "Valid %s" % time.strftime("%A %d %b %Y at %H:%M (Berlin)"),
            prop=dict(size=fontsize),
            frameon=True,
            loc=loc,
        )
    else:
        at = AnchoredText(
            "Forecast for %s" % time.strftime("%A %d %b %Y at %H:%M UTC"),
            prop=dict(size=fontsize),
            frameon=True,
            loc=loc,
        )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return at


def convert_timezone(dt_from, from_tz="utc", to_tz="Europe/Berlin"):
    """Convert between two timezones. dt_from needs to be a Timestamp
    object, don't know if it works otherwise."""
    dt_to = dt_from.tz_localize(from_tz).tz_convert(to_tz)
    # remove again the timezone information

    return dt_to.tz_localize(None)


def annotation(ax, text, loc="upper right", fontsize=8):
    """Put a general annotation in the plot."""
    at = AnchoredText("%s" % text, prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)

    return at


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Truncate a colormap by specifying the start and endpoint."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )

    return new_cmap


def get_colormap_norm(cmap_type, levels, extend="both", clip=False):
    colors_tuple = pd.read_csv(f"{COLORMAPS_DIR}/cmap_{cmap_type}.rgba").values
    cmap = colors.LinearSegmentedColormap.from_list("", colors_tuple, len(levels) + 1)
    # Adjust ncolors based on the extend parameter
    extra_bins = 2 if extend == "both" else 1 if extend in ["min", "max"] else 0
    ncolors = len(levels) - 1 + extra_bins
    norm = colors.BoundaryNorm(
        boundaries=levels, ncolors=ncolors, clip=clip, extend=extend
    )

    return cmap, norm


def remove_collections(elements):
    """Remove the collections of an artist to clear the plot without
    touching the background, which can then be used afterwards."""
    for element in elements:
        try:
            for coll in element.collections:
                coll.remove()
        except AttributeError:
            try:
                for coll in element:
                    coll.remove()
            except ValueError:
                logging.warning("Element is empty")
            except TypeError:
                element.remove()
        except ValueError:
            logging.warning("Collection is empty")


def plot_maxmin_points(
    ax, lon, lat, data, extrema, nsize, symbol, color="k", random=False
):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    # We have to first add some random noise to the field, otherwise it will find many maxima
    # close to each other. This is not the best solution, though...
    if random:
        data = np.random.normal(data, 0.2)

    if extrema == "max":
        data_ext = maximum_filter(data, nsize, mode="nearest")
    elif extrema == "min":
        data_ext = minimum_filter(data, nsize, mode="nearest")
    else:
        raise ValueError("Value for hilo must be either max or min")

    mxy, mxx = np.where(data_ext == data)
    # Filter out points on the border
    mxx, mxy = mxx[(mxy != 0) & (mxx != 0)], mxy[(mxy != 0) & (mxx != 0)]

    texts = []
    for i in range(len(mxy)):
        texts.append(
            ax.text(
                lon[mxy[i], mxx[i]],
                lat[mxy[i], mxx[i]],
                symbol,
                color=color,
                size=15,
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
                zorder=8,
            )
        )
        texts.append(
            ax.text(
                lon[mxy[i], mxx[i]],
                lat[mxy[i], mxx[i]],
                "\n" + str(data[mxy[i], mxx[i]].astype("int")),
                color="gray",
                size=10,
                clip_on=True,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="top",
                zorder=8,
            )
        )
    return texts


def add_vals_on_map(
    ax,
    x,
    y,
    var,
    levels=None,
    density=50,
    cmap="rainbow",
    norm=None,
    shift_x=0.0,
    shift_y=0.0,
    fontsize=7.5,
    lcolors=True,
    font_border_color="gray",
    font_border_width=1
):
    """Given an input projection, a variable containing the values and a plot put
    the values on a map exlcuing NaNs and taking care of not going
    outside of the map boundaries, which can happen.
    - shift_x and shift_y apply a shifting offset to all text labels
    - colors indicate whether the colorscale cmap should be used to map the values of the array"""

    if norm is None:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))

    m = mplcm.ScalarMappable(norm=norm, cmap=cmap)

    # Use isel to subsample
    subsampled = (
        var.isel(
            latitude=slice(1, var.sizes["latitude"] - 1, density),
            longitude=slice(1, var.sizes["longitude"] - 1, density),
        )
        .dropna("latitude", how="all")
        .dropna("longitude", how="all")
    )

    at = []
    for i_lat, lat in enumerate(subsampled["latitude"]):
        for i_lon, lon in enumerate(subsampled["longitude"]):
            val = subsampled.sel(latitude=lat, longitude=lon).item()

            # Get the corresponding indices in the full arrays
            full_i_lat = int(var.get_index("latitude").get_loc(lat.item()))
            full_i_lon = int(var.get_index("longitude").get_loc(lon.item()))

            # Use these indices to get x and y
            coord_x = x[full_i_lat, full_i_lon]
            coord_y = y[full_i_lat, full_i_lon]

            # Skip if the value is NaN
            if np.isnan(val):
                continue

            at.append(
                ax.annotate(
                    f"{int(val)}",
                    (coord_x + shift_x, coord_y + shift_y),
                    color=m.to_rgba(float(val)) if lcolors else "white",
                    weight="bold",
                    fontsize=fontsize,
                    path_effects=[
                        path_effects.withStroke(linewidth=font_border_width, foreground=font_border_color)
                    ],
                    zorder=6,
                )
            )

    return at


def divide_axis_for_cbar(ax, width="45%", height="2%", pad=-2, adjust=0.05):
    """Using inset_axes, divides axis in two to place the colorbars side to side.
    Note that we use the bbox explicitlly with padding to adjust the position of the colorbars
    otherwise they'll come out of the axis (don't really know why)"""
    ax_cbar = inset_axes(
        ax,
        width=width,
        height=height,
        loc="lower left",
        borderpad=pad,
        bbox_to_anchor=(adjust, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
    )
    ax_cbar_2 = inset_axes(
        ax,
        width=width,
        height=height,
        loc="lower right",
        borderpad=pad,
        bbox_to_anchor=(-adjust, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
    )

    return ax_cbar, ax_cbar_2


def find_image_filename(projection, variable_name, forecast_hour):
    filename = f"{subfolder_images.get(projection, '')}/{variable_name}_{forecast_hour:03d}.png"
    return filename


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug mode"
    )
    parser.add_argument(
        "--projection", type=str, default="nord", help="Map projection to use"
    )
    return parser.parse_args()


def set_output_dir(projection):
    output_dir = subfolder_images.get(projection, "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")
    return output_dir


def plot_cities(
    m,
    shapefile_path=f"{SHAPEFILES_DIR}/ne_10m_populated_places_simple/ne_10m_populated_places_simple",
):
    """
    Plots cities on a Basemap using a shapefile of populated places.

    Parameters:
    - ax: matplotlib axis object
    - m: Basemap object
    - shapefile_path: path to the shapefile without file extension (e.g., 'ne_10m_populated_places_simple')
    """
    # Read the shapefile
    m.readshapefile(
        shapefile=shapefile_path, name="ne_10m_populated_places", drawbounds=False
    )

    shapes = []
    texts = []
    for info, shape in zip(m.ne_10m_populated_places_info, m.ne_10m_populated_places):
        if (
            (info["longitude"] <= m.urcrnrlon - 0.25)
            & (info["longitude"] >= m.llcrnrlon + 0.25)
            & (info["latitude"] <= m.urcrnrlat - 0.25)
            & (info["latitude"] >= m.llcrnrlat + 0.25)
        ):
            shapes.append(
                plt.plot(
                    shape[0],
                    shape[1],
                    "o",
                    color="brown",
                    zorder=10,
                    markersize=3,
                    alpha=0.8,
                )
            )
            texts.append(
                plt.annotate(
                    info["name"],
                    xy=shape,
                    zorder=10,
                    fontsize=6,
                    xytext=(-5, 5),
                    textcoords="offset points",
                    weight="bold",
                    path_effects=[
                        path_effects.withStroke(linewidth=2, foreground="white")
                    ],
                )
            )
