import fsspec
import os
import re
import time
import requests
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.contrib.concurrent import process_map


from definitions import (
    COLORMAPS_DIR,
    SHAPEFILES_DIR,
    REMOTE_FOLDER,
    CACHE_DIR,
    logging,
    figsize_x,
    figsize_y,
)
from projections import proj_defs, subfolder_images


def setup_figure_and_projection(dset, projection, **kwargs):
    """
    Sets up the matplotlib figure, axis, and projection for plotting scripts.
    Returns (m, x, y, ax).
    """
    import matplotlib.pyplot as plt

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    m, x, y = get_projection(dset, projection, **kwargs)

    return m, x, y, ax


def get_latest_model_run(url=REMOTE_FOLDER):
    """
    Fetches the directory listing from the given URL and returns the latest model run folder.
    """
    response = requests.get(url)
    response.raise_for_status()
    # Find all folder names matching the pattern YYYYMMDDHH/
    folder_names = re.findall(r'href="(\d{10}/)"', response.text)
    if not folder_names:
        raise ValueError("No model run folders found at the URL.")
    # Remove trailing slash and sort
    folder_names = [f.rstrip("/") for f in folder_names]
    latest_run = sorted(folder_names)[-1]
    return latest_run


def get_files_sfc(
    vars=["T_2M", "U_10M", "V_10M"],
    run=get_latest_model_run(),
    projection=None,
):
    if not isinstance(vars, list):
        vars = [vars]
    valid_vars = [
        "ALB_RAD",  # Shortwave broadband albedo for diffuse radiation
        "ALHFL_S",  # Latent heat net flux at surface (average since model start)
        "ASHFL_S",  # Sensible heat net flux at surface (average since model start)
        "ASOB_S",  # Net short-wave radiation flux at surface (average since model start
        "ASWDIFD_S",  # Surface down solar diffuse radiation (average since model start)
        "ASWDIR_S",  # Surface down solar direct radiation (average since model start)
        "ATHB_S",  # Net long-wave radiation flux at surface (average since model start)
        "ATHD_S",  #
        "ATHU_S",  #
        "AUMFL_S",  # U-momentum flux at surface ρu0w0
        "AVMFL_S",  # V-momentum flux at surface ρv0w0
        "CAPE_CON",
        "CAPE_ML",
        "CIN_ML",
        "CLCT",
        "FR_LAND",
        "GRAU_GSP",  # Large scale graupel
        "HSURF",
        "HZEROCL",  # Height of 0 degree Celsius isotherm above MSL
        "H_SNOW",  # Snow Depth
        "LPI",  # Lightning Potential Index
        "PMSL",
        "PS",
        "RAIN_CON",
        "RAIN_GSP",
        "SDI_2",  # Supercell Detection Index
        "SNOWLMT",  # Height of snowfall limit above MSL
        "SNOW_CON",
        "SNOW_GSP",
        "TD_2M",
        "TOT_PREC",  # Total precipitation
        "TQV",
        "TWATER",  # Column integrated water (grid scale, including rain)
        "T_2M",  # Column integrated water vapour (grid scale)
        "T_G",  # Ground temperature
        "U_10M",
        "VMAX_10M",
        "V_10M",
        "WW",  # Weather code
        "W_SNOW",  # Snow depth water equivalent (mm)
    ]
    for var in vars:
        if var not in valid_vars:
            raise ValueError(f"Invalid variable {var}. Must be one of {valid_vars}")

    urls = []
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
        url = f"{REMOTE_FOLDER}/{run}/{var}/icon_2I_{run}_{surface_mapping}.grib"
        urls.append(url)

    files = process_map(download_file, urls, chunksize=1, max_workers=4, disable=True)
    logging.info("Loading files into xarray")
    dss = xr.open_mfdataset(
        files, engine="cfgrib", decode_timedelta=True, compat="override"
    )

    if projection is not None and projection in proj_defs:
        proj = proj_defs[projection]
        dss = dss.sel(
            latitude=slice(proj["llcrnrlat"], proj["urcrnrlat"]),
            longitude=slice(proj["llcrnrlon"], proj["urcrnrlon"]),
        )

    return dss.compute()


def get_file_mapping(var, lev_sel=None):
    # Define valid variables and their corresponding levels and mappings
    pressure_vars = ["U", "V", "T", "QV", "OMEGA", "FI", "RELHUM"]
    pressure_levels = [1000, 925, 850, 700, 500, 250]
    soil_vars = ["W_SO"]
    soil_levels = [0, 1, 2, 7]
    soil_vars_t = ["T_SO"]
    soil_levels_t = [0, 1, 2, 5, 15]
    shear_vars = ["WSHEAR_U", "WSHEAR_V"]

    mappings = []

    if var in pressure_vars:
        if lev_sel is not None:
            if lev_sel in pressure_levels:
                pressure_levels = [lev_sel]
            else:
                raise ValueError(f"Selected level is not in {pressure_levels}")
        mappings = [(f"isobaricInhPa-{lev}", lev) for lev in pressure_levels]
    elif var in soil_vars:
        if lev_sel is not None:
            if lev_sel in soil_levels:
                soil_levels = [lev_sel]
            else:
                raise ValueError(f"Selected level is not in {soil_levels}")
        mappings = [(f"depthBelowLandLayer-{lev}", lev) for lev in soil_levels]
    elif var in soil_vars_t:
        if lev_sel is not None:
            if lev_sel in soil_levels_t:
                soil_levels_t = [lev_sel]
            else:
                raise ValueError(f"Selected level is not in {soil_levels_t}")
        mappings = [(f"depthBelowLand-{lev}", lev) for lev in soil_levels_t]
    elif var in shear_vars:
        mappings = [("heightAboveGroundLayer-6000", 6000)]
    elif var == "CLCH":
        mappings = [("isobaricLayer-0", 0)]
    elif var == "CLCL":
        mappings = [("isobaricLayer-800", 800)]
    elif var == "CLCM":
        mappings = [("isobaricLayer-400", 400)]
    elif var == "UH_MAX":
        mappings = [("heightAboveSeaLayer-2000", 2000)]

    return mappings


def get_files_levels(
    vars=["T", "U", "V"], run=get_latest_model_run(), projection=None, lev_sel=None
):
    if not isinstance(vars, list):
        vars = [vars]

    valid_vars = [
        "U",
        "V",
        "T",
        "QV",
        "RELHUM",
        "OMEGA",
        "FI",  # Geopotential height
        "W_SO",
        "T_SO",
        "WSHEAR_U",  # U-component of (vertical) wind shear vector between two levels
        "WSHEAR_V",  # V-component of (vertical) wind shear vector between two levels
        "CLCH",
        "CLCL",
        "CLCM",
        "UH_MAX",  # Maximum amplitude of updraft helicity
    ]

    for var in vars:
        if var not in valid_vars:
            raise ValueError(f"Invalid variable {var}. Must be one of {valid_vars}")

    urls = []
    for var in vars:
        mappings = get_file_mapping(var, lev_sel=lev_sel)
        for mapping, _ in mappings:
            urls.append(f"{REMOTE_FOLDER}/{run}/{var}/icon_2I_{run}_{mapping}.grib")

    files = process_map(download_file, urls, chunksize=1, max_workers=4, disable=True)

    def preprocess(ds):
        attrs = next(iter(ds.data_vars.values())).attrs
        if "GRIB_typeOfLevel" in attrs:
            level_type = attrs["GRIB_typeOfLevel"]
            if level_type not in ds.dims:
                ds = ds.expand_dims(dim=level_type)
        return ds

    logging.info("Loading files into xarray")
    dss = xr.open_mfdataset(
        files,
        engine="cfgrib",
        decode_timedelta=True,
        preprocess=preprocess,
    )

    if projection is not None and projection in proj_defs:
        proj = proj_defs[projection]
        dss = dss.sel(
            latitude=slice(proj["llcrnrlat"], proj["urcrnrlat"]),
            longitude=slice(proj["llcrnrlon"], proj["urcrnrlon"]),
        )

    return dss.compute()


def download_file(url):
    logging.info(f"Fetching file {url}")
    file = fsspec.open_local(
        f"simplecache::{url}", simplecache={"cache_storage": CACHE_DIR}
    )
    return file


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
    # Allow long_name to be a string or a list of strings
    long_names_found = []
    if not isinstance(long_name, list):
        long_name = [long_name]
    for name in long_name:
        logging.debug(f"Looking for variable name {name}")
        for var_name, var_data in dataset.data_vars.items():
            var_long_name = var_data.attrs.get("long_name")
            long_names_found.append(var_long_name)
            if var_long_name == name:
                return var_name
    raise ValueError(f"No variable with long_name in {long_name} found in the dataset. The following long names were found in the dataset: {long_names_found}")


def get_projection(
    dset=None,
    projection="it",
    countries=True,
    regions=True,
    labels=False,
    cities=False,
    color_borders="black",
    background=False,
):
    from mpl_toolkits.basemap import Basemap

    proj_options = proj_defs[projection]
    m = Basemap(**proj_options)
    if background:
        m.arcgisimage(service="World_Shaded_Relief", xpixels=1500)

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


def annotation(ax, text, loc="upper right", fontsize=7):
    """Put a general annotation in the plot."""
    at = AnchoredText("%s" % text, prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)

    return at


def add_annotations(ax, time, title, run):
    an_fc = annotation_forecast(ax, time)
    an_var = annotation(
        ax,
        title,
        loc="lower left",
    )
    an_run = annotation_run(ax, run)

    return an_fc, an_var, an_run


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
    font_border_color="black",
    font_border_width=1,
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
                        path_effects.withStroke(
                            linewidth=font_border_width, foreground=font_border_color
                        )
                    ],
                    zorder=10,
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
    output_dir = subfolder_images.get(projection, "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")
    filename = f"{output_dir}/{variable_name}_{forecast_hour:03d}.png"
    return filename


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


def add_colorbar(ax, c, size="2%", pad=0.1, position="bottom", cbar_kwargs={}):
    """Add colorbar to the bottom"""
    ax_divider = make_axes_locatable(ax)
    # Create an axis on the bottom of the main plot axis to host the colorbar
    ax_colorbar = ax_divider.append_axes(position, size=size, pad=pad)
    # and draw the colorbar inside. We grab the figure so that we
    # don't have to pass it in the function
    orientation = "vertical"
    if position == "bottom":
        orientation = "horizontal"
    colorbar = plt.gcf().colorbar(
        c, cax=ax_colorbar, orientation=orientation, drawedges=True, **cbar_kwargs
    )
    colorbar.minorticks_off()
    colorbar.dividers.set_color("white")
    colorbar.dividers.set_linewidth(0.4)
    colorbar.dividers.set_alpha(0.5)
    colorbar.outline.set_color("gray")
    colorbar.outline.set_alpha(0.5)
    colorbar.ax.tick_params(labelsize=7)

    return colorbar


def vector_plot(
    ax,
    data,
    u_name,
    v_name,
    projection,
    x,
    y,
    density=15,
    width=0.0015,
    headwidth=3.5,
    min_wind_threshold=2,
    max_wind_threshold=80,
    scale=5,
):
    # We need to reduce the number of points before plotting the vectors,
    # these values work pretty well
    if projection == "nord":
        density = 10
    wind_magnitude = np.clip(
        np.sqrt(
            data[u_name][::density, ::density] ** 2
            + data[v_name][::density, ::density] ** 2
        ),
        min_wind_threshold,
        max_wind_threshold,
    )
    u_norm = data[u_name][::density, ::density] / wind_magnitude
    v_norm = data[v_name][::density, ::density] / wind_magnitude
    x_sub = x[::density, ::density]
    y_sub = y[::density, ::density]

    cv = ax.quiver(
        x_sub,
        y_sub,
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

    return cv


def run_main_with_timing(main_func):
    start_time = time.time()
    main_func()
    elapsed_time = time.time() - start_time
    logging.info("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
