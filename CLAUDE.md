# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python scripts for plotting ICON-2I weather model output data. The model data is fetched from https://meteohub.agenziaitaliameteo.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS and cached locally using `fsspec` during plot generation.

## Running Plot Scripts

### Individual Scripts

Each `plot_*.py` script generates a specific type of meteorological visualization. Run any script with:

```bash
python plot_<name>.py [OPTIONS]
```

**Common Options:**
- `--projection <name>` - Map projection to use (default: `it`). See [projections.py](projections.py) for all available projections (e.g., `it`, `nord`, `sud`, `centro`, `toscana`, `lombardia`, etc.)
- `--debug` - Show plots interactively instead of saving to PNG files
- `--run <YYYYMMDDHH>` - Specific forecast run to fetch (default: latest available)
- `--level <number>` - Pressure level in hPa (default: 850, only used for upper-level plots)

**Examples:**
```bash
# Interactive debug mode for Lombardy
python plot_gusts.py --projection lombardia --debug

# Generate plots for a specific model run
python plot_cape_cin.py --run 2026081912 --projection nord
```

### Batch Plotting

Use [plots.sh](plots.sh) to generate multiple plots across multiple projections in parallel:

1. Create `plots.conf` from the template (if it doesn't exist):
   ```bash
   cp plots.conf.default plots.conf
   ```

2. Edit `plots.conf` to specify which scripts and projections to run

3. Execute:
   ```bash
   ./plots.sh
   ```

## Code Architecture

### Core Module Structure

- **[definitions.py](definitions.py)** - Global configuration (paths, figure size, remote data URL, logging setup)
- **[args.py](args.py)** - Command-line argument parsing used by all plot scripts
- **[utils.py](utils.py)** - Shared utilities for data fetching, plotting, colormap handling, map setup, and annotations
- **[projections.py](projections.py)** - Map projection definitions with geographic boundaries and EPSG codes

### Plot Script Pattern

All `plot_*.py` scripts follow a consistent structure:

1. **Imports and setup** - Import required modules, get args from `args.py`, set matplotlib backend to `Agg` when not in debug mode
2. **`main()` function**:
   - Fetch required variables using `utils.get_files_sfc()` or `utils.get_files_pl()`
   - Find variables by CF name or GRIB parameter ID using utility functions
   - Convert units using MetPy
   - Define contour levels and colormaps
   - Set up figure and projection with `utils.setup_figure_and_projection()`
   - Call plotting function, either directly (debug) or via multiprocessing Pool
3. **`plot_files()` function**:
   - Iterates over forecast timesteps
   - Creates matplotlib contourf/contour/vector plots
   - Adds annotations (valid time, variable name, run time, statistics)
   - Saves to PNG or displays interactively
   - Removes plot collections before next iteration (memory optimization)

### Data Fetching

The `utils.get_files_sfc()` and `utils.get_files_pl()` functions:
- Construct remote GRIB file URLs based on run time and variables
- Use `fsspec` with `simplecache` to cache downloaded files in `/tmp/` (or `CACHE_DIR` env var)
- Open with `xarray` and `cfgrib` engine
- Subset spatially based on projection bounds for efficiency

### Variable Discovery

Since GRIB files can have inconsistent metadata, use helper functions:
- `utils.find_variable_by_long_name(dset, ["name1", "name2"])` - Search by CF long_name attribute
- `utils.find_variable_by_grib_param_id(dset, param_id)` - Search by GRIB parameter ID
- These return the actual variable name in the dataset

### Colormaps

Custom colormaps are stored in [colormaps/](colormaps/) as text files with `R, G, B, A` values (0-255 range).

The function `utils.get_colormap_norm(cmap_name, levels, extend=None)` returns both a matplotlib colormap and norm object. The number of colors should match the number of contour levels for proper mapping.

### Parallelization

Plot scripts use Python's `multiprocessing.Pool` to parallelize across forecast timesteps:
- Dataset is chunked using `utils.chunks_dataset()` (chunk size from `definitions.chunks_size`)
- Number of worker processes controlled by `definitions.processes`
- Each chunk processed independently to generate PNG files

### Output

- PNG files saved to `images/<projection>/<variable_name>_<forecast_hour>.png`
- Output directory can be changed via `IMAGES_DIR` environment variable
- Savefig options (DPI, bbox, transparency) configured in `definitions.options_savefig`

## Environment Variables

- `IMAGES_DIR` - Override default output directory (default: `./images`)
- `CACHE_DIR` - Override fsspec cache location (default: `/tmp/`)

## Development Notes

- Forecast data is fetched on-demand and cached; no need to pre-download
- When adding a new plot type, follow the existing pattern in `plot_gusts.py` or similar scripts
- Use MetPy for unit conversions and meteorological calculations
- Use `mpcalc.smooth_n_point()` for field smoothing (especially MSLP)
- All map projections use Basemap (deprecated) with Mercator projection and varying geographic bounds
- Logos are added via `utils.add_logos_on_ax()` from the [logos/](logos/) directory
