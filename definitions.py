import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))
IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.join(ROOT_DIR, "images"))
COLORMAPS_DIR = os.path.join(ROOT_DIR, "colormaps")
SHAPEFILES_DIR = os.path.join(ROOT_DIR, "shapefiles")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/")

REMOTE_FOLDER = "https://meteohub.mistralportal.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS"

# Options for savefig
options_savefig = {"dpi": 110, "bbox_inches": "tight", "transparent": False}

chunks_size = 5
processes = 6
figsize_x = 11
figsize_y = 11
