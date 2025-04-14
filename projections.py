from definitions import IMAGES_DIR
import os

# Dictionary to map the output folder based on the projection employed
subfolder_images = {"nord": os.path.join(IMAGES_DIR, "nord")}

proj_defs = {
    "nord": {
        "projection": "merc",
        "llcrnrlon": 6.6,
        "llcrnrlat": 43.5,
        "urcrnrlon": 14.5,
        "urcrnrlat": 47.2,
        "resolution": "h",
        "epsg": 3857,
    },
    "centro": {
        "projection": "merc",
        "llcrnrlon": 9.8,
        "llcrnrlat": 40.9,
        "urcrnrlon": 15.2,
        "urcrnrlat": 44.1,
        "resolution": "h",
        "epsg": 3857,
    },
    "sud": {
        "projection": "merc",
        "llcrnrlon": 11.8,
        "llcrnrlat": 36.5,
        "urcrnrlon": 18.6,
        "urcrnrlat": 42.0,
        "resolution": "h",
        "epsg": 3857,
    },
    "it": {
        "projection": "merc",
        "llcrnrlon": 6,
        "llcrnrlat": 36,
        "urcrnrlon": 19,
        "urcrnrlat": 47.7,
        "resolution": "i",
        "epsg": 3857,
    },
}
