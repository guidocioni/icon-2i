import argparse
from utils import get_latest_model_run


def parse_timesteps(timesteps_str):
    """Parse timesteps string into list of integers.

    Accepts:
    - Comma-separated values: "0,1,5,6" → [0, 1, 5, 6]
    - Range: "0-72" → [0, 1, 2, ..., 72]
    - Combination: "0,1,5-8,12" → [0, 1, 5, 6, 7, 8, 12]

    Returns None if timesteps_str is None (default to all timesteps).
    """
    if timesteps_str is None:
        return None

    timesteps = []
    parts = timesteps_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range
            start, end = part.split('-')
            timesteps.extend(range(int(start), int(end) + 1))
        else:
            # Handle single value
            timesteps.append(int(part))

    return sorted(set(timesteps))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action="store_true", default=False, help="Enable debug mode"
)
parser.add_argument(
    "--projection", type=str, default="it", help="Map projection to use"
)
parser.add_argument(
    "--level", type=int, default=850, help="Level (used only for some plots)"
)
parser.add_argument(
    "--run",
    type=str,
    default=get_latest_model_run(),
    help="Forecast run to fetch (format %%Y%%m%%d%%H)",
)
parser.add_argument(
    "--timesteps",
    type=str,
    default=None,
    help="Timesteps to plot (e.g., '0,1,5,6' or '0-72' or '0,1,5-8,12'). Default: all timesteps",
)
args = parser.parse_args()

debug = args.debug
projection = args.projection
run = args.run
level = args.level
timesteps = parse_timesteps(args.timesteps)
