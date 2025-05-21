import argparse
from utils import get_latest_model_run

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action="store_true", default=False, help="Enable debug mode"
)
parser.add_argument(
    "--projection", type=str, default="it", help="Map projection to use"
)
parser.add_argument(
    "--level", type=int, default=None, help="Level (used only for some plots)"
)
parser.add_argument(
    "--run",
    type=str,
    default=get_latest_model_run(),
    help="Forecast run to fetch (format %Y%m%d%H)",
)
args = parser.parse_args()

debug = args.debug
projection = args.projection
run = args.run
level = args.level
