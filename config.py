"""Shared settings for the planet2vec pipeline. Every constant lives here, no script defines its own.

Pipeline (run in this order):
    prepare_earth.py      global/earth/*.tif        -> earth_areas/[body]/elevation/
    fetch_input_areas.py  input_sites.csv + global/ -> input_areas/[body]/elevation/
    gradient.py           .../elevation/            -> .../slope/, .../grad_x/, .../grad_y/   (input_areas AND earth_areas)
    vectorize.py          elevation + maps          -> .../vectors/*.npz                       (input_areas AND earth_areas)
    match.py              vectors                   -> output_areas.json  (read by index.html)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PLANET2VEC_ROOT") or Path(__file__).resolve().parent)

# ======================= FOLDERS =======================
GLOBAL_DIR = ROOT / "global"                  # global/{earth,moon,mars}/*.tif  (batched global elevation)
INPUT_SITES_CSV = ROOT / "input_sites.csv"    # id, name, latitude, longitude, body
INPUT_AREAS_DIR = ROOT / "input_areas"        # input_areas/[body]/{elevation,slope,grad_x,grad_y,vectors}/
EARTH_AREAS_DIR = ROOT / "earth_areas"        # earth_areas/[body]/...  same layout, Earth prepared at [body]'s scale
OUTPUT_JSON = ROOT / "output_areas.json"

ELEVATION, SLOPE, GRAD_X, GRAD_Y, VECTORS = "elevation", "slope", "grad_x", "grad_y", "vectors"

# ======================= METHODOLOGY (unchanged from planet2vec.py) =======================
Q_SHAPE = [10, 20, 30, 40, 50, 60, 70, 80, 90]   # elevation distribution shape
Q_SLOPE = [5, 25, 50, 75, 95]
N_BINS = 12                                       # orientation bins (12 x 15 deg)
MAX_NAN = 0.10                                    # drop windows with >10% NaN
BLOCK_W = {"elev_scale": 2.0, "elev_shape": 1.5, "slope": 1.0, "orient": 1.0}

N_PX = 64                                         # window side in pixels (same for Moon, Mars and Earth)

# Common analysis scale per body [m/px]. Earth is downsampled to it, and so are the input areas.
# (This replaces BODY_PIXEL_M: it is no longer "native pixel of the target" but "scale of the comparison".)
TARGET_M = {"mars": 200, "moon": 40}

# Sanity check only (fetch_input_areas.py): the global raster's CRS must describe the right body.
# Catches e.g. Moon data mistakenly tagged as EPSG:4326 (Earth). Values from memory - verify. Tolerance 1%.
BODY_RADIUS_M = {"mars": 3_389_500, "moon": 1_737_400}

# ======================= INPUT AREAS =======================
# Side of the area cut around each site. Window = N_PX * TARGET_M = 2.56 km (Moon) / 12.8 km (Mars), so a 2 km area
# cannot hold even one window. The default below is 5 windows wide: room for the centre window plus
# the extra sample windows (RADIUS_PX = 2 * N_PX around the centre).
AREA_KM = {b: 5 * N_PX * m / 1000 for b, m in TARGET_M.items()}     # moon 12.8 km, mars 64 km

SAMPLES = 3                    # extra query windows per site (0 = only the centre window)
RADIUS_PX = 2 * N_PX           # sample windows lie within this many px of the centre window
MIN_SEP = N_PX // 2            # minimum spacing between query windows [px] (Chebyshev)
SEED = 0
K = 5                          # matches per query window

# ======================= EARTH =======================
STRIDE_EARTH = N_PX // 2       # stride of the Earth window grid
EARTH_PX_M = 30                # working pixel for geographic Earth tiles (reprojected to UTM); multiple of 10 m (WorldCover)
BARE_FRAC = 0.5                # min. fraction of 10 m WorldCover 'bare' pixels inside a pixel
BARE_CLASSES = (60,)           # WorldCover classes kept as terrain (60 = bare / sparse vegetation)
ERODE_PX = 1                   # erosion of the bare mask (edge of forest etc.)
# ESA WorldCover (10 m). For 2020: v100/2020/map/ ..._2020_v100_
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
