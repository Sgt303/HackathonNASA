"""Shared helpers: sites CSV, raster I/O, downsampling, gradients, window geolocation."""
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS as _PCRS, Transformer
from scipy.ndimage import gaussian_filter, zoom

import config as C


# ======================= SMALL UTILITIES =======================
def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

def lon180(x):
    return (x + 180) % 360 - 180            # longitude to -180..180

def area_dir(base, body, sub):
    return Path(base) / body / sub

def block_slices():
    """[(block, start, end)] - column ranges of the feature blocks inside a vector."""
    dims = {"elev_scale": 1, "elev_shape": len(C.Q_SHAPE), "slope": len(C.Q_SLOPE), "orient": C.N_BINS}
    out, i = [], 0
    for b in C.BLOCK_W:
        out.append((b, i, i + dims[b]))
        i += dims[b]
    return out


# ======================= SITES (input_sites.csv) =======================
@dataclass
class Site:
    id: str
    name: str
    lat: float
    lon: float
    body: str

    @property
    def stem(self):
        return f"{self.id}_{safe(self.name)}"       # file name of this site's area in every sub-folder


def read_sites(path=None, body=None):
    """Reads input_sites.csv (columns: id, name, latitude, longitude, body). Optional filter by body."""
    path = Path(path or C.INPUT_SITES_CSV)
    need = ["id", "name", "latitude", "longitude", "body"]
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in rd]
    if not rows:
        return []
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; expected header: {', '.join(need)}")
    sites, seen = [], set()
    for i, row in enumerate(rows, start=2):
        b = row["body"].lower()
        if b not in C.TARGET_M:
            raise ValueError(f"{path.name} line {i}: unknown body '{row['body']}' (allowed: {sorted(C.TARGET_M)})")
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if abs(lat) > 90:
            raise ValueError(f"{path.name} line {i}: latitude {lat} out of range")
        if (b, row["id"]) in seen:
            raise ValueError(f"{path.name} line {i}: duplicate id '{row['id']}' for {b}")
        seen.add((b, row["id"]))
        sites.append(Site(row["id"], row["name"], lat, lon, b))
    return [s for s in sites if body is None or s.body == body]


# ======================= RASTER =======================
@dataclass
class Raster:
    arr: np.ndarray
    pixel_m: float           # pixel size in metres
    transform: object        # affine
    crs: object
    name: str = ""


def load(path):
    """Reads a GeoTIFF written by this pipeline (projected CRS in metres). nodata -> NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None and not np.isnan(nod):
            arr[arr == nod] = np.nan
        crs, tf = src.crs, src.transform
    if crs is None or not crs.is_projected:
        raise ValueError(f"{path}: needs a projected CRS in metres (the pipeline writes these itself).")
    if abs(abs(tf.a) - abs(tf.e)) > 1e-6 * abs(tf.a):
        raise ValueError(f"{path}: non-square pixels ({abs(tf.a)} x {abs(tf.e)}) are not supported.")
    return Raster(arr, float(abs(tf.a)), tf, crs, Path(path).stem)


def save(path, arr, transform, crs):
    """float32 GeoTIFF, NaN = nodata. Atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)
    os.replace(tmp, path)


def downsample(r, target_m):
    """Reduces resolution to target_m (NaN-aware averaging, also for fractional factors). Never upsamples.
    Returns a new Raster with the ACTUAL pixel size. NaN (e.g. masked terrain) is neither filled nor
    pulls the averages down near the edge of the mask."""
    f = target_m / r.pixel_m
    if f < 0.98:
        raise ValueError(f"{r.name}: target {target_m} m is finer than the data ({r.pixel_m:.1f} m); upsampling is not allowed. "
                         "Use data with a pixel <= the target scale, or raise TARGET_M in config.py.")
    if f < 1.02:
        return r
    a = r.arr
    ok = np.isfinite(a)
    num = gaussian_filter(np.where(ok, a, 0).astype("float32"), f / 2, mode="nearest")
    den = gaussian_filter(ok.astype("float32"), f / 2, mode="nearest")
    valid = ok & (den > 0.5)                                            # original NaN stay NaN
    sm = np.where(valid, num / np.maximum(den, 1e-6), 0.0).astype("float32")
    new = (round(a.shape[0] / f), round(a.shape[1] / f))
    zf = (new[0] / a.shape[0], new[1] / a.shape[1])
    zs = zoom(sm, zf, order=1, grid_mode=True, mode="nearest")
    zv = zoom(valid.astype("float32"), zf, order=1, grid_mode=True, mode="nearest")
    out = np.where(zv > 0.5, zs / np.maximum(zv, 1e-6), np.nan)         # normalise by weights -> no bias towards 0
    actual = r.pixel_m * a.shape[0] / new[0]
    tf = None
    if r.transform is not None:
        tf = r.transform * Affine.scale(a.shape[1] / out.shape[1], a.shape[0] / out.shape[0])
    print(f"{r.name}: {r.pixel_m:.1f} m -> {actual:.1f} m, {a.shape} -> {out.shape}")
    return Raster(out.astype("float32"), actual, tf, r.crs, r.name)


def grad(dem, px):
    """Gradient maps: gx, gy (dimensionless dz/dx, dz/dy, rows grow downwards) and slope in degrees."""
    gy, gx = np.gradient(dem, px)
    return gx, gy, np.degrees(np.arctan(np.hypot(gx, gy)))


# ======================= GEOLOCATION OF WINDOWS =======================
_TR = {}
def _transformer(crs):
    """Raster CRS -> geographic CRS of the same body/datum; result order (lon, lat)."""
    key = crs.to_wkt()
    if key not in _TR:
        p = _PCRS.from_wkt(key)
        _TR[key] = Transformer.from_crs(p, p.geodetic_crs, always_xy=True)
    return _TR[key]

def window_centers(r, pos, n):
    """Centres (lat, lon) of n x n windows with top-left corners pos (N, 2) = (row, col),
    computed in the raster's own CRS."""
    tf = r.transform
    cx, cy = pos[:, 1] + n / 2, pos[:, 0] + n / 2
    x = tf.a * cx + tf.b * cy + tf.c
    y = tf.d * cx + tf.e * cy + tf.f
    lon, lat = _transformer(r.crs).transform(x, y)
    return np.asarray(lat, dtype="float64"), lon180(np.asarray(lon, dtype="float64"))
