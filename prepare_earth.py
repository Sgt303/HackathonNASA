"""Prepares the Earth side of the database from global/earth/*.tif.

For each global Earth tile (run once, resumable):
  1. geographic tile -> reprojected to its UTM zone at EARTH_PX_M (projected tiles are used as they are),
  2. ESA WorldCover 'bare / sparse vegetation' mask (only this terrain goes to the database),
  3. for every body: downsample to TARGET_M[body] (masking BEFORE downsampling - averaging skips NaN),
  4. saved as earth_areas/[body]/elevation/<tile>.tif

Earth has one copy per body because it must be compared with the body at the SAME scale.

Usage:  python prepare_earth.py [--body moon] [--overwrite]
"""
import argparse
import json
from math import floor

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
from scipy.ndimage import binary_erosion

import config as C
from common import Raster, downsample, save


# ======================= READING A GLOBAL EARTH TILE =======================
def _read_tile(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("no CRS")
        nod = src.nodata
        if src.crs.is_projected:
            arr = src.read(1).astype("float32")
            if nod is not None and not np.isnan(nod):
                arr[arr == nod] = np.nan
            return Raster(arr, float(abs(src.transform.a)), src.transform, src.crs, path.stem)
        # geographic -> UTM zone of the tile centre (WorldCover masking needs metres)
        lon0 = (src.bounds.left + src.bounds.right) / 2
        lat0 = (src.bounds.bottom + src.bounds.top) / 2
        dst_crs = CRS.from_epsg((32600 if lat0 >= 0 else 32700) + int((lon0 + 180) // 6) + 1)
        tf, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds,
                                               resolution=C.EARTH_PX_M)
        arr = np.full((h, w), np.nan, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=arr, dst_transform=tf, dst_crs=dst_crs,
                  src_nodata=nod, dst_nodata=np.nan, resampling=Resampling.bilinear)
        return Raster(arr, float(C.EARTH_PX_M), tf, dst_crs, path.stem)


# ======================= WORLDCOVER MASK =======================
def _wc_tile_name(lat, lon):       # WorldCover tiles 3x3 deg, name = lower-left corner
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"

def bare_mask(r):
    """Downloads ESA WorldCover (10 m) for the raster's area; returns a bool mask (raster grid) of 'bare' terrain.
    Call on the NATIVE-resolution raster (before downsample). Pixel must be a multiple of 10 m."""
    H, W = r.arr.shape
    tf = r.transform
    px = abs(tf.a)
    up = round(px / 10)
    if up < 1 or abs(up * 10 - px) > 1e-6:
        raise ValueError(f"pixel {px:.2f} m is not a multiple of 10 m (set EARTH_PX_M in config.py or reproject the tile)")

    left, top = tf.c, tf.f
    right, bottom = left + W * tf.a, top + H * tf.e
    west, south, east, north = transform_bounds(r.crs, "EPSG:4326", left, bottom, right, top)
    tiles = [(la, lo) for la in range(floor(south / 3) * 3, floor(north / 3) * 3 + 1, 3)
                      for lo in range(floor(west / 3) * 3, floor(east / 3) * 3 + 1, 3)]
    print("WorldCover tiles:", [_wc_tile_name(*t) for t in tiles])

    wc_tf = tf * Affine.scale(1 / up)
    wc = np.zeros((H * up, W * up), dtype="uint8")
    loaded = 0
    for la, lo in tiles:
        try:
            with rasterio.open(C.WC_URL.format(tile=_wc_tile_name(la, lo))) as src:
                reproject(source=rasterio.band(src, 1), destination=wc, dst_transform=wc_tf, dst_crs=r.crs,
                          src_nodata=0, dst_nodata=0, init_dest_nodata=False, resampling=Resampling.nearest)
            loaded += 1
        except rasterio.errors.RasterioIOError as e:
            print("  WorldCover tile unavailable:", _wc_tile_name(la, lo), "->", e)
    if loaded == 0:
        raise RuntimeError("no WorldCover tile could be loaded (ocean / outside coverage / no internet)")

    frac = np.isin(wc, C.BARE_CLASSES).reshape(H, up, W, up).sum(axis=(1, 3), dtype="uint16") / up**2
    bare = frac >= C.BARE_FRAC - 1e-9
    if C.ERODE_PX > 0:     # drop pixels at the border (gradient at a forest edge picks up canopy height)
        bare = binary_erosion(bare, iterations=C.ERODE_PX, border_value=1)
    bare &= np.isfinite(r.arr)
    print(f"{r.name}: bare {bare.mean():.1%}")
    return bare


# ======================= MAIN =======================
def _meta(body):
    return dict(target_m=C.TARGET_M[body], earth_px=C.EARTH_PX_M, bare_frac=C.BARE_FRAC,
                bare_classes=list(C.BARE_CLASSES), erode_px=C.ERODE_PX)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=sorted(C.TARGET_M))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    bodies = [a.body] if a.body else sorted(C.TARGET_M)

    d = C.GLOBAL_DIR / "earth"
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff")) if d.is_dir() else []
    if not files:
        raise SystemExit(f"No GeoTIFFs in {d}")

    for b in bodies:                      # do not mix tiles prepared with different parameters
        out = C.EARTH_AREAS_DIR / b
        out.mkdir(parents=True, exist_ok=True)
        mp = out / "meta.json"
        if mp.exists() and not a.overwrite:
            old = json.loads(mp.read_text())
            if old != _meta(b):
                raise SystemExit(f"{out} was prepared with other parameters:\n  {old}\nnow:\n  {_meta(b)}\n"
                                 "Use --overwrite or change the parameters back.")
        mp.write_text(json.dumps(_meta(b), indent=1))

    done = failed = 0
    for p in files:
        target = {b: C.EARTH_AREAS_DIR / b / C.ELEVATION / f"{p.stem}.tif" for b in bodies}
        todo = [b for b in bodies if a.overwrite or not target[b].exists()]
        if not todo:
            print(f"{p.stem}: already prepared, skipping")
            continue
        try:
            r = _read_tile(p)
            mask = bare_mask(r)
            if not mask.any():
                print(f"{p.stem}: no bare terrain, nothing to save")
                continue
            r = Raster(np.where(mask, r.arr, np.nan).astype("float32"), r.pixel_m, r.transform, r.crs, r.name)
            for b in todo:
                dr = downsample(r, C.TARGET_M[b])
                if min(dr.arr.shape) < C.N_PX or not np.isfinite(dr.arr).any():
                    print(f"{p.stem} [{b}]: smaller than one window or empty, skipped")
                    continue
                save(target[b], dr.arr, dr.transform, dr.crs)
                print(f"  saved {target[b]}")
            done += 1
        except Exception as e:
            failed += 1
            print(f"SKIPPED {p.stem}: {type(e).__name__}: {e}")
    print(f"tiles processed: {done}, skipped: {failed}")


if __name__ == "__main__":
    main()
