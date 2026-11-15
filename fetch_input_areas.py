"""Cuts an area around every site of input_sites.csv out of the global rasters in global/[body]/.

For each site:
  1. local azimuthal-equidistant projection centred on the site (metres, no lat/lon distortion),
  2. the global tiles that cover the area are reprojected into it at their native pixel size,
  3. the result is downsampled to TARGET_M[body] (the common scale used for the comparison with Earth),
  4. saved as input_areas/[body]/elevation/<id>_<name>.tif

Usage:  python fetch_input_areas.py [--body moon] [--overwrite]
"""
import argparse

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS as _PCRS
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform_bounds

import config as C
from common import Raster, read_sites, downsample, save


def _radius(crs):
    el = _PCRS.from_wkt(crs.to_wkt()).ellipsoid
    if el is None:
        raise ValueError("CRS has no ellipsoid")
    return el.semi_major_metre

def _native_pixel_m(src):
    """Pixel size in metres. Geographic CRS: N-S size (E-W is finer away from the equator, so N-S is the limit)."""
    if src.crs.is_projected:
        return abs(src.transform.a)
    return abs(src.transform.e) * np.pi / 180 * _radius(src.crs)

def _index_tiles(body):
    """Headers of all global tiles of a body (read once)."""
    d = C.GLOBAL_DIR / body
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff")) if d.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs in {d}")
    exp = C.BODY_RADIUS_M[body]
    tiles = []
    for p in files:
        with rasterio.open(p) as s:
            if s.crs is None:
                raise ValueError(f"{p.name}: no CRS")
            R = _radius(s.crs)
            if abs(R - exp) / exp > 0.01:
                raise ValueError(f"{p.name}: CRS ellipsoid radius {R:.0f} m does not match {body} ({exp} m) - "
                                 "wrong CRS in the file?")
            tiles.append({"path": p, "bounds": s.bounds, "crs": s.crs, "px": _native_pixel_m(s)})
    return tiles

def _local_crs(src_crs, lat, lon):
    """Azimuthal equidistant projection centred on (lat, lon), on the SAME datum as the global data."""
    base = _PCRS.from_wkt(src_crs.to_wkt()).geodetic_crs
    pc = ProjectedCRS(conversion=AzimuthalEquidistantConversion(lat, lon), geodetic_crs=base)
    return CRS.from_wkt(pc.to_wkt())


def fetch_site(site, tiles, overwrite=False):
    out = C.INPUT_AREAS_DIR / site.body / C.ELEVATION / f"{site.stem}.tif"
    if out.exists() and not overwrite:
        print(f"{site.body}/{site.stem}: exists, skipping")
        return
    area_m = C.AREA_KM[site.body] * 1000
    crs = _local_crs(tiles[0]["crs"], site.lat, site.lon)

    # tiles that overlap the area (+ margin for interpolation)
    half = area_m / 2 + 3 * max(t["px"] for t in tiles)
    hit = []
    for t in tiles:
        l, b, r, u = transform_bounds(crs, t["crs"], -half, -half, half, half, densify_pts=21)
        tb = t["bounds"]
        if l < tb.right and r > tb.left and b < tb.top and u > tb.bottom:
            hit.append(t)
    if not hit:
        raise ValueError("no global tile covers this site")
    px = [t["px"] for t in hit]
    if max(px) / min(px) > 1.02:
        raise ValueError(f"tiles covering the site have different resolutions: {sorted(set(round(p, 1) for p in px))} m")
    native = float(px[0])

    n = int(np.ceil(area_m / native))
    tf = Affine(native, 0, -n * native / 2, 0, -native, n * native / 2)    # site = centre of the grid
    arr = np.full((n, n), np.nan, dtype="float32")
    for t in hit:
        with rasterio.open(t["path"]) as src:
            reproject(source=rasterio.band(src, 1), destination=arr, dst_transform=tf, dst_crs=crs,
                      src_nodata=src.nodata, dst_nodata=np.nan, init_dest_nodata=False,
                      resampling=Resampling.bilinear)
    if not np.isfinite(arr).any():
        raise ValueError("global tiles overlap the area but contain no valid data there")
    print(f"{site.body}/{site.stem}: {n}x{n} px @ {native:.1f} m, NaN {np.isnan(arr).mean():.1%}")

    r = downsample(Raster(arr, native, tf, crs, site.stem), C.TARGET_M[site.body])
    if min(r.arr.shape) < C.N_PX:
        raise ValueError(f"area is {r.arr.shape[1]}x{r.arr.shape[0]} px at {r.pixel_m:.0f} m, smaller than the "
                         f"{C.N_PX} px window. Increase AREA_KM['{site.body}'] in config.py "
                         f"(window = {C.N_PX * C.TARGET_M[site.body] / 1000:.2f} km).")
    save(out, r.arr, r.transform, r.crs)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=sorted(C.TARGET_M))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    sites = read_sites(body=a.body)
    index, done, failed = {}, 0, 0
    for s in sites:
        try:
            if s.body not in index:
                index[s.body] = _index_tiles(s.body)
            fetch_site(s, index[s.body], a.overwrite)
            done += 1
        except Exception as e:
            failed += 1
            print(f"SKIPPED {s.body}/{s.stem}: {type(e).__name__}: {e}")
    print(f"sites: {len(sites)}, ok: {done}, skipped: {failed}")


if __name__ == "__main__":
    main()
