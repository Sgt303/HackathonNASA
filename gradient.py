"""Computes the parameter maps from every elevation GeoTIFF, for the input areas AND for the Earth areas:

    .../elevation/<name>.tif  ->  .../slope/<name>.tif   slope [deg]
                                  .../grad_x/<name>.tif  dz/dx   (columns)
                                  .../grad_y/<name>.tif  dz/dy   (rows, growing downwards)

The orientation histogram used by vectorize.py is derived from grad_x / grad_y (it needs both components).
NaN (masked terrain) spreads by one pixel into the maps, exactly as in planet2vec.py.

Usage:  python gradient.py [--body moon] [--overwrite]
"""
import argparse

import config as C
from common import load, save, grad


def run(body, overwrite=False):
    for base in (C.INPUT_AREAS_DIR, C.EARTH_AREAS_DIR):
        src_dir = base / body / C.ELEVATION
        if not src_dir.is_dir():
            print(f"{base.name}/{body}: no elevation folder, skipping")
            continue
        for f in sorted(src_dir.glob("*.tif")):
            outs = {k: base / body / k / f.name for k in (C.SLOPE, C.GRAD_X, C.GRAD_Y)}
            if not overwrite and all(p.exists() for p in outs.values()):
                print(f"{base.name}/{body}/{f.stem}: maps exist, skipping")
                continue
            r = load(f)
            gx, gy, sl = grad(r.arr, r.pixel_m)
            for k, a in ((C.SLOPE, sl), (C.GRAD_X, gx), (C.GRAD_Y, gy)):
                save(outs[k], a, r.transform, r.crs)
            print(f"{base.name}/{body}/{f.stem}: slope, grad_x, grad_y saved")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=sorted(C.TARGET_M))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for b in ([a.body] if a.body else sorted(C.TARGET_M)):
        run(b, a.overwrite)


if __name__ == "__main__":
    main()
