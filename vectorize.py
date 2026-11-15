"""Turns elevation + parameter maps into window feature vectors.

  Earth areas : every N_PX x N_PX window on a grid with stride STRIDE_EARTH (windows with >MAX_NAN NaN are dropped)
  Input areas : the centre window (index 0) + SAMPLES random windows around it - the query windows

Vectors are RAW features (elevation scale/shape, slope percentiles, sorted orientation histogram): no
standardisation, no weights. Both belong to match.py, so changing BLOCK_W never requires re-running this step.

Output: [input_areas|earth_areas]/[body]/vectors/<name>.npz with
  feats (N, D) float32, pos (N, 2) = (row, col) of the window's top-left corner, lat, lon (window centres),
  n (window side, px), pixel_m.

Usage:  python vectorize.py [--body moon] [--overwrite]
"""
import argparse
import os

import numpy as np

import config as C
from common import load, window_centers, safe


# ======================= FEATURES (same as planet2vec.py) =======================
def _orient_hist(gx, gy, n_bins=C.N_BINS):
    mag = np.hypot(gx, gy)
    ok = np.isfinite(mag)
    ang = (np.degrees(np.arctan2(gy[ok], gx[ok])) + 90) % 180      # tangent, mod 180 deg
    h, _ = np.histogram(ang, bins=n_bins, range=(0, 180), weights=mag[ok])
    return h / h.sum() if h.sum() > 0 else h

def _window_features(d, s, gx, gy):
    rel = d - np.nanmin(d)
    p5, p95 = np.nanpercentile(rel, [5, 95])
    rg = p95 - p5
    return {
        "elev_scale": np.array([np.log1p(rg)]),
        "elev_shape": (np.nanpercentile(rel, C.Q_SHAPE) - p5) / max(rg, 1e-3),
        "slope": np.nanpercentile(s, C.Q_SLOPE),
        "orient": np.sort(_orient_hist(gx, gy))[::-1],
    }

def _vector(f):
    return np.concatenate([f[b] for b in C.BLOCK_W])               # block order = BLOCK_W order


# ======================= WINDOW POSITIONS =======================
def _center(shape, n):
    return ((shape[0] - n) // 2, (shape[1] - n) // 2)

def _grid(shape, n, stride):
    return [(a, b) for a in range(0, shape[0] - n + 1, stride) for b in range(0, shape[1] - n + 1, stride)]

def _sample_positions(dem, n, samples, radius, seed, min_sep):
    """Centre window + `samples` random windows within `radius` px of it. Rejects windows outside the raster,
    with >MAX_NAN NaN, and closer than min_sep px (Chebyshev) to an already chosen one."""
    rng = np.random.default_rng(seed)
    c = _center(dem.shape, n)
    chosen, tries = [c], 0
    while len(chosen) < samples + 1:
        tries += 1
        if tries > 5000:
            print(f"WARNING: drew only {len(chosen) - 1}/{samples} extra windows "
                  "(increase AREA_KM / RADIUS_PX or decrease MIN_SEP)")
            break
        a = c[0] + int(rng.integers(-radius, radius + 1))
        b = c[1] + int(rng.integers(-radius, radius + 1))
        if a < 0 or b < 0 or a + n > dem.shape[0] or b + n > dem.shape[1]:
            continue
        if np.isnan(dem[a:a + n, b:b + n]).mean() > C.MAX_NAN:
            continue
        if any(max(abs(a - x), abs(b - y)) < min_sep for x, y in chosen):
            continue
        chosen.append((a, b))
    return chosen


# ======================= ONE AREA =======================
def _load_maps(base, body, name):
    p = {k: base / body / k / f"{name}.tif" for k in (C.ELEVATION, C.SLOPE, C.GRAD_X, C.GRAD_Y)}
    for k, path in p.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing" + (" - run gradient.py" if k != C.ELEVATION else ""))
    r = load(p[C.ELEVATION])
    sl, gx, gy = (load(p[k]).arr for k in (C.SLOPE, C.GRAD_X, C.GRAD_Y))
    if not (r.arr.shape == sl.shape == gx.shape == gy.shape):
        raise ValueError(f"{name}: elevation and maps have different shapes - re-run gradient.py --overwrite")
    return r, sl, gx, gy

def _vectorize_area(r, sl, gx, gy, cand, n):
    feats, kept = [], []
    for (a, b) in cand:
        sx = np.s_[a:a + n, b:b + n]
        if np.isnan(r.arr[sx]).mean() > C.MAX_NAN:
            continue
        feats.append(_vector(_window_features(r.arr[sx], sl[sx], gx[sx], gy[sx])))
        kept.append((a, b))
    return (np.array(feats, dtype="float32").reshape(len(feats), -1),
            np.array(kept, dtype="int32").reshape(len(kept), 2))

def _save(path, feats, pos, r, n):
    lat, lon = window_centers(r, pos, n)
    tmp = path.with_name(path.stem + ".tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, feats=feats, pos=pos, lat=lat, lon=lon, n=n, pixel_m=r.pixel_m)
    os.replace(tmp, path)


def run(body, overwrite=False):
    from common import read_sites
    n = C.N_PX

    # --- Earth: all windows of every prepared tile ---
    edir = C.EARTH_AREAS_DIR / body / C.ELEVATION
    for f in (sorted(edir.glob("*.tif")) if edir.is_dir() else []):
        out = C.EARTH_AREAS_DIR / body / C.VECTORS / f"{f.stem}.npz"
        if out.exists() and not overwrite:
            print(f"earth/{body}/{f.stem}: vectors exist, skipping")
            continue
        try:
            r, sl, gx, gy = _load_maps(C.EARTH_AREAS_DIR, body, f.stem)
            feats, pos = _vectorize_area(r, sl, gx, gy, _grid(r.arr.shape, n, C.STRIDE_EARTH), n)
            if len(feats) == 0:
                print(f"earth/{body}/{f.stem}: no window with >={1 - C.MAX_NAN:.0%} valid pixels")
                continue
            _save(out, feats, pos, r, n)
            print(f"earth/{body}/{f.stem}: {len(feats)} windows")
        except Exception as e:
            print(f"SKIPPED earth/{body}/{f.stem}: {type(e).__name__}: {e}")

    # --- input areas: query windows ---
    for s in read_sites(body=body):
        out = C.INPUT_AREAS_DIR / body / C.VECTORS / f"{s.stem}.npz"
        if out.exists() and not overwrite:
            print(f"{body}/{s.stem}: vectors exist, skipping")
            continue
        try:
            r, sl, gx, gy = _load_maps(C.INPUT_AREAS_DIR, body, s.stem)
            if min(r.arr.shape) < n:
                raise ValueError(f"area {r.arr.shape} px smaller than the {n} px window (AREA_KM in config.py)")
            cand = _sample_positions(r.arr, n, C.SAMPLES, C.RADIUS_PX, C.SEED, C.MIN_SEP)
            feats, pos = _vectorize_area(r, sl, gx, gy, cand, n)
            if len(pos) == 0 or tuple(pos[0]) != tuple(cand[0]):
                raise ValueError(f"centre window has >{C.MAX_NAN:.0%} NaN")
            _save(out, feats, pos, r, n)
            print(f"{body}/{s.stem}: {len(feats)} query windows")
        except Exception as e:
            print(f"SKIPPED {body}/{s.stem}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=sorted(C.TARGET_M))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for b in ([a.body] if a.body else sorted(C.TARGET_M)):
        run(b, a.overwrite)


if __name__ == "__main__":
    main()
