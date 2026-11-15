"""Matches every query window of the input areas against the Earth windows and writes output_areas.json.

Same method as planet2vec.py:
  1. standardise: mean / std of every feature column, fitted on ALL Earth windows of that body's scale,
     applied to Earth and to the query windows,
  2. embedding = per block:  BLOCK_W[block] * Z_block / sqrt(dim_block),
  3. distance = Euclidean norm in the embedding space; the K closest Earth windows are the matches.

Weights (BLOCK_W) are applied here, so changing them only needs a re-run of this script.

Usage:  python match.py [--body moon]
"""
import argparse
import json
import os

import numpy as np

import config as C
from common import read_sites, block_slices


# ======================= LOADING =======================
def load_earth(body):
    """All Earth window vectors of a body. Returns dict(feats, lat, lon, tile, tiles, n, pixel_m)."""
    files = sorted((C.EARTH_AREAS_DIR / body / C.VECTORS).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No Earth vectors in {C.EARTH_AREAS_DIR / body / C.VECTORS} - run prepare_earth.py, gradient.py, vectorize.py")
    F, LAT, LON, T, names, px = [], [], [], [], [], []
    n = None
    for j, f in enumerate(files):
        z = np.load(f)
        n = int(z["n"]) if n is None else n
        if int(z["n"]) != n:
            raise ValueError(f"{f.name}: window size {int(z['n'])} px differs from {n} px")
        F.append(z["feats"]); LAT.append(z["lat"]); LON.append(z["lon"])
        T.append(np.full(len(z["feats"]), j, dtype="int32"))
        names.append(f.stem); px.append(float(z["pixel_m"]))
    px = np.array(px)
    if (px.max() - px.min()) / px.mean() > 0.02:
        raise ValueError(f"Earth tiles have different scales: {px.min():.1f}-{px.max():.1f} m/px")
    return dict(feats=np.concatenate(F), lat=np.concatenate(LAT), lon=np.concatenate(LON),
                tile=np.concatenate(T), tiles=names, n=n, pixel_m=float(px.mean()))


# ======================= METHOD =======================
def fit_scaler(X):
    mu = np.nanmean(X, axis=0, dtype="float64")
    s = np.nanstd(X, axis=0, dtype="float64")
    return mu, np.where(s > 1e-9, s, 1.0)

def embed(X, mu, sd):
    Z = (X - mu) / sd
    E = np.empty_like(Z, dtype="float32")
    for name, a, b in block_slices():
        E[:, a:b] = C.BLOCK_W[name] * Z[:, a:b] / np.sqrt(b - a)
    return E

def _dist(E, q, chunk=500_000):
    out = np.empty(len(E), dtype="float32")
    for i in range(0, len(E), chunk):
        out[i:i + chunk] = np.linalg.norm(E[i:i + chunk] - q, axis=1)
    return out

def search(q, E, k):
    """Top-k rows of E closest to the embedding q: (indices, distances), best first."""
    d = _dist(E, q)
    k = min(k, len(d))
    top = np.argpartition(d, k - 1)[:k]          # NaN distances sort last
    top = top[np.argsort(d[top])]
    return top, d[top]


# ======================= OUTPUT (same format as sites.json) =======================
def match_body(body, first_id=1):
    earth = load_earth(body)
    mu, sd = fit_scaler(earth["feats"])
    E = embed(earth["feats"], mu, sd)
    print(f"{body}: {len(E)} Earth windows from {len(earth['tiles'])} tiles, dim {E.shape[1]}")

    entries, nid = [], first_id
    for site in read_sites(body=body):
        p = C.INPUT_AREAS_DIR / body / C.VECTORS / f"{site.stem}.npz"
        if not p.exists():
            print(f"SKIPPED {body}/{site.stem}: no vectors (run fetch_input_areas.py, gradient.py, vectorize.py)")
            continue
        z = np.load(p)
        if int(z["n"]) != earth["n"] or abs(float(z["pixel_m"]) - earth["pixel_m"]) / earth["pixel_m"] > 0.02:
            print(f"SKIPPED {body}/{site.stem}: scale mismatch - query {float(z['pixel_m']):.1f} m/{int(z['n'])} px, "
                  f"Earth {earth['pixel_m']:.1f} m/{earth['n']} px")
            continue
        Q = embed(z["feats"], mu, sd)
        side_km = round(int(z["n"]) * float(z["pixel_m"]) / 1000, 3)
        for sq in range(len(Q)):
            top, _ = search(Q[sq], E, C.K)
            for rank, j in enumerate(top, 1):
                lat, lon = earth["lat"][j], earth["lon"][j]
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                entries.append({
                    "id": nid, "body": body, "source": site.name,
                    "site": {"name": f"{site.name}, kwadrat {sq}",
                             "lat": round(float(z["lat"][sq]), 4), "lon": round(float(z["lon"][sq]), 4)},
                    "earth": {"name": f"{earth['tiles'][earth['tile'][j]]}, hit {rank}",
                              "lat": round(float(lat), 4), "lon": round(float(lon), 4)},
                    "side_km": side_km,
                })
                nid += 1
        print(f"{body}/{site.stem}: {len(Q)} query windows matched")
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=sorted(C.TARGET_M))
    a = ap.parse_args()

    data = []
    for b in ([a.body] if a.body else sorted(C.TARGET_M)):
        try:
            data += match_body(b)
        except FileNotFoundError as e:
            print(f"SKIPPED {b}: {e}")
    if a.body and C.OUTPUT_JSON.exists():        # single-body run: keep the other bodies' entries
        old = json.loads(C.OUTPUT_JSON.read_text(encoding="utf-8") or "[]")
        data = [e for e in old if e.get("body") != a.body] + data
    tmp = str(C.OUTPUT_JSON) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, C.OUTPUT_JSON)
    print(f"{C.OUTPUT_JSON.name}: {len(data)} entries")


if __name__ == "__main__":
    main()
