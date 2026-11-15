/*
 * recalc.js  -  jeden plik, dwa zastosowania (bez zależności zewnętrznych)
 *
 *  1) PRZEGLĄDARKA (index.html):  window.Planet2Vec
 *       loadFeatures(body)            wczytuje features/{body}.json + pliki binarne
 *       computeEntries(db, weights)   ten sam algorytm co planet2vec.py (search + collect_hits)
 *                                     -> wpisy w formacie sites.json
 *
 *  2) NODE (jednorazowo, po `python planet2vec.py --features`):
 *       node recalc.js profiles [--features features] [--out profiles]
 *         rysuje panele PNG (mapa wysokości + histogramy, jak w notebooku) do
 *         profiles/moon, profiles/mars (kwadraty celu) i profiles/earth (okna Ziemi) + profiles/earth/index.json
 *       node recalc.js search <body> [w_elev_scale w_elev_shape w_slope w_orient]
 *         wypisuje trafienia dla podanych wag (do sprawdzania wyników w konsoli)
 *
 * Algorytm (identyczny z planet2vec.py):
 *   embedding = hstack_b( w_b * Z_b / sqrt(dim_b) ),  Z_b = (cechy_b - mu_b) / sd_b      (Z jest w features/*.f32)
 *   odległość = ||embedding_zapytania - embedding_okna||  ->  d^2 = sum_b w_b^2 * ||dZ_b||^2 / dim_b
 *   mu/sd nie zależą od wag, dlatego zmiana wag nie wymaga ponownego czytania GeoTIFF-ów.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
    if (typeof require !== 'undefined' && require.main === module) api.cli(process.argv.slice(2));
  } else {
    root.Planet2Vec = api;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ============================== WSPÓLNE ============================== */
  const BLOCKS = ['elev_scale', 'elev_shape', 'slope', 'orient'];
  // Muszą się zgadzać z BLOCK_W w planet2vec.py (po wczytaniu features/{body}.json są nadpisywane wartościami z pliku).
  const DEFAULT_WEIGHTS = { elev_scale: 2.0, elev_shape: 1.5, slope: 1.0, orient: 1.0 };

  const safe = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, '_');
  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  const key4 = (body, lat, lon) => body + '|' + r4(lat).toFixed(4) + '|' + r4(lon).toFixed(4);

  // Ścieżki obrazków (te same reguły w Node i w przeglądarce)
  const queryProfilePath = (body, source, square) => `profiles/${body}/${safe(source)}_${square}.png`;
  const earthFileName = (body, chunk, row, col) => `${body}_${safe(chunk)}_${row}_${col}.png`;

  function makeDb(meta, feat, geo, idx) {
    return { meta, body: meta.body, n: meta.n, pixel_m: meta.pixel_m, k: meta.k, count: meta.count, dim: meta.dim,
             blocks: meta.blocks, rasters: meta.rasters, queries: meta.queries, feat, geo, idx };
  }

  function defaultWeights(db) {
    const w = {};
    for (const b of db.blocks) w[b.name] = b.default_weight;
    return w;
  }

  /* ============================== ALGORYTM ============================== */
  // Top-k najbliższych okien dla jednego zapytania. c[i] = w_b^2 / dim_b dla kolumny i należącej do bloku b.
  function searchOne(db, zq, c, k, d2) {
    const { feat, count, dim } = db;
    for (let j = 0, o = 0; j < count; j++, o += dim) {
      let s = 0;
      for (let i = 0; i < dim; i++) { const t = zq[i] - feat[o + i]; s += c[i] * t * t; }
      d2[j] = s;
    }
    const topI = [], topV = [];
    const finite = [];
    for (let j = 0; j < count; j++) {
      const v = d2[j];
      if (!(v >= 0) || v === Infinity) continue;           // NaN / inf -> pomijamy (jak w numpy argsort: na końcu)
      finite.push(Math.sqrt(v));
      if (topV.length < k || v < topV[topV.length - 1]) {
        let p = topV.length;
        while (p > 0 && topV[p - 1] > v) p--;
        topV.splice(p, 0, v); topI.splice(p, 0, j);
        if (topV.length > k) { topV.pop(); topI.pop(); }
      }
    }
    finite.sort((a, b) => a - b);
    const m = finite.length >> 1;
    const median = finite.length % 2 ? finite[m] : (finite[m - 1] + finite[m]) / 2;
    return { idx: topI, dist: topV.map(Math.sqrt), median };
  }

  function coefficients(db, w) {
    const c = new Float64Array(db.dim);
    for (const b of db.blocks) {
      const wb = +w[b.name];
      c.fill(wb * wb / b.dim, b.start, b.end);
    }
    return c;
  }

  function validateWeights(w) {
    let sum = 0;
    for (const b of BLOCKS) {
      const v = +w[b];
      if (!isFinite(v) || v < 0) throw new Error('Weight "' + b + '" must be a number >= 0.');
      sum += v;
    }
    if (sum <= 0) throw new Error('At least one weight must be greater than 0.');
  }

  /**
   * Odpowiednik search() + collect_hits() + _entries_for_site() z planet2vec.py.
   * Zwraca wpisy w formacie sites.json (+ dist, dist_rel, source, square).
   */
  function computeEntries(db, weights, k) {
    validateWeights(weights);
    k = k || db.k;
    const c = coefficients(db, weights);
    const d2 = new Float64Array(db.count);
    const out = [];
    let id = 0;
    for (const q of db.queries) {
      const res = searchOne(db, q.z, c, k, d2);
      res.idx.forEach((j, t) => {
        const lat = db.geo[2 * j], lon = db.geo[2 * j + 1];
        const rank = t + 1;
        if (!isFinite(lat) || !isFinite(lon)) return;       // brak georeferencji - pomijamy (jak w Pythonie)
        const chunk = db.rasters[db.idx[3 * j]];
        out.push({
          id: ++id, body: db.body, source: q.source, square: q.square,
          site: { name: q.source + ', kwadrat ' + q.square, lat: r4(q.lat), lon: r4(q.lon) },
          earth: { name: chunk + ', hit ' + rank, lat: r4(lat), lon: r4(lon) },
          side_km: Math.round(q.side_m) / 1000,
          dist: res.dist[t], dist_rel: res.dist[t] / res.median,
        });
      });
    }
    return out;
  }

  /* ============================== ŁADOWANIE ============================== */
  const toAligned = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

  async function loadFeatures(body, base) {
    base = base || 'features/';
    const getJson = async (f) => {
      const r = await fetch(base + f, { cache: 'no-store' });
      if (!r.ok) throw new Error(base + f + ': HTTP ' + r.status);
      return r.json();
    };
    const getBin = async (f) => {
      const r = await fetch(base + f, { cache: 'no-store' });
      if (!r.ok) throw new Error(base + f + ': HTTP ' + r.status);
      return r.arrayBuffer();
    };
    const meta = await getJson(body + '.json');
    const [f, g, i] = await Promise.all([getBin(meta.files.feat), getBin(meta.files.geo), getBin(meta.files.idx)]);
    return makeDb(meta, new Float32Array(f), new Float64Array(g), new Int32Array(i));
  }

  function loadFeaturesFs(dir, body) {
    const fs = require('fs'), path = require('path');
    const meta = JSON.parse(fs.readFileSync(path.join(dir, body + '.json'), 'utf8'));
    const rd = (f) => toAligned(fs.readFileSync(path.join(dir, f)));
    return makeDb(meta, new Float32Array(rd(meta.files.feat)), new Float64Array(rd(meta.files.geo)),
                  new Int32Array(rd(meta.files.idx)));
  }

  /* ============================== RYSOWANIE PNG (Node) ============================== */
  // --- font 5x7 (A-Z, 0-9, kilka znaków); małe litery rysowane jako wielkie ---
  const FONT_SRC = {
    'A': '.###. #...# #...# ##### #...# #...# #...#', 'B': '####. #...# #...# ####. #...# #...# ####.',
    'C': '.###. #...# #.... #.... #.... #...# .###.', 'D': '####. #...# #...# #...# #...# #...# ####.',
    'E': '##### #.... #.... ####. #.... #.... #####', 'F': '##### #.... #.... ####. #.... #.... #....',
    'G': '.###. #...# #.... #.### #...# #...# .###.', 'H': '#...# #...# #...# ##### #...# #...# #...#',
    'I': '.###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.', 'J': '..### ...#. ...#. ...#. ...#. #..#. .##..',
    'K': '#...# #..#. #.#.. ##... #.#.. #..#. #...#', 'L': '#.... #.... #.... #.... #.... #.... #####',
    'M': '#...# ##.## #.#.# #.#.# #...# #...# #...#', 'N': '#...# ##..# #.#.# #..## #...# #...# #...#',
    'O': '.###. #...# #...# #...# #...# #...# .###.', 'P': '####. #...# #...# ####. #.... #.... #....',
    'Q': '.###. #...# #...# #...# #.#.# #..#. .##.#', 'R': '####. #...# #...# ####. #.#.. #..#. #...#',
    'S': '.#### #.... #.... .###. ....# ....# ####.', 'T': '##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..',
    'U': '#...# #...# #...# #...# #...# #...# .###.', 'V': '#...# #...# #...# #...# #...# .#.#. ..#..',
    'W': '#...# #...# #...# #.#.# #.#.# ##.## #...#', 'X': '#...# #...# .#.#. ..#.. .#.#. #...# #...#',
    'Y': '#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..', 'Z': '##### ....# ...#. ..#.. .#... #.... #####',
    '0': '.###. #...# #..## #.#.# ##..# #...# .###.', '1': '..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.',
    '2': '.###. #...# ....# ...#. ..#.. .#... #####', '3': '.###. #...# ....# ..##. ....# #...# .###.',
    '4': '...#. ..##. .#.#. #..#. ##### ...#. ...#.', '5': '##### #.... ####. ....# ....# #...# .###.',
    '6': '.###. #.... #.... ####. #...# #...# .###.', '7': '##### ....# ...#. ..#.. .#... .#... .#...',
    '8': '.###. #...# #...# .###. #...# #...# .###.', '9': '.###. #...# #...# .#### ....# ....# .###.',
    ' ': '..... ..... ..... ..... ..... ..... .....', '.': '..... ..... ..... ..... ..... .##.. .##..',
    ',': '..... ..... ..... ..... .##.. .##.. #....', '-': '..... ..... ..... ##### ..... ..... .....',
    '+': '..... ..#.. ..#.. ##### ..#.. ..#.. .....', '_': '..... ..... ..... ..... ..... ..... #####',
    '(': '...#. ..#.. .#... .#... .#... ..#.. ...#.', ')': '.#... ..#.. ...#. ...#. ...#. ..#.. .#...',
    '/': '....# ....# ...#. ..#.. .#... #.... #....', ':': '..... .##.. .##.. ..... .##.. .##.. .....',
    '%': '##..# ##..# ...#. ..#.. .#... #..## #..##', '=': '..... ..... ##### ..... ##### ..... .....',
    '|': '..#.. ..#.. ..#.. ..#.. ..#.. ..#.. ..#..',
    '#': '.#.#. .#.#. ##### .#.#. ##### .#.#. .#.#.', '?': '.###. #...# ....# ...#. ..#.. ..... ..#..',
  };
  const FONT = {};
  for (const ch in FONT_SRC) FONT[ch] = FONT_SRC[ch].split(' ');

  class Canvas {
    constructor(w, h, bg) {
      this.w = w; this.h = h; this.px = Buffer.alloc(w * h * 3);
      for (let i = 0; i < w * h; i++) { this.px[3 * i] = bg[0]; this.px[3 * i + 1] = bg[1]; this.px[3 * i + 2] = bg[2]; }
    }
    set(x, y, c) {
      if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
      const o = 3 * (y * this.w + x); this.px[o] = c[0]; this.px[o + 1] = c[1]; this.px[o + 2] = c[2];
    }
    rect(x, y, w, h, c) { for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) this.set(x + i, y + j, c); }
    frame(x, y, w, h, c) { this.rect(x, y, w, 1, c); this.rect(x, y + h - 1, w, 1, c); this.rect(x, y, 1, h, c); this.rect(x + w - 1, y, 1, h, c); }
    text(x, y, str, s, c) {
      let cx = x;
      for (const chRaw of String(str)) {
        const rows = FONT[chRaw.toUpperCase()] || FONT['?'];
        for (let r = 0; r < 7; r++) for (let q = 0; q < 5; q++) if (rows[r][q] === '#') this.rect(cx + q * s, y + r * s, s, s, c);
        cx += 6 * s;
      }
      return cx;
    }
    textWidth(str, s) { return String(str).length * 6 * s - s; }
  }

  // --- PNG (RGB 8-bit), bez zależności ---
  const CRC_T = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(buf) { let c = 0xFFFFFFFF; for (let i = 0; i < buf.length; i++) c = CRC_T[(c ^ buf[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
  function encodePng(cv) {
    const zlib = require('zlib');
    const raw = Buffer.alloc((cv.w * 3 + 1) * cv.h);
    for (let y = 0; y < cv.h; y++) { raw[y * (cv.w * 3 + 1)] = 0; cv.px.copy(raw, y * (cv.w * 3 + 1) + 1, y * cv.w * 3, (y + 1) * cv.w * 3); }
    const chunk = (type, data) => {
      const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
      const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
      const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
      return Buffer.concat([len, td, crc]);
    };
    const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(cv.w, 0); ihdr.writeUInt32BE(cv.h, 4); ihdr[8] = 8; ihdr[9] = 2;
    return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr),
                          chunk('IDAT', zlib.deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
  }

  // --- paleta "terrain" z matplotlib ---
  const TERRAIN = [[0, [0.2, 0.2, 0.6]], [0.15, [0, 0.6, 1]], [0.25, [0, 0.8, 0.4]], [0.5, [1, 1, 0.6]], [0.75, [0.5, 0.36, 0.33]], [1, [1, 1, 1]]];
  function terrain(t) {
    t = Math.min(1, Math.max(0, t));
    for (let i = 1; i < TERRAIN.length; i++) {
      if (t <= TERRAIN[i][0]) {
        const [t0, a] = TERRAIN[i - 1], [t1, b] = TERRAIN[i], f = (t - t0) / (t1 - t0);
        return [0, 1, 2].map((k) => Math.round(255 * (a[k] + (b[k] - a[k]) * f)));
      }
    }
    return [255, 255, 255];
  }

  // --- statystyki okna (to samo co _window_row w planet2vec.py) ---
  function windowStats(dem, gx, gy, n) {
    const N = n * n;
    let mn = Infinity;
    for (let i = 0; i < N; i++) if (isFinite(dem[i]) && dem[i] < mn) mn = dem[i];
    const rel = new Float32Array(N); let mx = 0, valid = 0;
    for (let i = 0; i < N; i++) {
      if (isFinite(dem[i])) { rel[i] = dem[i] - mn; valid++; if (rel[i] > mx) mx = rel[i]; } else rel[i] = NaN;
    }
    const hist = (vals, lo, hi, bins) => {
      const h = new Float64Array(bins); let tot = 0;
      for (const v of vals) { let b = Math.floor((v - lo) / (hi - lo) * bins); if (b >= bins) b = bins - 1; if (b < 0) b = 0; h[b]++; tot++; }
      return tot ? h.map((x) => x / tot) : h;
    };
    const relOk = [], slopes = [];
    for (let i = 0; i < N; i++) {
      if (isFinite(rel[i])) relOk.push(rel[i]);
      const m = Math.hypot(gx[i], gy[i]);
      if (isFinite(m)) slopes.push(Math.min(31, Math.max(0, Math.atan(m) * 180 / Math.PI)));
    }
    const elev = hist(relOk, 0, mx > 0 ? mx : 1, 30);
    const slope = hist(slopes, 0, 32, 16);
    const sl = slopes.slice().sort((a, b) => a - b);
    const medSlope = sl.length ? (sl.length % 2 ? sl[sl.length >> 1] : (sl[(sl.length >> 1) - 1] + sl[sl.length >> 1]) / 2) : 0;
    const orient = new Float64Array(12); let tot = 0;
    for (let i = 0; i < N; i++) {
      const m = Math.hypot(gx[i], gy[i]);
      if (!isFinite(m)) continue;
      let a = (Math.atan2(gy[i], gx[i]) * 180 / Math.PI + 90) % 180; if (a < 0) a += 180;
      orient[Math.min(11, Math.floor(a / 15))] += m; tot += m;
    }
    if (tot > 0) for (let b = 0; b < 12; b++) orient[b] /= tot;
    const sorted = Float64Array.from(orient).sort().reverse();
    return { rel, max: mx, validFrac: valid / N, elev, slope, medSlope, orient, sorted, orientMax: Math.max(...orient) };
  }

  const INK = [34, 34, 34], GREY = [120, 120, 120], AXIS = [160, 160, 160], BG = [255, 255, 255];
  function barChart(cv, x, y, w, h, vals, color, ymax) {
    const m = ymax || Math.max(...vals, 1e-9), bw = w / vals.length;
    cv.rect(x, y, w, h, [247, 247, 247]);
    vals.forEach((v, i) => {
      const bh = Math.round(h * Math.min(1, v / m)), x0 = Math.round(x + i * bw), x1 = Math.round(x + (i + 1) * bw) - 1;
      cv.rect(x0, y + h - bh, Math.max(1, x1 - x0), bh, color);
    });
    cv.rect(x, y + h, w, 1, AXIS); cv.rect(x - 1, y, 1, h + 1, AXIS);
  }
  const fmtNum = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));

  /** Panel jak w notebooku: mapa wysokości względnej + 4 histogramy. win = {dem, gx, gy} (n*n float32). */
  function renderPanel(win, n, title, subtitle) {
    const st = windowStats(win.dem, win.gx, win.gy, n);
    const W = 720, H = 372, cv = new Canvas(W, H, BG);
    cv.text(10, 8, title, 2, INK);
    cv.text(10, 26, subtitle, 1, GREY);

    const s = Math.max(1, Math.floor(272 / n)), mx0 = 14, my0 = 46, side = n * s;
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const v = st.rel[r * n + c];
      cv.rect(mx0 + c * s, my0 + r * s, s, s, isFinite(v) ? terrain(st.max > 0 ? v / st.max : 0) : [211, 211, 211]);
    }
    cv.frame(mx0 - 1, my0 - 1, side + 2, side + 2, AXIS);
    const cbx = mx0 + side + 10;
    for (let j = 0; j < side; j++) cv.rect(cbx, my0 + j, 12, 1, terrain(1 - j / (side - 1)));
    cv.frame(cbx - 1, my0 - 1, 14, side + 2, AXIS);
    cv.text(cbx + 18, my0, fmtNum(st.max) + ' M', 1, INK);
    cv.text(cbx + 18, my0 + side - 7, '0 M', 1, INK);
    cv.text(mx0, my0 + side + 8, 'REL ELEV (M ABOVE MIN)', 2, INK);
    cv.text(mx0, my0 + side + 28, 'VALID ' + Math.round(st.validFrac * 100) + '%', 1, GREY);

    const cells = [
      ['ELEV HIST', 'MAX ' + fmtNum(st.max) + ' M', st.elev, [46, 139, 87], null, '0', fmtNum(st.max) + ' M'],
      ['SLOPE HIST', 'MEDIAN ' + st.medSlope.toFixed(1) + ' DEG', st.slope, [205, 62, 62], null, '0', '32 DEG'],
      ['EDGE DIR', 'MOD 180 DEG', st.orient, [128, 80, 170], null, '0', '180 DEG'],
      ['SORTED DIR', 'MAX ' + st.orientMax.toFixed(2), st.sorted, [235, 140, 30], 1, '1', '12'],
    ];
    cells.forEach(([t, sub, vals, col, ymax, l0, l1], i) => {
      const x = 392 + (i % 2) * 164, y = 46 + Math.floor(i / 2) * 160;
      cv.text(x, y, t, 2, INK);
      cv.text(x, y + 16, sub, 1, GREY);
      barChart(cv, x, y + 28, 150, 102, Array.from(vals), col, ymax);
      cv.text(x, y + 136, l0, 1, GREY);
      cv.text(x + 150 - cv.textWidth(l1, 1), y + 136, l1, 1, GREY);
    });
    return encodePng(cv);
  }

  /* ============================== NODE CLI ============================== */
  function cliProfiles(featDir, outDir) {
    const fs = require('fs'), path = require('path');
    const files = fs.readdirSync(featDir).filter((f) => /^[A-Za-z0-9_]+\.json$/.test(f) && !f.endsWith('_win.json'));
    if (!files.length) throw new Error('No features/*.json in ' + featDir + ' - run: python planet2vec.py --features');
    const earthDir = path.join(outDir, 'earth');
    fs.mkdirSync(earthDir, { recursive: true });
    const indexPath = path.join(earthDir, 'index.json');
    let index = {};
    if (fs.existsSync(indexPath)) { try { index = JSON.parse(fs.readFileSync(indexPath, 'utf8')); } catch (e) { index = {}; } }

    for (const f of files) {
      const meta = JSON.parse(fs.readFileSync(path.join(featDir, f), 'utf8'));
      const body = meta.body, n = meta.n;
      const wins = JSON.parse(fs.readFileSync(path.join(featDir, meta.files.windows), 'utf8'));
      const data = new Float32Array(toAligned(fs.readFileSync(path.join(featDir, meta.files.windows_data))));
      if (data.length !== wins.length * 3 * n * n) throw new Error(body + ': rozmiar ' + meta.files.windows_data + ' nie pasuje do ' + meta.files.windows);
      fs.mkdirSync(path.join(outDir, body), { recursive: true });
      for (const k of Object.keys(index)) if (k.startsWith(body + '|')) delete index[k];      // odśwież wpisy tego ciała
      const km = (n * meta.pixel_m / 1000).toFixed(2);
      let nq = 0, ne = 0;
      wins.forEach((w, i) => {
        const o = i * 3 * n * n, A = n * n;
        const win = { dem: data.subarray(o, o + A), gx: data.subarray(o + A, o + 2 * A), gy: data.subarray(o + 2 * A, o + 3 * A) };
        const pos = 'LAT ' + w.lat.toFixed(4) + ' LON ' + w.lon.toFixed(4) + ' | ' + n + ' PX = ' + km + ' KM';
        if (w.kind === 'query') {
          const file = path.join(outDir, body, safe(w.source) + '_' + w.square + '.png');
          fs.writeFileSync(file, renderPanel(win, n, body + ' ' + w.source + ' #' + w.square, pos));
          nq++;
        } else {
          const name = earthFileName(body, w.chunk, w.row, w.col);
          fs.writeFileSync(path.join(earthDir, name), renderPanel(win, n, 'EARTH ' + w.chunk, pos + ' | ROW ' + w.row + ' COL ' + w.col));
          if (isFinite(w.lat) && isFinite(w.lon)) index[key4(body, w.lat, w.lon)] = name;
          ne++;
        }
      });
      console.log(body + ': ' + nq + ' PNG (cel) -> ' + path.join(outDir, body) + ', ' + ne + ' PNG (Ziemia) -> ' + earthDir);
    }
    fs.writeFileSync(indexPath, JSON.stringify(index, null, 1));
    console.log('index: ' + Object.keys(index).length + ' okien Ziemi -> ' + indexPath);
  }

  function cliSearch(featDir, body, ws) {
    const db = loadFeaturesFs(featDir, body);
    const w = defaultWeights(db);
    if (ws.length) BLOCKS.forEach((b, i) => { if (ws[i] !== undefined) w[b] = +ws[i]; });
    console.log('weights', w);
    const t0 = Date.now();
    const entries = computeEntries(db, w);
    console.log(entries.length + ' wpisów, ' + (Date.now() - t0) + ' ms');
    for (const e of entries) console.log(String(e.id).padStart(4), e.site.name.padEnd(34), e.earth.name.padEnd(18), e.earth.lat.toFixed(4).padStart(9), e.earth.lon.toFixed(4).padStart(10), 'd=' + e.dist.toFixed(3));
  }

  function cli(argv) {
    const opt = (name, def) => { const i = argv.indexOf('--' + name); return i >= 0 ? argv[i + 1] : def; };
    const cmd = argv[0];
    try {
      if (cmd === 'profiles') cliProfiles(opt('features', 'features'), opt('out', 'profiles'));
      else if (cmd === 'search') cliSearch(opt('features', 'features'), argv[1], argv.slice(2).filter((a) => !a.startsWith('--') && !isNaN(a)));
      else console.log('Użycie:\n  node recalc.js profiles [--features features] [--out profiles]\n  node recalc.js search <moon|mars> [w_elev_scale w_elev_shape w_slope w_orient]');
    } catch (e) { console.error(e.message); process.exit(1); }
  }

  return { BLOCKS, DEFAULT_WEIGHTS, loadFeatures, loadFeaturesFs, defaultWeights, computeEntries, validateWeights,
           queryProfilePath, earthFileName, key4, renderPanel, cli };
});
