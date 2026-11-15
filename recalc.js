/*
 * recalc.js  -  STUB (to be deleted)
 *
 * Browser-side weight recalculation and PNG profile generation were removed together with planet2vec.py's
 * --features export. Weights (BLOCK_W) are now applied in match.py, which writes output_areas.json.
 *
 * This file only keeps the API that index.html still references, so the page loads and nothing crashes:
 *   Planet2Vec.BLOCKS / DEFAULT_WEIGHTS / defaultWeights / validateWeights   (the weights panel is hidden)
 *   Planet2Vec.queryProfilePath / key4                                       (paths of existing profile pictures)
 *   Planet2Vec.loadFeatures / computeEntries                                 (reject with a clear message)
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Planet2Vec = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const BLOCKS = ['elev_scale', 'elev_shape', 'slope', 'orient'];
  const DEFAULT_WEIGHTS = { elev_scale: 2.0, elev_shape: 1.5, slope: 1.0, orient: 1.0 };   // = BLOCK_W in config.py
  const NOT_AVAILABLE = 'Recalculating in the browser is no longer available - change BLOCK_W in config.py and run match.py.';

  const safe = (s) => String(s).replace(/[^A-Za-z0-9._-]+/g, '_');
  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  const key4 = (body, lat, lon) => body + '|' + r4(lat).toFixed(4) + '|' + r4(lon).toFixed(4);
  const queryProfilePath = (body, source, square) => `profiles/${body}/${safe(source)}_${square}.png`;
  const earthFileName = (body, chunk, row, col) => `${body}_${safe(chunk)}_${row}_${col}.png`;

  function defaultWeights() { return { ...DEFAULT_WEIGHTS }; }

  function validateWeights(w) {
    let sum = 0;
    for (const b of BLOCKS) {
      const v = +w[b];
      if (!isFinite(v) || v < 0) throw new Error('Weight "' + b + '" must be a number >= 0.');
      sum += v;
    }
    if (sum <= 0) throw new Error('At least one weight must be greater than 0.');
  }

  const loadFeatures = () => Promise.reject(new Error(NOT_AVAILABLE));
  const computeEntries = () => { throw new Error(NOT_AVAILABLE); };

  return { BLOCKS, DEFAULT_WEIGHTS, defaultWeights, validateWeights, queryProfilePath, earthFileName, key4,
           loadFeatures, computeEntries };
});
