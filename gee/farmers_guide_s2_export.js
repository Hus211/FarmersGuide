// ============================================================================
// FARMERS GUIDE — Sentinel-2 export pipeline (Google Earth Engine)
// ----------------------------------------------------------------------------
// AOIs:    Lusaka Province + 3 sub-AOIs (Chilanga, Kafue, Chongwe)
// Seasons: 2022/23, 2023/24, 2024/25 maize seasons (Nov–May, Zambia AER IIa)
// Bands:   Reflectance (B2,B3,B4,B5,B6,B7,B8,B11,B12) + NDVI, EVI, NDRE, GCI, NDWI
// Output:  10-day median composites, GeoTIFF, EPSG:32735 (UTM 35S), 10m
//
// HOW TO RUN:
//   1. Open https://code.earthengine.google.com
//   2. Paste this entire script into a new repository file
//   3. Hit "Run" — verify the map preview and the printed collection sizes
//   4. Set DRY_RUN = false
//   5. Hit "Run" again — exports queue in the "Tasks" panel (right side)
//   6. Click "Run" on each task to start it (or use the Python ee API to batch)
//
// NOTES:
//   - First validate with one season × one sub-AOI before bulk-running
//   - 21 composites × 3 AOIs × 3 seasons = ~189 export tasks at full scale
//   - Lusaka Province export is heavy (~2.5GB/composite) — run last, on-demand
// ============================================================================

// ---------- 0. Toggles --------------------------------------------------------
var DRY_RUN = true;              // Set false to actually queue exports
var EXPORT_DRIVE_FOLDER = 'farmers_guide_s2';

// ---------- 1. AOIs -----------------------------------------------------------
// Lusaka Province bounding box — covers all districts in the province
var lusakaProvince = ee.Geometry.Rectangle([27.50, -16.50, 29.50, -15.00]);

// Sub-AOIs — tighter boxes for training data and validation
var aois = {
  'chilanga': ee.Geometry.Rectangle([28.15, -15.65, 28.40, -15.45]),
  'kafue':    ee.Geometry.Rectangle([28.05, -15.95, 28.35, -15.60]),
  'chongwe':  ee.Geometry.Rectangle([28.50, -15.55, 29.05, -15.10]),
  'lusaka_province': lusakaProvince
};

// ---------- 2. Seasons --------------------------------------------------------
// Zambia maize: planting Nov, harvest Apr–May. Window stretched to capture
// land prep (Oct) and post-harvest senescence (early Jun).
var seasons = {
  '2022_23': {start: '2022-10-15', end: '2023-06-15'},
  '2023_24': {start: '2023-10-15', end: '2024-06-15'},
  '2024_25': {start: '2024-10-15', end: '2025-06-15'}
};

// ---------- 3. Cloud masking — Cloud Score+ (recommended over QA60) ----------
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var CS_BAND = 'cs';                  // 'cs' panchromatic, 'cs_cdf' multispectral
var CLEAR_THRESHOLD = 0.60;          // 0.5–0.65 typical; raise for stricter mask

function maskClouds(img) {
  return img.updateMask(img.select(CS_BAND).gte(CLEAR_THRESHOLD));
}

// ---------- 4. Vegetation indices --------------------------------------------
function addIndices(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndre = img.normalizedDifference(['B8', 'B5']).rename('NDRE');
  // Gao's NDWI for canopy water content (B8 vs SWIR-1)
  var ndwi = img.normalizedDifference(['B8', 'B11']).rename('NDWI');
  var evi = img.expression(
    '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
    {NIR: img.select('B8'), RED: img.select('B4'), BLUE: img.select('B2')}
  ).rename('EVI');
  var gci = img.expression(
    '(NIR / GREEN) - 1',
    {NIR: img.select('B8'), GREEN: img.select('B3')}
  ).rename('GCI');
  return img.addBands([ndvi, ndre, ndwi, evi, gci]);
}

// ---------- 5. Build season collection ---------------------------------------
function buildSeasonCollection(seasonKey) {
  var s = seasons[seasonKey];
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(s.start, s.end)
    .filterBounds(lusakaProvince)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .linkCollection(csPlus, [CS_BAND])
    .map(maskClouds)
    .map(addIndices);
  return s2;
}

// ---------- 6. 10-day median composites --------------------------------------
function tenDayComposites(collection, seasonKey) {
  var s = seasons[seasonKey];
  var startDate = ee.Date(s.start);
  var endDate = ee.Date(s.end);
  var nWindows = endDate.difference(startDate, 'day').divide(10).ceil();
  var indices = ee.List.sequence(0, nWindows.subtract(1));

  var composites = ee.ImageCollection.fromImages(
    indices.map(function(i) {
      var winStart = startDate.advance(ee.Number(i).multiply(10), 'day');
      var winEnd = winStart.advance(10, 'day');
      var composite = collection.filterDate(winStart, winEnd).median();
      return composite
        .set('system:time_start', winStart.millis())
        .set('window_start', winStart.format('YYYY-MM-dd'))
        .set('window_index', i)
        .set('season', seasonKey);
    })
  );

  // Drop fully empty composites (gaps with zero clear obs)
  return composites.map(function(img) {
    return img.set('has_data', img.bandNames().size().gt(0));
  });
}

// ---------- 7. Sanity prints (always run) ------------------------------------
print('=== FARMERS GUIDE / S2 EXPORT — DRY_RUN:', DRY_RUN, '===');

Object.keys(seasons).forEach(function(seasonKey) {
  var coll = buildSeasonCollection(seasonKey);
  var composites = tenDayComposites(coll, seasonKey);
  print(seasonKey + ' — raw scenes:', coll.size());
  print(seasonKey + ' — 10-day windows:', composites.size());
});

// ---------- 8. Map preview — Chilanga, most recent season --------------------
Map.centerObject(aois.chilanga, 11);
Map.addLayer(aois.chilanga, {color: 'FF0000'}, 'Chilanga AOI', false);
Map.addLayer(aois.kafue, {color: 'FFAA00'}, 'Kafue AOI', false);
Map.addLayer(aois.chongwe, {color: '00AAFF'}, 'Chongwe AOI', false);
Map.addLayer(aois.lusaka_province, {color: 'AAAAAA'}, 'Lusaka Province', false);

var previewSeason = buildSeasonCollection('2024_25');
var previewMid = previewSeason
  .filterDate('2025-02-01', '2025-02-28')
  .median()
  .clip(aois.chilanga);

Map.addLayer(
  previewMid, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
  'Chilanga RGB Feb 2025'
);
Map.addLayer(
  previewMid, {bands: ['NDVI'], min: 0.0, max: 0.9, palette: ['white', 'green']},
  'Chilanga NDVI Feb 2025'
);

// ---------- 9. NDVI time-series chart at a sample point ---------------------
var samplePoint = ee.Geometry.Point([28.27, -15.55]); // Chilanga area
var ndviSeries = ui.Chart.image.series({
  imageCollection: tenDayComposites(buildSeasonCollection('2024_25'), '2024_25')
    .select('NDVI'),
  region: samplePoint,
  reducer: ee.Reducer.mean(),
  scale: 10
}).setOptions({
  title: 'NDVI 10-day composites — Chilanga sample, 2024/25',
  vAxis: {title: 'NDVI', minValue: 0, maxValue: 1},
  hAxis: {title: 'Date', format: 'MMM yyyy'},
  lineWidth: 2,
  pointSize: 4
});
print(ndviSeries);

// ---------- 10. Export driver -------------------------------------------------
var BANDS_OUT = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12',
                 'NDVI', 'EVI', 'NDRE', 'GCI', 'NDWI'];

function queueExports(seasonKey, aoiKey) {
  var aoi = aois[aoiKey];
  var coll = buildSeasonCollection(seasonKey);
  var composites = tenDayComposites(coll, seasonKey);

  // Pull window count client-side once
  var n = composites.size().getInfo();
  var imgList = composites.toList(n);

  for (var i = 0; i < n; i++) {
    var img = ee.Image(imgList.get(i));
    var dateStr = ee.Date(img.get('system:time_start'))
      .format('YYYYMMdd').getInfo();
    var name = ['fg', aoiKey, seasonKey, dateStr].join('_');

    Export.image.toDrive({
      image: img.select(BANDS_OUT).clip(aoi).toFloat(),
      description: name,
      folder: EXPORT_DRIVE_FOLDER,
      fileNamePrefix: name,
      region: aoi,
      scale: 10,
      crs: 'EPSG:32735',
      maxPixels: 1e10,
      fileFormat: 'GeoTIFF',
      formatOptions: {cloudOptimized: true}
    });
  }
  print('Queued ' + n + ' exports for ' + aoiKey + ' / ' + seasonKey);
}

// ---------- 11. Export queue --------------------------------------------------
// Strategy:
//   Step A — uncomment ONE line, set DRY_RUN=false, validate one task end-to-end
//   Step B — uncomment all sub-AOI × season combos, run as a batch
//   Step C — only after sub-AOIs are clean, run the full Lusaka Province export
//            (this will be ~21 large GeoTIFFs per season — gigabytes each)

if (!DRY_RUN) {
  // --- Step A: Validation export (start here) ---
  queueExports('2024_25', 'chilanga');

  // --- Step B: Full sub-AOI matrix (uncomment when ready) ---
  // queueExports('2024_25', 'kafue');
  // queueExports('2024_25', 'chongwe');
  // queueExports('2023_24', 'chilanga');
  // queueExports('2023_24', 'kafue');
  // queueExports('2023_24', 'chongwe');
  // queueExports('2022_23', 'chilanga');
  // queueExports('2022_23', 'kafue');
  // queueExports('2022_23', 'chongwe');

  // --- Step C: Province-wide (heavy — only after sub-AOIs validate) ---
  // queueExports('2024_25', 'lusaka_province');
  // queueExports('2023_24', 'lusaka_province');
  // queueExports('2022_23', 'lusaka_province');
}

// ============================================================================
// END
// ----------------------------------------------------------------------------
// Filename convention: fg_<aoi>_<season>_<YYYYMMDD>.tif
//   e.g.  fg_chilanga_2024_25_20250201.tif
// Band order in output: B2,B3,B4,B5,B6,B7,B8,B11,B12,NDVI,EVI,NDRE,GCI,NDWI
// Downstream: ingest into HDF5 cubes via your existing thesis pipeline,
//   stack temporal axis as (T, H, W, 14) tensors for the satellite CNN.
// ============================================================================
