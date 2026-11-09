"""
src/01_gee_export.py
Phase 1 — Google Earth Engine Export

Pulls Sentinel-2 built-up indices for each zone in a shapefile,
month by month, with Mauritania-specific cloud+dust masking.
"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


# ─────────────────────────────────────────────
# 0. GEE INITIALIZATION
# ─────────────────────────────────────────────

def init_gee():
    project_id = os.environ.get("GEE_PROJECT")
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✓ GEE initialized")
    except Exception as e:
        print(f"✗ GEE init failed: {e}")
        print("  Inside container, run:")
        print("  docker compose run --rm gee_export earthengine authenticate")
        raise


# ─────────────────────────────────────────────
# 1. CLOUD + DUST MASKING
# ─────────────────────────────────────────────

def mask_s2_clouds_and_dust(image):
    """
    Multi-layer mask tailored for Mauritania:
      Layer 1 — QA60 bitmask (opaque clouds + cirrus)
      Layer 2 — SCL scene classification (shadows, snow, high-prob cloud)
      Layer 3 — B1/B2 ratio (dust/aerosol haze — harmattan specific)
      Layer 4 — AOT band (aerosol optical thickness > 0.6 = dust storm)
    """
    # Layer 1: QA60
    qa = image.select('QA60')
    qa_mask = (qa.bitwiseAnd(1 << 10).eq(0)
                 .And(qa.bitwiseAnd(1 << 11).eq(0)))

    # Layer 2: SCL — keep only clear pixels (4=vegetation, 5=bare soil,
    # 6=water, 7=unclassified) — exclude clouds, shadows, saturated
    scl = image.select('SCL')
    scl_mask = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
                   .And(scl.neq(8)).And(scl.neq(9))
                   .And(scl.neq(10)).And(scl.neq(11)))

    # Layer 3: Dust/Haze — B1 (aerosol 443nm) / B2 (blue 490nm)
    # Ratio > 0.85 = heavy aerosol (harmattan dust)
    b1 = image.select('B1').toFloat()
    b2 = image.select('B2').toFloat()
    dust_mask = b1.divide(b2.add(1e-6)).lt(0.85)

    # Layer 4: AOT (Aerosol Optical Thickness), scaled [0,1]
    aot = image.select('AOT').toFloat().divide(1000)
    aot_mask = aot.lt(0.6)

    combined = qa_mask.And(scl_mask).And(dust_mask).And(aot_mask)

    return (image.updateMask(combined)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))


# ─────────────────────────────────────────────
# 2. INDEX COMPUTATION
# ─────────────────────────────────────────────

def compute_indices(image):
    """
    Compute built-up indices from Sentinel-2 SR bands.

    Applies three masks BEFORE index computation (critical for Saharan cities):
      1. Water/Sebkha mask  — NDWI > 0.0  (lagoon pixels inflate BAEI/NDBI)
      2. Sand mask          — SWIR2/SWIR1 > 0.95 (dunes similar to built-up in SWIR1)
      3. Vegetation mask    — NDVI > 0.15 (rare in Mauritania but present near Ksar)

    Also adds bu_fraction: binary band (1 where NDBI > 0.05).
    Mean of bu_fraction = % valid pixels classified as built-up.
    This is more robust than mean index value for time series analysis.
    """
    blue  = image.select('B2')
    green = image.select('B3')
    red   = image.select('B4')
    nir   = image.select('B8')
    swir1 = image.select('B11')
    swir2 = image.select('B12')
    eps   = 1e-10

    # ── Land cover masks ────────────────────────────────────
    ndwi       = green.subtract(nir).divide(green.add(nir).add(eps))
    is_water   = ndwi.gt(0.0)

    sand_ratio = swir2.divide(swir1.add(eps))
    is_sand    = sand_ratio.gt(0.95)

    ndvi_raw   = nir.subtract(red).divide(nir.add(red).add(eps))
    is_veg     = ndvi_raw.gt(0.15)

    keep  = is_water.Not().And(is_sand.Not()).And(is_veg.Not())
    image = image.updateMask(keep)

    # Re-select bands after masking
    blue  = image.select('B2')
    green = image.select('B3')
    red   = image.select('B4')
    nir   = image.select('B8')
    swir1 = image.select('B11')
    swir2 = image.select('B12')

    # ── Index computation ────────────────────────────────────
    ndvi  = nir.subtract(red).divide(nir.add(red).add(eps))
    ndbi  = swir1.subtract(nir).divide(swir1.add(nir).add(eps)).rename('ndbi')
    bu    = ndbi.subtract(ndvi).rename('bu')
    brba  = red.divide(swir1.add(eps)).rename('brba')
    baei  = red.add(0.3).divide(green.add(swir1).add(eps)).rename('baei')
    vbish = blue.divide(swir2.add(eps)).rename('vbish')

    # Built-up fraction: primary metric for time series
    bu_fraction = ndbi.gt(0.05).rename('bu_fraction')

    return image.addBands([ndbi, bu, brba, baei, vbish, bu_fraction])


# ─────────────────────────────────────────────
# 3. MONTHLY COMPOSITE
# ─────────────────────────────────────────────

def build_monthly_composite(year, month, region, max_cloud_pct=30):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, 'month')

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterDate(start, end)
             .filterBounds(region)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
             .map(mask_s2_clouds_and_dust)
             .map(compute_indices))

    n_images  = col.size()
    composite = col.median()  # median is more robust than mean for this use case
    return composite, n_images


# ─────────────────────────────────────────────
# 4. ZONAL STATISTICS
# ─────────────────────────────────────────────

def extract_zonal_stats(composite, zone_geom, zone_name,
                        year, month, n_images):
    bands = ['ndbi', 'bu', 'brba', 'baei', 'vbish']
    scale = 10  # Sentinel-2 10m resolution

    try:
        means = composite.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=zone_geom,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        maxes = composite.select(bands).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=zone_geom,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        # bu_fraction mean = % valid pixels classified as built-up
        frac = composite.select('bu_fraction').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=zone_geom,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        # Also compute pixel count to flag zones with few valid observations
        count = composite.select('ndbi').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=zone_geom,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        n_img    = n_images.getInfo()
        n_pixels = count.get('ndbi', None)

    except Exception as e:
        print(f"\n  ✗ {zone_name} {year}-{month:02d}: {e}")
        return None

    # End-of-month date
    last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)

    row = {
        'datadate' : last_day.strftime('%Y-%m-%d'),
        'zone'     : zone_name,
        'n_images' : n_img,
        'n_pixels' : n_pixels,
        'quality_flag': 'LOW' if n_img < 2 else 'OK',
        'bu_fraction' : frac.get('bu_fraction'),  # primary metric
    }

    for band in bands:
        row[f'{band}_mean'] = means.get(band)
        row[f'{band}_max']  = maxes.get(band)

    return row


# ─────────────────────────────────────────────
# 5. SHAPEFILE LOADING
# ─────────────────────────────────────────────

def load_zones(shp_path, zone_field):
    gdf = gpd.read_file(shp_path)

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
        print(f"  Reprojected CRS → WGS84")

    # Simplify geometries slightly to speed up GEE requests
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)

    print(f"  {len(gdf)} zones found in '{zone_field}' field")

    zones = {}
    for _, row in gdf.iterrows():
        name = str(row[zone_field]).strip().replace(' ', '_').replace("'", "")
        geom_json = json.loads(
            gpd.GeoSeries([row.geometry], crs='EPSG:4326').to_json()
        )
        geom_ee = ee.Geometry(geom_json['features'][0]['geometry'])
        zones[name] = geom_ee

    return zones


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def run(shapefile, zone_field, start, end, output_dir,
        max_cloud_pct=30):

    init_gee()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loading: {shapefile}")
    zones = load_zones(shapefile, zone_field)

    # Build month list
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')
    months   = []
    cur = start_dt
    while cur < end_dt:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)

    total = len(months) * len(zones)
    print(f"📅 {len(months)} months × {len(zones)} zones = {total} requests\n")

    all_rows = []

    for zone_name, zone_geom in zones.items():
        print(f"\n🗺  Zone: {zone_name}")
        zone_rows = []

        for year, month in tqdm(months, desc=f"  {zone_name}", unit="month"):
            composite, n_images = build_monthly_composite(
                year, month, zone_geom, max_cloud_pct
            )
            row = extract_zonal_stats(
                composite, zone_geom, zone_name, year, month, n_images
            )
            if row:
                zone_rows.append(row)

            time.sleep(0.3)  # polite rate limiting

        all_rows.extend(zone_rows)

        # Checkpoint: save per-zone CSV immediately
        zone_df  = pd.DataFrame(zone_rows)
        zone_csv = output_path / f"raw_{zone_name}.csv"
        zone_df.to_csv(zone_csv, index=False)
        print(f"  ✓ Saved {zone_csv} ({len(zone_rows)} rows)")

    # ── Wide format combined output ──────────────────────────
    print("\n📊 Building combined wide CSV...")
    full_df = pd.DataFrame(all_rows)

    value_cols = ['bu_fraction',
                  'ndbi_mean','ndbi_max','bu_mean','bu_max',
                  'brba_mean','brba_max','baei_mean','baei_max',
                  'vbish_mean','vbish_max','n_images','quality_flag']

    wide = full_df.pivot(index='datadate', columns='zone', values=value_cols)
    wide.columns = [f"{v}_{z}" for v, z in wide.columns]
    wide = wide.reset_index().sort_values('datadate')

    out_csv = output_path / "builtup_indices_monthly_NEW.csv"
    wide.to_csv(out_csv, index=False)
    print(f"✓ Combined output: {out_csv}")
    print(f"  Shape: {wide.shape}")

    return wide


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapefile',  required=True)
    parser.add_argument('--zone_field', required=True)
    parser.add_argument('--start',      default='2015-01-01')
    parser.add_argument('--end',        default='2026-01-01')
    parser.add_argument('--output',     default='/data/raw/')
    parser.add_argument('--max_cloud',  type=int, default=30)
    args = parser.parse_args()

    run(args.shapefile, args.zone_field,
        args.start, args.end, args.output, args.max_cloud)