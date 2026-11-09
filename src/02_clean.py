"""
src/02_clean.py
Phase 2 — Data Cleaning

Handles:
  - inf / -inf values (BRBA dust contamination)
  - LOW quality months (< 2 images)
  - Outlier detection per zone per index
  - Linear interpolation for short gaps (≤ 2 months)
  - Quality report generation
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def clean(input_csv, output_csv, report_csv):

    print(f"📂 Reading: {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=['datadate'])
    df = df.sort_values('datadate').reset_index(drop=True)

    print(f"   Shape: {df.shape}")

    # ── Identify numeric index columns ───────────────────────
    index_prefixes = ['ndbi_mean','ndbi_max','bu_mean','bu_max',
                      'brba_mean','brba_max','baei_mean','baei_max',
                      'vbish_mean','vbish_max']

    index_cols = [c for c in df.columns
                  if any(c.startswith(p) for p in index_prefixes)]

    print(f"   Index columns: {len(index_cols)}")

    quality_report = []

    for col in index_cols:
        series = df[col].copy()
        n_total = len(series)

        # Count issues before cleaning
        n_inf    = np.isinf(series.replace([np.inf, -np.inf], np.nan)
                             .fillna(np.nan)).sum()
        n_inf    = series.isin([np.inf, -np.inf]).sum()
        n_nan_orig = series.isna().sum()

        # Step 1: Replace inf/-inf with NaN
        series = series.replace([np.inf, -np.inf], np.nan)

        # Step 2: IQR-based outlier detection (per column)
        q1 = series.quantile(0.05)
        q3 = series.quantile(0.95)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        n_outliers = series[(series < lower) | (series > upper)].count()
        series[(series < lower) | (series > upper)] = np.nan

        # Step 3: Interpolate gaps ≤ 2 consecutive months
        # (longer gaps left as NaN — too risky to interpolate)
        series = series.interpolate(
            method='linear', limit=2, limit_direction='both'
        )

        n_nan_after = series.isna().sum()
        df[col] = series

        quality_report.append({
            'column'       : col,
            'n_total'      : n_total,
            'n_inf'        : n_inf,
            'n_nan_original': n_nan_orig,
            'n_outliers_clipped': n_outliers,
            'n_nan_after_interp': n_nan_after,
            'pct_valid_final'   : round(
                100 * (1 - n_nan_after / n_total), 1
            ),
        })

    # ── Handle LOW quality months ─────────────────────────────
    # Columns flagged quality_flag_<zone> = 'LOW' → mark index values
    # as uncertain (we don't delete, just flag so downstream knows)
    quality_flag_cols = [c for c in df.columns if c.startswith('quality_flag_')]
    for qcol in quality_flag_cols:
        zone = qcol.replace('quality_flag_', '')
        zone_index_cols = [c for c in index_cols if c.endswith(f'_{zone}')]
        low_mask = df[qcol] == 'LOW'
        if low_mask.sum() > 0:
            print(f"  ⚠  {zone}: {low_mask.sum()} LOW-quality months")
            # We keep values but add a _reliable column for filtering
            for ic in zone_index_cols:
                reliable_col = ic.replace(f'_{zone}', f'_reliable_{zone}')
                df[reliable_col] = ~low_mask

    # ── Save outputs ─────────────────────────────────────────
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Clean data saved: {output_csv}")
    print(f"   Shape: {df.shape}")

    report_df = pd.DataFrame(quality_report)
    report_df.to_csv(report_csv, index=False)
    print(f"✓ Quality report: {report_csv}")

    # Quick summary
    print("\n── Quality Summary ───────────────────────────────")
    print(report_df[['column','n_inf','n_outliers_clipped',
                     'pct_valid_final']].to_string(index=False))

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    clean(args.input, args.output, args.report)