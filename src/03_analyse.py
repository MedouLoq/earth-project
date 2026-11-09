"""
src/03_analyse.py
Phase 3 — STL Decomposition + Lag Correlation Analysis

Replicates what InsightEx showed in slides 9 and 22-31,
but at moughataa level and with full methodological transparency.

Outputs:
  - STL decomposition plots per zone per index
  - Lag correlation tables (vs economic variables if provided)
  - Summary CSV of best correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import STL
from scipy import stats
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Color palette (consistent with InsightEx slides) ─────────
COLORS = {
    'original' : '#ADB5C8',
    'trend'    : '#E63946',
    'seasonal' : '#457B9D',
    'residual' : '#F4A261',
}


# ─────────────────────────────────────────────
# 1. STL DECOMPOSITION
# ─────────────────────────────────────────────

def run_stl(series: pd.Series, period: int = 12) -> dict:
    """
    Run STL decomposition on a monthly time series.
    period=12 for monthly data (annual seasonality).
    Returns dict with trend, seasonal, resid components.
    """
    # STL requires no NaN — forward fill short gaps
    s = series.ffill().bfill()

    stl = STL(s, period=period, robust=True)
    result = stl.fit()

    return {
        'trend'    : pd.Series(result.trend,    index=series.index),
        'seasonal' : pd.Series(result.seasonal, index=series.index),
        'resid'    : pd.Series(result.resid,    index=series.index),
    }


def plot_stl(series: pd.Series, components: dict,
             title: str, output_path: Path):
    """
    4-panel STL plot matching the style from slide 9.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 1, hspace=0.05)

    panels = [
        ('Observed',  series,                   COLORS['original']),
        ('Trend',     components['trend'],       COLORS['trend']),
        ('Seasonal',  components['seasonal'],    COLORS['seasonal']),
        ('Residual',  components['resid'],       COLORS['residual']),
    ]

    axes = []
    for i, (label, data, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i], sharex=axes[0] if axes else None)
        axes.append(ax)

        if label == 'Residual':
            ax.scatter(data.index, data.values,
                       color=color, s=15, alpha=0.7, zorder=3)
            ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        else:
            ax.plot(data.index, data.values,
                    color=color, lw=1.4, alpha=0.9)
            if label == 'Observed':
                ax.plot(series.index, series.values,
                        color=COLORS['original'], lw=0.8, alpha=0.4)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis='y', alpha=0.3, ls=':')
        ax.tick_params(axis='x',
                       labelbottom=(i == len(panels) - 1),
                       labelsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# 2. LAG CORRELATION
# ─────────────────────────────────────────────

def lag_correlation(x: pd.Series, y: pd.Series,
                    max_lag: int = 24) -> pd.DataFrame:
    """
    Compute Pearson correlation between x and lagged y.
    Positive lag = x leads y (x predicts y).
    Negative lag = y leads x.

    Returns DataFrame with columns: lag, correlation, p_value, n_obs
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_aligned = x.iloc[lag:]
            y_aligned = y.iloc[:len(x) - lag]
        else:
            x_aligned = x.iloc[:len(x) + lag]
            y_aligned = y.iloc[-lag:]

        # Align on index
        combined = pd.concat([x_aligned.reset_index(drop=True),
                               y_aligned.reset_index(drop=True)],
                              axis=1).dropna()

        if len(combined) < 10:
            continue

        r, p = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
        results.append({
            'lag'        : lag,
            'correlation': round(r, 4),
            'p_value'    : round(p, 4),
            'n_obs'      : len(combined),
            'significant': p < 0.05,
        })

    return pd.DataFrame(results)


def plot_ccf(ccf_df: pd.DataFrame, title: str, output_path: Path):
    """
    Cross-correlation function plot matching slide 16 style.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#E63946' if sig else '#ADB5C8'
              for sig in ccf_df['significant']]

    ax.bar(ccf_df['lag'], ccf_df['correlation'],
           color=colors, width=0.8, alpha=0.85)

    # 95% confidence bands (approximate: ±1.96/sqrt(n))
    n = ccf_df['n_obs'].median()
    ci = 1.96 / np.sqrt(n)
    ax.axhline( ci, color='red', ls='--', lw=1, alpha=0.6,
                label=f'IC 95% (±{ci:.3f})')
    ax.axhline(-ci, color='red', ls='--', lw=1, alpha=0.6)
    ax.axhline(0, color='black', lw=0.8)

    # Mark best lag
    best = ccf_df.loc[ccf_df['correlation'].abs().idxmax()]
    ax.axvline(best['lag'], color='green', ls=':', lw=1.5,
               label=f"Best lag = {int(best['lag'])} "
                     f"(r={best['correlation']:.3f})")

    ax.set_xlabel('Lag (mois)', fontsize=10)
    ax.set_ylabel('Corrélation', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, ls=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────

def run(input_csv, output_dir, economic_csv=None, max_lag=24):

    output_path = Path(output_dir)
    plots_path  = output_path / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Reading: {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=['datadate'])
    df = df.set_index('datadate').sort_index()

    # ── Detect zones and indices ──────────────────────────────
    # Columns follow pattern: {index}_{stat}_{zone}
    # e.g. baei_mean_Nouakchott-nord
    mean_cols = [c for c in df.columns if '_mean_' in c]
    zones     = list(set(c.split('_mean_')[1] for c in mean_cols
                         if not c.startswith('quality')))
    indices   = ['ndbi', 'bu', 'brba', 'baei', 'vbish']

    print(f"   Zones  : {zones}")
    print(f"   Indices: {indices}")

    # ── Load economic data if provided ────────────────────────
    econ_df = None
    if economic_csv:
        econ_df = pd.read_csv(economic_csv, parse_dates=['date'])
        econ_df = econ_df.set_index('date').sort_index()
        print(f"   Economic variables: {econ_df.columns.tolist()}")

    all_correlations = []

    # ── STL decomposition per zone per index ──────────────────
    print(f"\n🔬 Running STL decompositions...")
    stl_results = {}

    for zone in zones:
        for idx in indices:
            col = f'{idx}_mean_{zone}'
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 24:
                print(f"  ⚠  Skipping {col}: only {len(series)} obs")
                continue

            try:
                components = run_stl(series)
                stl_results[col] = components

                plot_stl(
                    series, components,
                    title=f'{idx.upper()} — {zone}',
                    output_path=plots_path / f'stl_{idx}_{zone}.png'
                )
                print(f"  ✓ STL: {col}")

            except Exception as e:
                print(f"  ✗ STL failed {col}: {e}")

    # ── Lag correlations vs economic variables ─────────────────
    if econ_df is not None:
        print(f"\n📈 Computing lag correlations vs economic variables...")

        for econ_var in econ_df.columns:
            y = econ_df[econ_var].dropna()

            for col, components in stl_results.items():
                # Correlate using the TREND component (removes seasonality noise)
                x_trend = components['trend']

                # Align monthly dates
                x_m = x_trend.resample('ME').mean()
                y_m = y.resample('ME').mean()

                # Common date range
                common = x_m.index.intersection(y_m.index)
                if len(common) < 12:
                    continue

                ccf = lag_correlation(
                    x_m.loc[common], y_m.loc[common], max_lag=max_lag
                )

                best = ccf.loc[ccf['correlation'].abs().idxmax()]
                all_correlations.append({
                    'satellite_var' : col,
                    'economic_var'  : econ_var,
                    'best_lag_months': int(best['lag']),
                    'best_correlation': best['correlation'],
                    'p_value'       : best['p_value'],
                    'significant'   : best['significant'],
                })

                # Save CCF plot for strong correlations only
                if abs(best['correlation']) > 0.3:
                    plot_ccf(
                        ccf,
                        title=f"CCF: {col} → {econ_var}",
                        output_path=plots_path / f'ccf_{col}_{econ_var}.png'
                    )

        # ── Summary table ─────────────────────────────────────
        corr_df = pd.DataFrame(all_correlations)
        corr_df = corr_df.sort_values('best_correlation',
                                       key=abs, ascending=False)

        corr_csv = output_path / 'lag_correlations.csv'
        corr_df.to_csv(corr_csv, index=False)
        print(f"\n✓ Correlation table: {corr_csv}")

        print("\n── Top 15 Correlations ───────────────────────────────")
        print(corr_df.head(15)[
            ['satellite_var','economic_var',
             'best_lag_months','best_correlation','significant']
        ].to_string(index=False))

    print(f"\n✓ All plots saved to: {plots_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',        required=True,
                        help='Cleaned built-up indices CSV')
    parser.add_argument('--output',       required=True,
                        help='Output directory for plots and tables')
    parser.add_argument('--economic_csv', default=None,
                        help='Optional: CSV with economic variables '
                             '(date column + one column per variable)')
    parser.add_argument('--max_lag',      type=int, default=24,
                        help='Maximum lag in months to test (default: 24)')
    args = parser.parse_args()

    run(args.input, args.output, args.economic_csv, args.max_lag)