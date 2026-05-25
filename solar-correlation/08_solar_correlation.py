"""
Script 08: Solar Event Correlation (Sunrise/Sunset vs Peak Activity)
=====================================================================
Computes Pearson correlation between daily peak vocal activity time and
sunrise/sunset times for each species.

Method (from paper Section 3.2.6):
  - Morning window: hours 5-9, extract peak hour per day
  - Evening window: hours 17-20, extract peak hour per day
  - Compute sunrise/sunset times using astronomical calculations
  - Pearson r between peak_hour and sunrise/sunset decimal time

Paper Reference: Section 3.2.6, Table 6 (Black Drongo Pearson = 0.83)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import datetime
import os

try:
    import pytz
    from astral import LocationInfo
    from astral.sun import sunrise, sunset
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False
    print("WARNING: 'astral' package not installed. Install with: pip install astral pytz")
    print("Using fallback sunrise/sunset estimation.")

# =============================================================================
# CONFIGURATION (imported from 00_config.py)
# =============================================================================
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

LATITUDE = config.LATITUDE
LONGITUDE = config.LONGITUDE
TIMEZONE_STR = config.TIMEZONE_STR
MIN_DAYS = 5  # Minimum days with >10 detections for a species to be included

# (Morning/Evening window splitting is defined in the paper Section 3.2.6
# but NOT implemented in notebook cell 31. Kept simple to match notebook.)


def run_solar_correlation(input_csv, output_dir):
    """Core logic: compute solar event correlations from a detections CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # LOAD DATA
    # =============================================================================
    print("Loading filtered detections...")
    results_df = pd.read_csv(input_csv)
    results_df['Date_Only'] = pd.to_datetime(results_df['Date_Only']).dt.date

    # Filter: only days with substantial activity (>10 detections per species per day)
    daily_counts = results_df.groupby(['common_name', 'Date_Only']).size().reset_index(name='daily_count')
    daily_counts = daily_counts[daily_counts['daily_count'] > 10]
    valid_birds = daily_counts['common_name'].value_counts()[lambda x: x > MIN_DAYS].index
    daily_counts = daily_counts[daily_counts['common_name'].isin(valid_birds)]

    filtered_results = results_df.merge(
        daily_counts[['common_name', 'Date_Only']],
        on=['common_name', 'Date_Only'], how='inner'
    )

    print(f"Species with sufficient data: {len(valid_birds)}")

    # =============================================================================
    # EXTRACT PEAK ACTIVITY HOURS
    # =============================================================================
    print("Extracting peak activity hours...")

    peak_results = filtered_results.groupby(
        ['common_name', 'Date_Only']
    )['hour'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    peak_results.rename(columns={'hour': 'peak_hour'}, inplace=True)
    peak_results['date'] = pd.to_datetime(peak_results['Date_Only'])

    # =============================================================================
    # COMPUTE SUNRISE/SUNSET TIMES
    # =============================================================================
    print("Computing sunrise/sunset times...")

    min_date = peak_results['Date_Only'].min()
    max_date = peak_results['Date_Only'].max()
    date_range = pd.date_range(start=min_date, end=max_date)

    sun_data = []
    if ASTRAL_AVAILABLE:
        city = LocationInfo("Sanjay Van", "India", TIMEZONE_STR, LATITUDE, LONGITUDE)
        timezone = pytz.timezone(TIMEZONE_STR)

        for current_date in date_range:
            try:
                sr_dt = sunrise(city.observer, date=current_date.date(), tzinfo=timezone)
                ss_dt = sunset(city.observer, date=current_date.date(), tzinfo=timezone)
                sun_data.append({
                    'Date_Only': current_date.date(),
                    'Sunrise': sr_dt.hour + sr_dt.minute / 60 + sr_dt.second / 3600,
                    'Sunset': ss_dt.hour + ss_dt.minute / 60 + ss_dt.second / 3600
                })
            except ValueError:
                continue
    else:
        # Fallback: approximate sunrise/sunset for Delhi (28.5N)
        for current_date in date_range:
            day_of_year = current_date.timetuple().tm_yday
            # Simple sinusoidal approximation
            sr_approx = 6.0 + 0.75 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            ss_approx = 18.0 + 0.75 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            sun_data.append({
                'Date_Only': current_date.date(),
                'Sunrise': sr_approx,
                'Sunset': ss_approx
            })

    df_sun = pd.DataFrame(sun_data)

    # =============================================================================
    # MERGE AND COMPUTE CORRELATIONS
    # =============================================================================
    print("Computing Pearson correlations...")

    merged = pd.merge(peak_results, df_sun, on='Date_Only')

    pearson_results = []
    for bird in merged['common_name'].unique():
        subset = merged[merged['common_name'] == bird]

        if len(subset) > 1 and subset['peak_hour'].std() > 0:
            # Notebook cell 31: overall Pearson for sunrise and sunset
            coef_sr, p_val_sr = pearsonr(subset['peak_hour'], subset['Sunrise'])
            coef_ss, p_val_ss = pearsonr(subset['peak_hour'], subset['Sunset'])

            pearson_results.append({
                'Bird': bird,
                'Pearson_Sunrise': round(coef_sr, 3),
                'P-Val_Sunrise': round(p_val_sr, 4),
                'Pearson_Sunset': round(coef_ss, 3),
                'P-Val_Sunset': round(p_val_ss, 4),
                'Sample_Size': len(subset)
            })

    pearson_df = pd.DataFrame(pearson_results).sort_values(by='Pearson_Sunrise', ascending=False)
    print("Pearson Correlations with Sunrise/Sunset:")
    print(pearson_df)

    # =============================================================================
    # SAVE AND DISPLAY
    # =============================================================================
    csv_path = os.path.join(output_dir, "solar_correlation_results.csv")
    pearson_df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # =============================================================================
    # PLOT: Sunrise time series + peak activity overlay for top species
    # =============================================================================
    print("\nGenerating sunrise/peak activity overlay plots...")

    # NOTE: The notebook (cell 31) does not generate overlay plots.
    # Below is a bonus visualization not from notebook code.
    # Comment out if strict notebook-only output is needed.
    top_sunrise_species = pearson_df.head(3)['Bird'].tolist()

    for bird in top_sunrise_species:
        bird_data = merged[merged['common_name'] == bird].sort_values('date')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(bird_data['date'], bird_data['peak_hour'],
                'o-', color='#2ecc71', linewidth=2, markersize=4, label='Peak Activity Hour')
        ax.plot(bird_data['date'], bird_data['Sunrise'],
                '-', color='#FF8C00', linewidth=2, label='Sunrise Time')

        r_val = pearson_df[pearson_df['Bird'] == bird]['Pearson_Sunrise'].values[0]
        ax.set_title(f'{bird}: Peak Activity vs Sunrise (r={r_val:.3f})', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Hour of Day', fontsize=12)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        clean_name = bird.replace(" ", "_").lower()
        outpath = os.path.join(output_dir, f"solar_overlay_{clean_name}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone. All results saved to:", output_dir)


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================

def _resolve_dependency_aggregate(args, dep_filename):
    """Find a dependency aggregate CSV. Priority: sibling of --aggregate-file > project DB."""
    if args.aggregate_file:
        agg_dir = os.path.dirname(args.aggregate_file)
        candidate = os.path.join(agg_dir, dep_filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _run_watcher_mode():
    """Aggregate-aware watcher mode for solar correlation."""
    args = config.parse_common_args(description="08 – Solar Correlation (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    # Try aggregate first, fall back to legacy resolution
    agg_path = _resolve_dependency_aggregate(args, "filtered_detections.csv")
    if agg_path:
        print(f"Loading from aggregate: {agg_path}")
        df = config.load_aggregate(agg_path)
        df = config.filter_aggregate_for_output(df, args.start_date, args.end_date, args.spots)
        if df.empty:
            print("ERROR: Aggregate is empty after filtering.")
            sys.exit(1)
        tmp_csv = os.path.join(args.output_dir, "_filtered_input.csv")
        df.to_csv(tmp_csv, index=False)
        run_solar_correlation(tmp_csv, args.output_dir)
        os.remove(tmp_csv)
    else:
        detection_csv = config.resolve_detection_csv(args)
        if not detection_csv:
            print("ERROR: No filtered_detections.csv found. Run 01 first.")
            sys.exit(1)
        run_solar_correlation(detection_csv, args.output_dir)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    _run_watcher_mode()
