"""
Script 07: Migratory vs Resident Bird Classification
======================================================
!!  PAPER-DERIVED — NOT FROM NOTEBOOK CODE  !!

The notebooks (graphs_analysis.ipynb cell 33) contain only a stub:
    get_active_months() + print("SCI / Kurtosis / PMR ... Executed Correctly")
The actual SCI, Residual Kurtosis, and PMR computation code is NOT present
in any notebook. This script was RECONSTRUCTED from the paper's mathematical
definitions in Section 3.2.5 (3.2.6.1, 3.2.6.2, 3.2.6.3) and the threshold
values stated in Section 4.3.4. Review carefully before trusting outputs.

Metrics implemented:
1. SCI (Seasonal Concentration Index):
   - 60-day sliding window, fraction of total detections in max window
   - Threshold: SCI > 0.9

2. Residual Kurtosis (K):
   - Kurtosis of residuals after linear detrending daily counts
   - Threshold: K > 15

3. PMR (Peak-to-Median Ratio):
   - Max daily count / median daily count
   - Threshold: PMR > 50

Species classified migratory if ALL three thresholds exceeded.

Paper Reference: Section 3.2.5, Section 4.3.4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
import os
import sys

# =============================================================================
# CONFIGURATION (imported from 00_config.py)
# =============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

# Thresholds from paper Section 4.3.4
SCI_THRESHOLD = 0.9
KURTOSIS_THRESHOLD = 15
PMR_THRESHOLD = 50

# Sliding window size (days)
WINDOW_SIZE = 60

# Small constant to avoid division by zero in PMR
EPSILON = 1e-6


# =============================================================================
# CORE ANALYSIS
# =============================================================================
def run_migratory_classification(detection_csv, output_dir):
    """Classify species as migratory or resident using SCI, Kurtosis, PMR.

    Parameters
    ----------
    detection_csv : str
        Path to ``filtered_detections.csv``.
    output_dir : str
        Directory where CSV results and plots are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------
    print("Loading filtered detections...")
    results_df = pd.read_csv(detection_csv)
    results_df['Date_Only'] = pd.to_datetime(results_df['Date_Only'])

    # Compute daily detection counts per species (aggregated across all sites/hours)
    daily_counts = results_df.groupby(['common_name', 'Date_Only']).size().reset_index(name='daily_count')

    # Create full date range
    all_dates = pd.date_range(
        start=daily_counts['Date_Only'].min(),
        end=daily_counts['Date_Only'].max()
    ).normalize()

    unique_birds = daily_counts['common_name'].unique()
    print(f"Analyzing {len(unique_birds)} species over {len(all_dates)} days")

    # ------------------------------------------------------------------
    # COMPUTE METRICS
    # ------------------------------------------------------------------
    print("\nComputing SCI, Kurtosis, PMR for each species...")

    results = []
    for bird in unique_birds:
        bird_data = daily_counts[daily_counts['common_name'] == bird].copy()

        # Create full time series (fill missing days with 0)
        full_ts = pd.DataFrame({'Date_Only': all_dates})
        full_ts = full_ts.merge(bird_data[['Date_Only', 'daily_count']], on='Date_Only', how='left')
        full_ts['daily_count'] = full_ts['daily_count'].fillna(0)

        counts = full_ts['daily_count'].values
        total_detections = counts.sum()

        if total_detections == 0:
            continue

        # --- SCI: Seasonal Concentration Index ---
        if len(counts) >= WINDOW_SIZE:
            rolling_sum = pd.Series(counts).rolling(window=WINDOW_SIZE).sum().dropna()
            rs_max = rolling_sum.max()
        else:
            rs_max = total_detections
        sci = rs_max / total_detections

        # --- Residual Kurtosis ---
        n = len(counts)
        x = np.arange(n)
        if n > 2 and np.std(counts) > 0:
            coeffs = np.polyfit(x, counts, 1)
            predicted = np.polyval(coeffs, x)
            residuals = counts - predicted

            r_mean = residuals.mean()
            r_std = residuals.std()
            if r_std > 0:
                k_value = np.mean(((residuals - r_mean) / r_std) ** 4)
            else:
                k_value = 0.0
        else:
            k_value = 0.0

        # --- PMR: Peak-to-Median Ratio ---
        c_max = counts.max()
        c_median = np.median(counts)
        pmr = c_max / (c_median + EPSILON)

        # Classification
        is_migratory = (sci > SCI_THRESHOLD) and (k_value > KURTOSIS_THRESHOLD) and (pmr > PMR_THRESHOLD)

        results.append({
            'Species': bird,
            'SCI': round(sci, 4),
            'Kurtosis': round(k_value, 2),
            'PMR': round(pmr, 2),
            'Total_Detections': int(total_detections),
            'Classification': 'Migratory' if is_migratory else 'Resident'
        })

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values('SCI', ascending=False)

    n_migratory = (metrics_df['Classification'] == 'Migratory').sum()
    n_resident = (metrics_df['Classification'] == 'Resident').sum()
    print(f"\nClassification Results:")
    print(f"  Migratory: {n_migratory} species")
    print(f"  Resident: {n_resident} species")

    # Save full results
    csv_path = os.path.join(output_dir, "migratory_classification_all_species.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # ------------------------------------------------------------------
    # PLOT: Distributions of SCI, Kurtosis, PMR
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Migratory Classification Metrics', fontsize=16, fontweight='bold')

    colors = {'Migratory': '#e74c3c', 'Resident': '#2ecc71'}

    # SCI distribution
    ax = axes[0]
    for cls in ['Resident', 'Migratory']:
        subset = metrics_df[metrics_df['Classification'] == cls]
        ax.hist(subset['SCI'], bins=20, alpha=0.7, label=cls, color=colors[cls])
    ax.axvline(SCI_THRESHOLD, color='black', linestyle='--', label=f'Threshold={SCI_THRESHOLD}')
    ax.set_xlabel('Seasonal Concentration Index (SCI)')
    ax.set_ylabel('Count')
    ax.set_title('SCI Distribution')
    ax.legend()

    # Kurtosis distribution
    ax = axes[1]
    for cls in ['Resident', 'Migratory']:
        subset = metrics_df[metrics_df['Classification'] == cls]
        ax.hist(subset['Kurtosis'], bins=20, alpha=0.7, label=cls, color=colors[cls])
    ax.axvline(KURTOSIS_THRESHOLD, color='black', linestyle='--', label=f'Threshold={KURTOSIS_THRESHOLD}')
    ax.set_xlabel('Residual Kurtosis (K)')
    ax.set_title('Kurtosis Distribution')
    ax.legend()

    # PMR distribution
    ax = axes[2]
    for cls in ['Resident', 'Migratory']:
        subset = metrics_df[metrics_df['Classification'] == cls]
        ax.hist(subset['PMR'].clip(upper=200), bins=20, alpha=0.7, label=cls, color=colors[cls])
    ax.axvline(PMR_THRESHOLD, color='black', linestyle='--', label=f'Threshold={PMR_THRESHOLD}')
    ax.set_xlabel('Peak-to-Median Ratio (PMR)')
    ax.set_title('PMR Distribution')
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(output_dir, "migratory_classification_distributions.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

    # --- Scatter: SCI vs PMR ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in ['Resident', 'Migratory']:
        subset = metrics_df[metrics_df['Classification'] == cls]
        ax.scatter(subset['SCI'], subset['PMR'].clip(upper=300),
                   alpha=0.6, label=cls, color=colors[cls], s=40)
    ax.axvline(SCI_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(PMR_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Seasonal Concentration Index (SCI)', fontsize=12)
    ax.set_ylabel('Peak-to-Median Ratio (PMR)', fontsize=12)
    ax.set_title('Migratory Classification: SCI vs PMR', fontsize=14)
    ax.legend()
    plt.tight_layout()

    outpath = os.path.join(output_dir, "migratory_sci_vs_pmr.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

    print("\nDone. All results saved to:", output_dir)


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================

def _run_watcher_mode():
    """Watcher mode: load birdnet aggregate, filter inline, classify."""
    args = config.parse_common_args(description="07 – Migratory Classification (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    df = config.load_filtered_detections(args)
    if df.empty:
        print("ERROR: No data after filtering. Run BirdNET inference first.")
        sys.exit(1)

    tmp_csv = os.path.join(args.output_dir, "_filtered_input.csv")
    df.to_csv(tmp_csv, index=False)
    run_migratory_classification(tmp_csv, args.output_dir)
    os.remove(tmp_csv)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    _run_watcher_mode()
