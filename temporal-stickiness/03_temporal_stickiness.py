"""
Script 03: Activity Regularity (Temporal Stickiness)
=====================================================
Computes Activity Regularity (AR) per species using Spearman rank correlation
of consecutive-day hourly activity vectors.

Formula (from paper Section 3.2.3):
  For each species s, location j, day k:
    X_{s,j,k} = [c1, c2, ..., c24]  (hourly detection counts)
    rho_{s,j,k} = Spearman(X_{s,j,k}, X_{s,j,k+1})
    AR_s = mean of all valid rho values across locations and days

Produces:
  - Bar chart: Top 80 species by temporal stickiness (Fig 15 in paper)
  - Combined with average daily call volume
  - CSV export of all species stickiness values

Paper Reference: Section 4.3.2 - "top 80 species ranked by temporal stickiness"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import sys

# Import shared config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

ACTIVITY_HOURS = range(0, 24)
TOP_N = 80  # Paper shows top 80


def run_analysis(input_csv, output_dir):
    """Core logic: compute Activity Regularity and produce outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # LOAD DATA
    # =============================================================================
    print("Loading filtered detections...")
    results_df = pd.read_csv(input_csv)
    results_df['Date'] = pd.to_datetime(results_df['Date'])

    activity_df = results_df[
        (results_df['confidence'] >= 0.3) &
        (results_df['hour'].isin(ACTIVITY_HOURS))
    ].copy()

    species_list = activity_df['label'].unique()
    spot_list = activity_df['Spot'].unique()
    date_list = sorted(activity_df['Date'].unique())
    num_days = activity_df['Date'].nunique()

    print(f"Species: {len(species_list)}, Spots: {len(spot_list)}, Days: {num_days}")

    # =============================================================================
    # COMPUTE TEMPORAL STICKINESS (Activity Regularity)
    # =============================================================================
    print("\nCalculating Activity Regularity...")

    # Pre-compute hourly count pivot: (species, spot, date) -> 24-hour vector
    hourly_counts = (
        activity_df
        .groupby(['label', 'Spot', 'Date', 'hour'])
        .size()
        .unstack(level='hour', fill_value=0)
        .reindex(columns=list(ACTIVITY_HOURS), fill_value=0)
    )

    temporal_stickiness = {}
    for idx, species in enumerate(species_list):
        if idx % 20 == 0:
            print(f"  Processing species {idx+1}/{len(species_list)}...")

        species_spot_correlations = []

        # Check if species exists in the pre-computed index
        if species not in hourly_counts.index.get_level_values('label'):
            continue

        species_data = hourly_counts.loc[species]

        for spot in spot_list:
            if spot not in species_data.index.get_level_values('Spot'):
                continue

            spot_data = species_data.loc[spot]  # DataFrame indexed by Date
            spot_dates = spot_data.index

            spot_day_correlations = []
            for i in range(len(date_list) - 1):
                day_k = date_list[i]
                day_k_plus_1 = date_list[i + 1]

                if day_k not in spot_dates or day_k_plus_1 not in spot_dates:
                    continue

                series_k = spot_data.loc[day_k]
                series_k1 = spot_data.loc[day_k_plus_1]

                # Only compute if both days have detections
                if series_k.sum() > 0 and series_k1.sum() > 0:
                    corr, _ = spearmanr(series_k.values, series_k1.values)
                    if not np.isnan(corr):
                        spot_day_correlations.append(corr)

            if spot_day_correlations:
                species_spot_correlations.append(np.mean(spot_day_correlations))

        if species_spot_correlations:
            temporal_stickiness[species] = np.mean(species_spot_correlations)

    # =============================================================================
    # COMPUTE AVERAGE DAILY CALLS
    # =============================================================================
    print("Computing average daily call volumes...")
    avg_calls_all = activity_df.groupby("label").size().reset_index(name="total_calls")
    avg_calls_all["Avg_Calls_Per_Day"] = avg_calls_all["total_calls"] / num_days

    # =============================================================================
    # COMBINE AND EXPORT
    # =============================================================================
    temporal_df = pd.DataFrame(
        list(temporal_stickiness.items()),
        columns=['label', 'Activity_Regularity']
    )
    combined_df = pd.merge(temporal_df, avg_calls_all, on="label", how="left")
    combined_df = combined_df.sort_values(by='Activity_Regularity', ascending=False)

    csv_path = os.path.join(output_dir, "all_species_activity_regularity.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"\nSaved full results to: {csv_path}")

    # =============================================================================
    # PLOT: Top N species by Activity Regularity + Avg Daily Calls
    # =============================================================================
    top_temporal = combined_df.head(TOP_N)
    top_calls = combined_df.set_index("label").reindex(
        top_temporal["label"]
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle(
        f'Activity Regularity: Top {TOP_N} Species\n'
        f'(Average Spearman rho of consecutive-day hourly patterns)',
        fontsize=16, fontweight='bold'
    )

    # Left: Stickiness bars
    sns.barplot(
        x='Activity_Regularity', y='label',
        data=top_temporal, palette='plasma', ax=ax1
    )
    ax1.set_title('Activity Regularity (Predictability)', fontsize=14)
    ax1.set_xlabel("Average Spearman Correlation (rho)", fontsize=12)
    ax1.set_ylabel("Species", fontsize=12)
    ax1.set_xlim(-0.2, 1.0)
    ax1.grid(axis='x', linestyle='--', alpha=0.6)

    # Right: Average daily calls
    sns.barplot(
        x="Avg_Calls_Per_Day", y="label",
        data=top_calls, palette="magma", ax=ax2
    )
    ax2.set_title('Average Daily Call Volume', fontsize=14)
    ax2.set_xlabel("Average Calls per Day", fontsize=12)
    ax2.set_ylabel("")
    ax2.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "temporal_stickiness_top_species.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

    print("\nDone. All results saved to:", output_dir)


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================

def _run_watcher_mode():
    """Watcher mode: load birdnet aggregate, filter inline, run analysis."""
    args = config.parse_common_args(description="03 – Temporal Stickiness (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    df = config.load_filtered_detections(args)
    if df.empty:
        print("ERROR: No data after filtering. Run BirdNET inference first.")
        sys.exit(1)

    tmp_csv = os.path.join(args.output_dir, "_filtered_input.csv")
    df.to_csv(tmp_csv, index=False)
    run_analysis(tmp_csv, args.output_dir)
    os.remove(tmp_csv)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    _run_watcher_mode()
