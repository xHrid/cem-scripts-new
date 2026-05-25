"""
Script 02: Species Activity Heatmaps (Normalized & Non-Normalized)
===================================================================
Generates per-site heatmaps showing hourly bird activity patterns.

Produces:
  - Non-normalized heatmap: Average detections per hour (Fig 13 in paper)
  - Normalized heatmap: Proportion of daily activity per species (Fig 14 in paper)

Paper Reference: Section 4.3.1 - Dominant Species
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import shared config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

TOP_N_SPECIES = 25


def run_heatmaps(input_csv, output_dir, top_n=25):
    """Core logic: generate heatmaps from a detections CSV."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading filtered detections...")
    results_df = pd.read_csv(input_csv)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    results_df = results_df[~results_df['label'].str.contains("Engine|Siren", na=False)]
    print(f"Loaded {len(results_df)} detections across {sorted(results_df['Spot'].unique())}")

    # --- Non-Normalized Heatmaps ---
    print("\n--- Generating Non-Normalized Heatmaps ---")
    for spot in sorted(results_df['Spot'].unique()):
        spot_df = results_df[results_df['Spot'] == spot]
        num_days = spot_df['Date'].dt.date.nunique()
        if num_days == 0:
            continue

        top_species = spot_df['label'].value_counts().nlargest(top_n).index
        spot_df_top = spot_df[spot_df['label'].isin(top_species)]

        activity_pivot = spot_df_top.pivot_table(
            index='label', columns='hour', values='filename',
            aggfunc='count', fill_value=0
        )
        average_activity = activity_pivot / num_days

        plt.figure(figsize=(20, 10))
        sns.heatmap(average_activity, cmap="YlGnBu", linewidths=0.5,
                    annot=True, fmt=".2f",
                    cbar_kws={'label': 'Avg. Detections per Hour'})
        plt.title(f"Average Detections per Hour - {spot.replace('_', ' ').title()} "
                  f"(Averaged over {num_days} days)", fontsize=14)
        plt.xlabel("Hour of Day")
        plt.ylabel("Species")
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"heatmap_non_normalized_{spot}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"  Saved: {outpath}")

    # --- Normalized Heatmaps ---
    print("\n--- Generating Normalized Heatmaps ---")
    for spot in sorted(results_df['Spot'].unique()):
        spot_df = results_df[results_df['Spot'] == spot]
        num_days = spot_df['Date'].dt.date.nunique()
        if num_days == 0:
            continue

        top_species = spot_df['label'].value_counts().nlargest(top_n).index
        spot_df_top = spot_df[spot_df['label'].isin(top_species)]

        activity_pivot = spot_df_top.pivot_table(
            index='label', columns='hour', values='filename',
            aggfunc='count', fill_value=0
        )
        average_activity = activity_pivot / num_days
        normalized_activity = average_activity.div(average_activity.sum(axis=1), axis=0)

        plt.figure(figsize=(20, 10))
        sns.heatmap(normalized_activity, cmap="YlGnBu", linewidths=0.5,
                    annot=True, fmt=".2f",
                    cbar_kws={'label': 'Proportion of Daily Activity'})
        plt.title(f"Normalized Hourly Activity - {spot.replace('_', ' ').title()} "
                  f"(Proportion per Species)", fontsize=14)
        plt.xlabel("Hour of Day")
        plt.ylabel("Species")
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"heatmap_normalized_{spot}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone. All heatmaps saved to:", output_dir)


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================

def _run_watcher_mode():
    """Watcher mode: load birdnet aggregate, filter inline, generate heatmaps."""
    args = config.parse_common_args(description="02 – Activity Heatmaps (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    df = config.load_filtered_detections(args)
    if df.empty:
        print("ERROR: No data after filtering. Run BirdNET inference first.")
        sys.exit(1)

    tmp_csv = os.path.join(args.output_dir, "_filtered_input.csv")
    df.to_csv(tmp_csv, index=False)
    run_heatmaps(tmp_csv, args.output_dir, top_n=TOP_N_SPECIES)
    os.remove(tmp_csv)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    _run_watcher_mode()
