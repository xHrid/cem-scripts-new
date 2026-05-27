"""
02: Species Activity Heatmaps (Normalized & Non-Normalized)
============================================================
Flow: Aggregate → 3-step filter → heatmap generation → save PNGs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from importlib import import_module
config = import_module("00_config")


# =============================================================================
# ANALYSIS
# =============================================================================
def run_heatmaps(df, output_dir, top_n=config.TOP_N_SPECIES):
    os.makedirs(output_dir, exist_ok=True)

    df = df[~df["label"].str.contains("Engine|Siren", na=False)]
    print(f"Working with {len(df)} detections across {sorted(df['Spot'].unique())}")

    for spot in sorted(df["Spot"].unique()):
        spot_df = df[df["Spot"] == spot]
        num_days = spot_df["Date"].dt.date.nunique()
        if num_days == 0:
            continue

        top_species = spot_df["label"].value_counts().nlargest(top_n).index
        spot_top = spot_df[spot_df["label"].isin(top_species)]

        pivot = spot_top.pivot_table(
            index="label", columns="hour", values="filename",
            aggfunc="count", fill_value=0,
        )
        avg = pivot / num_days

        # Non-normalized
        plt.figure(figsize=(20, 10))
        sns.heatmap(avg, cmap="YlGnBu", linewidths=0.5, annot=True, fmt=".2f",
                    cbar_kws={"label": "Avg. Detections per Hour"})
        plt.title(f"Average Detections per Hour - {spot.replace('_', ' ').title()} "
                  f"(Averaged over {num_days} days)", fontsize=14)
        plt.xlabel("Hour of Day")
        plt.ylabel("Species")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_non_normalized_{spot}.png"), dpi=300)
        plt.close()

        # Normalized
        norm = avg.div(avg.sum(axis=1), axis=0)
        plt.figure(figsize=(20, 10))
        sns.heatmap(norm, cmap="YlGnBu", linewidths=0.5, annot=True, fmt=".2f",
                    cbar_kws={"label": "Proportion of Daily Activity"})
        plt.title(f"Normalized Hourly Activity - {spot.replace('_', ' ').title()}", fontsize=14)
        plt.xlabel("Hour of Day")
        plt.ylabel("Species")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_normalized_{spot}.png"), dpi=300)
        plt.close()

    print(f"Done. Heatmaps saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    df = config.filter_detections(
        config.AGGREGATE_FILE, config.EBIRD_FILE,
        config.DATE_START, config.DATE_END, config.SPOT_NAMES,
    )
    if df.empty:
        print("ERROR: No data after filtering.")
        return
    run_heatmaps(df, config.OUTPUT_DIR_02_HEATMAPS)


if __name__ == "__main__":
    main()
