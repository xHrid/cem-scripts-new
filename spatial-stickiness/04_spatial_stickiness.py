"""
Script 04: Habitat Affinity (Spatial Stickiness)
=================================================
Computes Habitat Affinity (HA) per species using Spearman rank correlation
of consecutive-day spatial distribution vectors.

Formula (from paper Section 3.2.4):
  For each species s, day k:
    Y_{s,k} = [c_j1, c_j2, ..., c_jm]  (detection counts per site)
    rho_{s,k} = Spearman(Y_{s,k}, Y_{s,k+1})
    HA_s = mean of all valid rho values

Only species present at ALL sites are analyzed (requires spatial variation).

Produces:
  - Bar chart: Species ranked by spatial stickiness (Fig 16 in paper)
  - Aligned activity heatmap showing per-site detection levels
  - CSV export

Paper Reference: Section 4.3.2 - "spatial stickiness (Average Spearman correlation, rho)"
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


def run_analysis(input_csv, output_dir):
    """Core logic: compute Habitat Affinity and produce outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # LOAD DATA
    # =============================================================================
    print("Loading filtered detections...")
    results_df = pd.read_csv(input_csv)
    results_df['Date'] = pd.to_datetime(results_df['Date'])

    spatial_df = results_df.copy()
    spot_list = sorted(spatial_df['Spot'].unique())
    date_list = sorted(spatial_df['Date'].unique())
    total_spots = len(spot_list)

    print(f"Total spots: {total_spots}, Total days: {len(date_list)}")

    # =============================================================================
    # FILTER: Only species present at ALL sites
    # =============================================================================
    species_spot_counts = spatial_df.groupby('label')['Spot'].nunique()
    species_list = species_spot_counts[species_spot_counts == total_spots].index.tolist()
    print(f"Species present at all {total_spots} spots: {len(species_list)}")

    if len(spot_list) < 2:
        raise ValueError("Spatial stickiness requires data from at least 2 spots.")

    if not species_list:
        raise ValueError("No species found at all spots. Check data.")

    # =============================================================================
    # COMPUTE SPATIAL STICKINESS (Habitat Affinity)
    # =============================================================================
    print("\nCalculating Habitat Affinity...")

    # Pre-compute spatial count pivot: (species, date) -> [count per spot]
    spot_counts_pivot = (
        spatial_df[spatial_df['label'].isin(species_list)]
        .groupby(['label', 'Date', 'Spot'])
        .size()
        .unstack(level='Spot', fill_value=0)
        .reindex(columns=spot_list, fill_value=0)
    )

    spatial_stickiness = {}
    for idx, species in enumerate(species_list):
        if idx % 10 == 0:
            print(f"  Processing species {idx+1}/{len(species_list)}...")

        if species not in spot_counts_pivot.index.get_level_values('label'):
            continue

        species_data = spot_counts_pivot.loc[species]  # DataFrame indexed by Date
        available_dates = species_data.index

        daily_rank_correlations = []
        for i in range(len(date_list) - 1):
            day_k = date_list[i]
            day_k_plus_1 = date_list[i + 1]

            if day_k not in available_dates or day_k_plus_1 not in available_dates:
                continue

            counts_k = species_data.loc[day_k]
            counts_k_plus_1 = species_data.loc[day_k_plus_1]

            # Need variance in both to compute correlation
            if counts_k.nunique() > 1 and counts_k_plus_1.nunique() > 1:
                corr, _ = spearmanr(counts_k.values, counts_k_plus_1.values)
                if not np.isnan(corr):
                    daily_rank_correlations.append(corr)

        if daily_rank_correlations:
            spatial_stickiness[species] = np.mean(daily_rank_correlations)

    # =============================================================================
    # COMPUTE PER-SPOT ACTIVITY FOR HEATMAP
    # =============================================================================
    print("Computing per-spot activity levels...")
    activity_df = results_df[results_df['label'].isin(species_list)].copy()
    daily_counts = activity_df.groupby(['label', 'Spot', 'Date']).size().reset_index(name='daily_count')
    heatmap_data = daily_counts.groupby(['label', 'Spot'])['daily_count'].mean().unstack(fill_value=0)

    # =============================================================================
    # COMBINE AND EXPORT
    # =============================================================================
    spatial_df_out = pd.DataFrame(
        list(spatial_stickiness.items()),
        columns=['label', 'Habitat_Affinity']
    ).sort_values(by='Habitat_Affinity', ascending=False)

    # Merge with spot activity
    spot_activity_df = heatmap_data.reset_index()
    combined_spatial = pd.merge(spatial_df_out, spot_activity_df, on='label', how='outer')
    combined_spatial = combined_spatial.sort_values(by='Habitat_Affinity', ascending=False)

    csv_path = os.path.join(output_dir, "all_species_habitat_affinity.csv")
    combined_spatial.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # =============================================================================
    # PLOT: Spatial Stickiness Bar Chart
    # =============================================================================
    plt.figure(figsize=(10, max(12, len(spatial_df_out) * 0.3)))
    sns.barplot(
        x='Habitat_Affinity', y='label',
        data=spatial_df_out, palette='viridis'
    )
    plt.title(f'Habitat Affinity ({len(spatial_df_out)} Species at All Sites)', fontsize=16)
    plt.xlabel('Average Spearman Correlation (rho)', fontsize=12)
    plt.ylabel('Species', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    outpath = os.path.join(output_dir, "spatial_stickiness_bar_chart.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

    # --- Aligned Heatmap ---
    if not heatmap_data.empty:
        ordered_species = spatial_df_out['label'].tolist()
        heatmap_ordered = heatmap_data.reindex(ordered_species).fillna(0)

        plt.figure(figsize=(12, max(10, len(ordered_species) * 0.3)))
        sns.heatmap(heatmap_ordered, cmap='YlOrRd', annot=True, fmt='.1f',
                    cbar_kws={'label': 'Avg Daily Detections'})
        plt.title('Per-Site Activity (aligned with Habitat Affinity ranking)', fontsize=14)
        plt.xlabel('Monitoring Site')
        plt.ylabel('Species')
        plt.tight_layout()

        outpath = os.path.join(output_dir, "spatial_stickiness_heatmap.png")
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {outpath}")

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
    """Aggregate-aware watcher mode for spatial stickiness."""
    args = config.parse_common_args(description="04 – Spatial Stickiness (watcher)")
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
        run_analysis(tmp_csv, args.output_dir)
        os.remove(tmp_csv)
    else:
        detection_csv = config.resolve_detection_csv(args)
        if not detection_csv:
            print("ERROR: No filtered_detections.csv found. Run 01 first.")
            sys.exit(1)
        run_analysis(detection_csv, args.output_dir)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    _run_watcher_mode()
