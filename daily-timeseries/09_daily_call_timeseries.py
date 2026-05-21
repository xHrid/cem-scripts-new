"""
Script 09: Daily Call Frequency Time Series
=============================================
Generates per-species time series plots showing daily call counts over
the study period, with data gaps (no recorder) highlighted.

Also computes the eBird overlay comparison (z-score normalization of
recorder data vs eBird sighting data) for migratory validation.

Paper Reference: Section 4.3.4 (migratory patterns visible in time series),
                 Section 3.2.5 ("Silent Resident" problem - Call-to-Sighting Ratio)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# =============================================================================
# CONFIGURATION (imported from 00_config.py)
# =============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

# Set to None to plot all species, or provide list of specific species
SPECIES_TO_PLOT = None  # e.g., ['Common Hawk-Cuckoo', 'Hume\'s Warbler']

# Maximum species to plot (if SPECIES_TO_PLOT is None)
MAX_SPECIES = 50


def run_daily_timeseries(input_csv, output_dir, species_to_plot=None, max_species=50):
    """Core logic: generate daily call frequency time series from a detections CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # LOAD DATA
    # =============================================================================
    print("Loading filtered detections...")
    results_df = pd.read_csv(input_csv)
    results_df['Date_Only'] = pd.to_datetime(results_df['Date_Only']).dt.date

    # Determine global dates with data (recorder active)
    global_dates_with_data = set(results_df['Date_Only'].unique())
    all_dates = pd.date_range(
        start=min(global_dates_with_data),
        end=max(global_dates_with_data)
    ).date

    print(f"Study period: {min(all_dates)} to {max(all_dates)}")
    print(f"Days with data: {len(global_dates_with_data)} / {len(all_dates)}")

    # =============================================================================
    # DETERMINE SPECIES TO PLOT
    # =============================================================================
    if species_to_plot is None:
        # Use all unique species (capped at max_species by total detections)
        species_counts = results_df['common_name'].value_counts()
        unique_birds = species_counts.head(max_species).index.tolist()
    else:
        unique_birds = species_to_plot

    print(f"Generating time series for {len(unique_birds)} species...")

    # =============================================================================
    # GENERATE TIME SERIES PLOTS
    # =============================================================================
    sns.set_style("whitegrid")

    for idx, bird in enumerate(unique_birds):
        if idx % 10 == 0:
            print(f"  Processing {idx+1}/{len(unique_birds)}: {bird}")

        # Daily call count for this species
        bird_data = results_df[results_df['common_name'] == bird].groupby(
            'Date_Only'
        ).size().reset_index(name='call_count')

        # Full date range DataFrame
        df_plot = pd.DataFrame({'Date_Only': all_dates})
        df_plot = df_plot.merge(bird_data, on='Date_Only', how='left').fillna(0)

        # Mark global data gaps (recorder not active)
        df_plot['is_global_gap'] = df_plot['Date_Only'].apply(
            lambda x: x not in global_dates_with_data
        )

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Red line = no data available (gap)
        ax.plot(
            df_plot['Date_Only'], df_plot['call_count'],
            color='#e74c3c', linewidth=2.5, label='No Data Available', zorder=1
        )

        # Green line = recorder active (overwrite on active days)
        df_green = df_plot.copy()
        df_green.loc[df_green['is_global_gap'], 'call_count'] = np.nan
        ax.plot(
            df_green['Date_Only'], df_green['call_count'],
            color='#2ecc71', linewidth=3, label='Recorder Active', zorder=2
        )

        ax.set_title(f'Daily Call Frequency: {bird}', fontsize=16, pad=15)
        ax.set_ylabel('Number of Calls', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(frameon=True)
        plt.tight_layout()

        clean_name = bird.replace(" ", "_").lower()
        outpath = os.path.join(output_dir, f"ts_{clean_name}.png")
        plt.savefig(outpath, dpi=150)
        plt.close(fig)

    # =============================================================================
    # DATA AVAILABILITY HEATMAP (Figure 12 in paper)
    # =============================================================================
    print("\nGenerating data availability heatmap...")

    # Count recordings per site per date
    availability = results_df.groupby(['Spot', 'Date_Only']).size().reset_index(name='count')
    availability['Date_Only'] = pd.to_datetime(availability['Date_Only'])

    # Pivot for heatmap
    avail_pivot = availability.pivot_table(
        index='Spot', columns='Date_Only', values='count', fill_value=0
    )

    # Binary: 1 if any recordings, 0 otherwise
    avail_binary = (avail_pivot > 0).astype(int)

    plt.figure(figsize=(20, 4))
    sns.heatmap(avail_binary, cmap='Greens', cbar=False, linewidths=0.1)
    plt.title('Data Availability Across Sites', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Monitoring Site')
    plt.tight_layout()

    outpath = os.path.join(output_dir, "data_availability_heatmap.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")

    print(f"\nDone. {len(unique_birds)} time series plots saved to: {output_dir}")


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
    """Aggregate-aware watcher mode for daily call timeseries."""
    args = config.parse_common_args(description="09 – Daily Call Timeseries (watcher)")
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
        run_daily_timeseries(tmp_csv, args.output_dir,
                             species_to_plot=SPECIES_TO_PLOT, max_species=MAX_SPECIES)
        os.remove(tmp_csv)
    else:
        detection_csv = config.resolve_detection_csv(args)
        if not detection_csv:
            print("ERROR: No filtered_detections.csv found. Run 01 first.")
            sys.exit(1)
        run_daily_timeseries(detection_csv, args.output_dir,
                             species_to_plot=SPECIES_TO_PLOT, max_species=MAX_SPECIES)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    if "--output-dir" in sys.argv:
        _run_watcher_mode()
    else:
        # Standalone mode
        run_daily_timeseries("filtered_detections.csv", "results_timeseries",
                             species_to_plot=SPECIES_TO_PLOT, max_species=MAX_SPECIES)
