"""
Script 06: NDSI Box Plots Across Sites
========================================
Generates box plots of NDSI (and other acoustic indices) across monitoring sites.

Paper Reference: Section 4.3.3, Figure 17 - "box plots generated for all sites for index NDSI"
  - Site 1: High NDSI (biophony-dominated)
  - Sites 2,3,4: Low NDSI (anthrophony-dominated)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys

# =============================================================================
# CONFIGURATION (imported from 00_config.py)
# =============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

INDICES_DIR = config.INDICES_OUTPUT_DIR

# Indices to plot
INDICES_TO_PLOT = ['NDSI', 'ADI', 'ACI', 'AEI', 'MFC', 'CLS']


# =============================================================================
# CORE ANALYSIS
# =============================================================================
def run_boxplots(indices_csv_or_dir, output_dir):
    """Generate box plots of acoustic indices across monitoring sites.

    Parameters
    ----------
    indices_csv_or_dir : str
        Path to a single combined CSV **or** a directory containing
        per-spot ``*_indices.csv`` files.
    output_dir : str
        Directory where PNG plots and summary output are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------
    print("Loading acoustic index data...")

    if os.path.isfile(indices_csv_or_dir):
        df = pd.read_csv(indices_csv_or_dir)
    else:
        # Load all index CSVs from directory and combine
        csv_files = glob.glob(os.path.join(indices_csv_or_dir, "*_indices.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No index CSVs found in {indices_csv_or_dir}")

        dfs = []
        for f in csv_files:
            temp = pd.read_csv(f)
            # Extract spot from filename
            basename = os.path.basename(f)
            spot_match = __import__('re').search(r'spot(\d+)', basename)
            if spot_match:
                temp['Spot'] = f"Site {spot_match.group(1)}"
            dfs.append(temp)

        df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} records")
    print(f"Sites: {sorted(df['Spot'].unique())}")

    # If 'Spot' column uses spot_1 format, rename for paper consistency
    if df['Spot'].str.contains('spot_').any():
        df['Spot'] = df['Spot'].str.replace('spot_', 'Site ')

    # ------------------------------------------------------------------
    # GENERATE BOX PLOTS
    # ------------------------------------------------------------------
    print("\nGenerating box plots...")

    for index_name in INDICES_TO_PLOT:
        if index_name not in df.columns:
            print(f"  WARNING: {index_name} not found in data. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x='Spot',
            y=index_name,
            order=sorted(df['Spot'].unique()),
            palette='Set2'
        )
        plt.title(f'Distribution of {index_name} Across Monitoring Sites', fontsize=16)
        plt.xlabel('Monitoring Site', fontsize=12)
        plt.ylabel(f'{index_name} Value', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        outpath = os.path.join(output_dir, f"boxplot_{index_name}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone. All box plots saved to:", output_dir)


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
    """Aggregate-aware watcher mode for NDSI boxplots."""
    args = config.parse_common_args(description="06 – NDSI Boxplots (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    # Try aggregate first (acoustic_indices), fall back to legacy resolution
    agg_path = _resolve_dependency_aggregate(args, "acoustic_indices.csv")
    if agg_path:
        print(f"Loading from aggregate: {agg_path}")
        df = config.load_aggregate(agg_path)
        df = config.filter_aggregate_for_output(df, args.start_date, args.end_date, args.spots)
        if df.empty:
            print("ERROR: Aggregate is empty after filtering.")
            sys.exit(1)
        tmp_csv = os.path.join(args.output_dir, "_filtered_input.csv")
        df.to_csv(tmp_csv, index=False)
        run_boxplots(tmp_csv, args.output_dir)
        os.remove(tmp_csv)
    else:
        indices_csv = config.resolve_indices_csv(args)
        if not indices_csv:
            print("ERROR: No acoustic indices CSV found. Run 05 first.")
            sys.exit(1)
        run_boxplots(indices_csv, args.output_dir)

    config.save_processed_list(args.output_dir, ["aggregate"])


if __name__ == "__main__":
    if "--output-dir" in sys.argv:
        _run_watcher_mode()
    else:
        # Standalone mode
        run_boxplots(INDICES_DIR, "results_ndsi_boxplots")
