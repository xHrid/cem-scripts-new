"""
Script 01: Data Loading and Filtering Pipeline
================================================
Loads BirdNET classification CSVs, applies multi-stage filtering:
  - Step 1: Taxonomic verification (eBird checklist)
  - Step 2: Confidence thresholding (>=0.3)
  - Step 3: Minimum activity (>=10 total detections)

Output: filtered DataFrame saved as 'filtered_detections.csv'
Paper Reference: Section 3.2.1, Section 4.3 intro (91 confirmed species)
"""

import pandas as pd
import os
import re
import sys

# Import shared config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

# =============================================================================
# CONFIGURATION (standalone defaults)
# =============================================================================
EBIRD_SPECIES_FILE = config.EBIRD_SPECIES_FILE


# =============================================================================
# CORE FILTERING LOGIC (shared by both modes)
# =============================================================================

def filter_detections(results_df, ebird_file=None):
    """Apply 3-step filtering pipeline. Returns filtered DataFrame."""

    # -- Preprocessing: extract Spot & Date from filename --
    results_df['Spot'] = results_df['filename'].str.extract(
        r'(SPOT\d+)', expand=False
    ).str.lower().str.replace('spot', 'spot_')

    date_info = results_df['filename'].str.extract(r'_(\d{8})_')
    results_df['Date'] = pd.to_datetime(date_info[0], format='%Y%m%d')
    results_df.dropna(subset=['Spot', 'Date'], inplace=True)
    results_df['Date_Only'] = results_df['Date'].dt.date

    print(f"After preprocessing: {len(results_df)} detections")
    print(f"Spots found: {sorted(results_df['Spot'].unique())}")

    # -- Step 1: Taxonomic Verification (eBird checklist) --
    print("\nStep 1: Taxonomic verification...")
    if ebird_file and os.path.exists(ebird_file):
        with open(ebird_file, 'r') as file:
            sanjay_van_birds = [line.strip().split('_')[1] for line in file if '_' in line]
        before = results_df['common_name'].nunique()
        results_df = results_df[results_df['common_name'].isin(sanjay_van_birds)].copy()
        print(f"  Species before: {before}, after eBird filter: {results_df['common_name'].nunique()}")
    else:
        print(f"  WARNING: eBird species file not found. Skipping taxonomic filter.")

    # -- Step 2: Confidence Thresholding (>=0.3) --
    print("\nStep 2: Confidence thresholding (>=0.3)...")
    before = len(results_df)
    results_df = results_df[results_df['confidence'] >= 0.3].copy()
    print(f"  Detections before: {before}, after: {len(results_df)}")

    # -- Step 3: Minimum Activity (>=10 total detections) --
    print("\nStep 3: Minimum total detections (>=10)...")
    species_counts = results_df.groupby('common_name').size()
    valid_species = species_counts[species_counts >= 10].index
    before = results_df['common_name'].nunique()
    results_df = results_df[results_df['common_name'].isin(valid_species)].copy()
    print(f"  Species before: {before}, after: {results_df['common_name'].nunique()}")

    return results_df


def print_summary(df):
    """Print final dataset summary."""
    print(f"\n{'='*60}")
    print(f"FINAL DATASET:")
    print(f"  Total detections: {len(df)}")
    print(f"  Unique species: {df['common_name'].nunique()}")
    print(f"  Date range: {df['Date_Only'].min()} to {df['Date_Only'].max()}")
    print(f"  Sites: {sorted(df['Spot'].unique())}")
    print(f"{'='*60}")


# =============================================================================
# WATCHER MODE
# =============================================================================

def _run_watcher_mode():
    """Aggregate-aware watcher mode.

    Flow:
      1. Load birdnet_results aggregate (the dependency)
      2. Re-apply 3-step filtering to the FULL birdnet aggregate
         (filtering depends on global stats — can't be incremental)
      3. Save result as filtered_detections aggregate
      4. Output date-range-filtered subset to job output dir
    """
    import json
    args = config.parse_common_args(description="01 – Data Filtering (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load birdnet aggregate (dependency) ---
    # Try --aggregate-file first (watcher may pass the birdnet aggregate path),
    # then fall back to project DB resolution
    birdnet_aggregate_path = None
    if args.aggregate_file:
        # Watcher passes filtered_detections aggregate path, but we need birdnet's.
        # Look for birdnet_results.csv in same directory.
        agg_dir = os.path.dirname(args.aggregate_file)
        candidate = os.path.join(agg_dir, "birdnet_results.csv")
        if os.path.exists(candidate):
            birdnet_aggregate_path = candidate

    if not birdnet_aggregate_path:
        birdnet_aggregate_path = config.resolve_detection_csv(args)

    if not birdnet_aggregate_path or not os.path.exists(birdnet_aggregate_path):
        print("ERROR: No birdnet_results.csv aggregate found. Run 00b first.")
        sys.exit(1)

    print(f"Loading birdnet aggregate: {birdnet_aggregate_path}")
    raw_df = config.load_aggregate(birdnet_aggregate_path)

    # Strip _processed_only marker rows
    if "_processed_only" in raw_df.columns:
        raw_df = raw_df[raw_df["_processed_only"] != True].drop(columns=["_processed_only"])

    print(f"Total raw detections: {len(raw_df)}")

    if raw_df.empty:
        print("ERROR: Birdnet aggregate has no detection data.")
        sys.exit(1)

    # --- Apply filtering ---
    ebird = EBIRD_SPECIES_FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(ebird):
        ebird = os.path.join(script_dir, "Sanjay_Van_Birds.txt")

    filtered = filter_detections(raw_df, ebird_file=ebird)
    print_summary(filtered)

    # --- Save as filtered_detections aggregate ---
    aggregate_path = config.resolve_aggregate_path(args, "filtered_detections.csv")
    print(f"Writing filtered aggregate: {aggregate_path}")
    config._atomic_csv_write(filtered, aggregate_path)

    # --- Output date-filtered subset ---
    output_df = config.filter_aggregate_for_output(
        filtered,
        start_date=args.start_date,
        end_date=args.end_date,
        spots=args.spots,
    )

    if not output_df.empty:
        out_csv = os.path.join(args.output_dir, "filtered_detections.csv")
        output_df.to_csv(out_csv, index=False)
        print(f"Output {len(output_df)} filtered detections for requested range")
    else:
        print("WARNING: No detections after filtering for requested range / spots.")

    config.save_processed_list(args.output_dir, [os.path.basename(birdnet_aggregate_path)])


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    _run_watcher_mode()
