"""
Script 01: Data Loading and Filtering Pipeline
================================================
Loads BirdNET classification CSVs, applies multi-stage filtering:
  - Step 1: Taxonomic verification (eBird checklist)
  - Step 2: Confidence thresholding (>0.3)
  - Step 3: Minimum activity (>10 total detections)
  - Step 4: Temporal consistency (>=5 active days with >10 daily calls)

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
    """Apply 4-step filtering pipeline. Returns filtered DataFrame."""

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

    # -- Step 2: Confidence Thresholding (>0.3) --
    print("\nStep 2: Confidence thresholding (>0.3)...")
    before = len(results_df)
    results_df = results_df[results_df['confidence'] >= 0.3].copy()
    print(f"  Detections before: {before}, after: {len(results_df)}")

    # -- Step 3: Minimum Activity (>10 total detections) --
    print("\nStep 3: Minimum total detections (>10)...")
    species_counts = results_df.groupby('common_name').size()
    valid_species = species_counts[species_counts > 10].index
    before = results_df['common_name'].nunique()
    results_df = results_df[results_df['common_name'].isin(valid_species)].copy()
    print(f"  Species before: {before}, after: {results_df['common_name'].nunique()}")

    # -- Step 4: Temporal Consistency (>=5 days with >10 daily calls) --
    print("\nStep 4: Temporal consistency (>=5 active days)...")
    daily_counts = results_df.groupby(['common_name', 'Date_Only']).size().reset_index(name='daily_calls')
    high_activity_days = daily_counts[daily_counts['daily_calls'] >= 10]
    bird_day_counts = high_activity_days.groupby('common_name').size().reset_index(name='count_of_valid_days')
    valid_birds = bird_day_counts[bird_day_counts['count_of_valid_days'] >= 5]['common_name'].unique()
    before = results_df['common_name'].nunique()
    results_df = results_df[results_df['common_name'].isin(valid_birds)].copy()
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
    """Watcher passes --output-dir. Reads birdnet_results.csv from project DB."""
    import json
    args = config.parse_common_args(description="01 – Data Filtering (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    # Find input: birdnet_results.csv in project database
    detection_csv = config.resolve_detection_csv(args)
    if not detection_csv:
        print("ERROR: No birdnet_results.csv found. Run 00b first.")
        sys.exit(1)

    print(f"Loading detections from: {detection_csv}")
    results_df = pd.read_csv(detection_csv)
    print(f"Total raw detections loaded: {len(results_df)}")

    # Resolve eBird species file
    ebird = EBIRD_SPECIES_FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(ebird):
        ebird = os.path.join(script_dir, "Sanjay_Van_Birds.txt")

    filtered = filter_detections(results_df, ebird_file=ebird)
    print_summary(filtered)

    out_csv = os.path.join(args.output_dir, "filtered_detections.csv")
    filtered.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")

    # processed.json: just record the input file
    config.save_processed_list(args.output_dir, [os.path.basename(detection_csv)])


# =============================================================================
# STANDALONE MODE
# =============================================================================

def _run_standalone_mode():
    """Original mode: discover classification CSVs, concatenate, filter, save."""
    CLASSIFICATION_FILES = config.get_existing_classification_csvs()

    print("Loading classification files...")
    dfs = []
    for f in CLASSIFICATION_FILES:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
        else:
            print(f"  WARNING: File not found: {f}")

    if not dfs:
        raise FileNotFoundError("No classification files found. Run 00b first.")

    results_df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw detections loaded: {len(results_df)}")

    filtered = filter_detections(results_df, ebird_file=EBIRD_SPECIES_FILE)
    print_summary(filtered)

    filtered.to_csv("filtered_detections.csv", index=False)
    print(f"\nSaved to: filtered_detections.csv")


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================
if __name__ == "__main__":
    if "--output-dir" in sys.argv:
        _run_watcher_mode()
    else:
        _run_standalone_mode()
