"""
Shared Configuration for Acoustic Analysis Scripts (Webapp)
============================================================
Runs exclusively via the watcher/webapp CLI pipeline.
The watcher passes --datasets, --output-dir, --project-dir, etc.
Scripts call parse_common_args() to get CLI values. Dataset dirs
are passed directly — no folder walking needed.
"""

import argparse
import os
import re
import json
import shutil
import pandas as pd
import librosa

# Asset files (resolved relative to this script's directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_NOISE_PATH = os.path.join(_SCRIPT_DIR, "static_noise.wav")
RAIN_NOISE_PATH = os.path.join(_SCRIPT_DIR, "rain_noise.wav")
RAINFALL_MODEL_PATH = os.path.join(_SCRIPT_DIR, "rainfall_model.joblib")
RAINFALL_ENCODER_PATH = os.path.join(_SCRIPT_DIR, "label_encoder.joblib")
EBIRD_SPECIES_FILE = os.path.join(_SCRIPT_DIR, "Sanjay_Van_Birds.txt")

# Coordinates
LATITUDE = 28.53
LONGITUDE = 77.18
TIMEZONE_STR = "Asia/Kolkata"

# BirdNET model (None = bundled)
BIRDNET_MODEL_PATH = None

# Audio
TARGET_SR = 48000

# =============================================================================
# CLI ARG PARSER
# =============================================================================

def parse_common_args(description="Analysis script"):
    """Parse CLI args that the watcher passes to every script.

    Returns argparse.Namespace with:
        datasets    : list[str]   — absolute paths to audio directories
        output_dir  : str         — where to write results
        root_dir    : str         — watcher root path
        project_dir : str         — project folder path
        noise_path  : str         — static noise WAV path
        skip_list   : str         — path to skip-list JSON
        spots       : str         — comma-separated spot names
        start_date  : str         — YYYYMMDD
        end_date    : str         — YYYYMMDD
        detection_csv : str       — path to input detections CSV (for downstream scripts)
        lat, lon    : float       — coordinates
    """
    parser = argparse.ArgumentParser(description=description)

    # Watcher-provided paths
    parser.add_argument("--datasets", nargs='*', default=[],
                        help="Directories containing WAV files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output files")
    parser.add_argument("--root-dir", type=str, default="",
                        help="Watcher root path")
    parser.add_argument("--project-dir", type=str, default="",
                        help="Project folder path")
    parser.add_argument("--noise-path", type=str, default="",
                        help="Path to static_noise.wav")
    parser.add_argument("--aggregate-file", type=str, default="",
                        help="Path to persistent aggregate CSV (scripts own read/write)")
    # Legacy support — ignored if --aggregate-file is set
    parser.add_argument("--skip-list", type=str, default="",
                        help="(DEPRECATED) Path to skip-list JSON")

    # Filtering
    parser.add_argument("--spots", type=str, default="",
                        help="Comma-separated spot names for filtering")
    parser.add_argument("--start-date", type=str, default="",
                        help="Start date YYYYMMDD")
    parser.add_argument("--end-date", type=str, default="",
                        help="End date YYYYMMDD")

    # Optional overrides
    parser.add_argument("--detection-csv", type=str, default=None,
                        help="Path to detections CSV (auto-resolved from project-dir if omitted)")
    parser.add_argument("--indices-csv", type=str, default=None,
                        help="Path to acoustic indices CSV")
    parser.add_argument("--lat", type=float, default=LATITUDE)
    parser.add_argument("--lon", type=float, default=LONGITUDE)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SR)
    parser.add_argument("--snr-db", type=int, default=18)

    return parser.parse_args()


def load_skip_list(skip_list_path):
    """Load already-processed filenames from skip-list JSON."""
    if not skip_list_path or not os.path.exists(skip_list_path):
        return set()
    try:
        with open(skip_list_path, "r") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError):
        return set()


def save_processed_list(output_dir, processed_files):
    """Write processed.json so watcher can update its cache."""
    path = os.path.join(output_dir, "processed.json")
    with open(path, "w") as f:
        json.dump(processed_files, f)


def resolve_detection_csv(args):
    """Find the master detections CSV. Priority: CLI arg > project DB."""
    if args.detection_csv and os.path.exists(args.detection_csv):
        return args.detection_csv

    if args.project_dir:
        db_path = os.path.join(args.project_dir, "system", "database",
                               "filtered_detections.csv")
        if os.path.exists(db_path):
            return db_path
        # Fall back to unfiltered birdnet results
        db_path2 = os.path.join(args.project_dir, "system", "database",
                                "birdnet_results.csv")
        if os.path.exists(db_path2):
            return db_path2

    return None


def resolve_indices_csv(args):
    """Find the master acoustic indices CSV."""
    if args.indices_csv and os.path.exists(args.indices_csv):
        return args.indices_csv

    if args.project_dir:
        db_path = os.path.join(args.project_dir, "system", "database",
                               "acoustic_indices.csv")
        if os.path.exists(db_path):
            return db_path

    return None


def resolve_noise_path(args):
    """Resolve noise WAV path from CLI arg or default."""
    if args.noise_path and os.path.exists(args.noise_path):
        return args.noise_path
    if os.path.exists(STATIC_NOISE_PATH):
        return STATIC_NOISE_PATH
    return None


def filter_by_date(filename, start_val, end_val):
    """Check if filename's embedded date is within range. None bounds = no limit."""
    match = re.search(r'_(\d{8})_', filename)
    if not match:
        return True  # no date in filename = include
    file_date = int(match.group(1))
    if start_val and file_date < start_val:
        return False
    if end_val and file_date > end_val:
        return False
    return True


# =============================================================================
# AGGREGATE FILE MANAGEMENT
# =============================================================================
# Aggregate CSVs are the persistent result store. They grow over time as new
# audio is processed. A special column `_processed_only` (bool) marks files
# that were processed but produced no detections — so we never re-scan them.
#
# Flow:
#   1. load_aggregate(path) → DataFrame (may be empty)
#   2. get_processed_filenames(df) → set of all filenames already handled
#   3. Script processes only NEW files
#   4. append_to_aggregate(new_df, path) — merges new results + saves
#   5. mark_empty_files(filenames, path) — stamps no-detection files
#   6. filter_aggregate_for_output(df, start, end, spots) → clean subset
# =============================================================================

# Sentinel column name for "processed but empty" marker rows
_PROCESSED_ONLY_COL = "_processed_only"


def load_aggregate(aggregate_path):
    """Load an existing aggregate CSV. Returns empty DataFrame if not found."""
    if not aggregate_path or not os.path.exists(aggregate_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(aggregate_path)
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def get_processed_filenames(aggregate_df):
    """Extract the set of all filenames present in the aggregate (data + empty markers)."""
    if aggregate_df.empty or "filename" not in aggregate_df.columns:
        return set()
    return set(aggregate_df["filename"].dropna().unique())


def append_to_aggregate(new_df, aggregate_path):
    """Append new result rows to the aggregate CSV. Creates file if absent.

    Handles column alignment: if aggregate has extra columns (e.g. _processed_only),
    new_df rows get NaN for those. Vice versa columns are added.
    """
    if new_df.empty:
        return

    os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)

    existing = load_aggregate(aggregate_path)
    if existing.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    _atomic_csv_write(combined, aggregate_path)


def mark_empty_files(filenames, aggregate_path):
    """Add marker rows for files that produced zero detections.

    These rows have _processed_only=True and only the filename column filled.
    Prevents re-processing files that legitimately have no bird calls.
    """
    if not filenames:
        return

    existing = load_aggregate(aggregate_path)
    already_marked = get_processed_filenames(existing)
    new_empties = [f for f in filenames if f not in already_marked]

    if not new_empties:
        return

    marker_rows = pd.DataFrame({
        "filename": new_empties,
        _PROCESSED_ONLY_COL: True,
    })

    os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)

    if existing.empty:
        combined = marker_rows
    else:
        combined = pd.concat([existing, marker_rows], ignore_index=True)

    _atomic_csv_write(combined, aggregate_path)


def filter_aggregate_for_output(aggregate_df, start_date="", end_date="", spots=""):
    """Filter aggregate to user-requested date range and spots. Strips marker rows.

    Args:
        aggregate_df: Full aggregate DataFrame.
        start_date: YYYYMMDD string (inclusive). Empty = no lower bound.
        end_date: YYYYMMDD string (inclusive). Empty = no upper bound.
        spots: Comma-separated spot names. Empty = all spots.

    Returns:
        Cleaned DataFrame with only real data rows matching the filters.
    """
    if aggregate_df.empty:
        return aggregate_df

    # Strip processed-only marker rows
    df = aggregate_df.copy()
    if _PROCESSED_ONLY_COL in df.columns:
        df = df[df[_PROCESSED_ONLY_COL] != True].drop(columns=[_PROCESSED_ONLY_COL])

    if df.empty:
        return df

    # Date filtering from filename
    if start_date or end_date:
        start_val = int(start_date) if start_date else None
        end_val = int(end_date) if end_date else None
        mask = df["filename"].apply(lambda f: filter_by_date(f, start_val, end_val))
        df = df[mask]

    # Spot filtering
    if spots:
        spot_list = [s.strip().upper() for s in spots.split(",")]
        if "spot" in df.columns:
            df = df[df["spot"].str.upper().isin(spot_list)]
        elif "Spot" in df.columns:
            df = df[df["Spot"].str.upper().isin(spot_list)]

    return df.reset_index(drop=True)


def _atomic_csv_write(df, path):
    """Write CSV atomically via temp file + rename. Prevents corruption on crash."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    shutil.move(tmp_path, path)


def resolve_aggregate_path(args, fallback_name="aggregate.csv"):
    """Resolve the aggregate file path from CLI args or project database dir.

    Priority: --aggregate-file > project_dir/system/database/<fallback_name>
    """
    if args.aggregate_file and args.aggregate_file.strip():
        return args.aggregate_file

    if args.project_dir:
        db_dir = os.path.join(args.project_dir, "system", "database")
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, fallback_name)

    return os.path.join(args.output_dir, fallback_name)


# =============================================================================
# DURATION-BASED SEGMENTATION (replaces folder_type / duty_cycle)
# =============================================================================

def infer_duty_cycle_from_duration(audio_length_samples, sr=TARGET_SR):
    """Infer recording schedule from file duration.

    Returns a string compatible with the old segment_audio() interface:
        "2R4W"  — file ≤ 150s (2.5 min)
        "5R5W"  — file 150–600s
        "30R30W" — file > 600s (10 min)
    """
    duration_sec = audio_length_samples / sr
    if duration_sec <= 150:
        return "2R4W"
    elif duration_sec <= 600:
        return "5R5W"
    else:
        return "30R30W"


def segment_audio(audio, sr=TARGET_SR, duty_cycle=None):
    """Segment audio based on duty cycle (auto-inferred from duration if not given).

    Returns list of 2-min numpy segments, or None if audio too short.
    """
    two_min_samples = int(120 * sr)

    if duty_cycle is None:
        duty_cycle = infer_duty_cycle_from_duration(len(audio), sr)

    segments = []

    if "2R4W" in duty_cycle:
        if len(audio) >= two_min_samples:
            segments.append(audio[:two_min_samples])
        elif len(audio) > 0:
            segments.append(audio)  # short file, take what we have

    elif "5R5W" in duty_cycle:
        if len(audio) >= two_min_samples:
            segments.append(audio[:two_min_samples])

    elif "30R30W" in duty_cycle:
        num_chunks = 10
        for start in range(0, len(audio), two_min_samples):
            end = start + two_min_samples
            if end <= len(audio):
                segments.append(audio[start:end])
            if len(segments) >= num_chunks:
                break

    return segments if segments else None


# =============================================================================
# FILENAME METADATA EXTRACTION
# =============================================================================

def extract_hour_from_filename(filename):
    """Extract hour from filename pattern: ..._YYYYMMDD_HHMMSS.wav"""
    match = re.search(r'_(\d{6})\.wav$', filename)
    if match:
        return int(match.group(1)[:2])
    return None


def extract_datetime_from_filename(filename):
    """Extract (year, month, date, hour, minute) from filename."""
    match_date = re.search(r'_(\d{8})_', filename)
    match_time = re.search(r'_(\d{6})\.wav$', filename)
    if match_time and match_date:
        date_str = match_date.group(1)
        time_str = match_time.group(1)
        return date_str[:4], date_str[4:6], date_str[6:], int(time_str[:2]), int(time_str[2:4])
    return None, None, None, None, None


def extract_spot_from_filename(filename):
    """Extract spot identifier from filename (e.g., SPOT1 → spot_1)."""
    match = re.search(r'(SPOT\d+)', filename, re.IGNORECASE)
    if match:
        raw = match.group(1).upper()
        num = re.search(r'\d+', raw).group()
        return f"spot_{num}"
    return None


# =============================================================================
# INLINE DATA FILTERING (replaces standalone 01_data_loading_and_filtering.py)
# =============================================================================

def filter_detections(results_df, ebird_file=None):
    """Apply 3-step filtering pipeline to BirdNET detections.

    Steps:
      1. Taxonomic verification (eBird checklist)
      2. Confidence thresholding (>=0.3)
      3. Minimum activity (>=10 total detections)

    Returns filtered DataFrame with Spot, Date, Date_Only columns added.
    """
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


def load_filtered_detections(args):
    """Load birdnet aggregate and apply inline filtering.

    This replaces the standalone data-filtering script. Downstream scripts
    call this directly to get filtered detections without an intermediate CSV.

    Args:
        args: Namespace from parse_common_args() with aggregate_file, project_dir,
              start_date, end_date, spots.

    Returns:
        Filtered DataFrame ready for analysis, or empty DataFrame on failure.
    """
    # --- Locate birdnet_results.csv aggregate ---
    birdnet_path = None
    if args.aggregate_file:
        agg_dir = os.path.dirname(args.aggregate_file)
        candidate = os.path.join(agg_dir, "birdnet_results.csv")
        if os.path.exists(candidate):
            birdnet_path = candidate

    if not birdnet_path:
        # Try project database
        if args.project_dir:
            db_path = os.path.join(args.project_dir, "system", "database",
                                   "birdnet_results.csv")
            if os.path.exists(db_path):
                birdnet_path = db_path

    if not birdnet_path or not os.path.exists(birdnet_path):
        print("ERROR: No birdnet_results.csv found. Run BirdNET inference first.",
              file=sys.stderr)
        return pd.DataFrame()

    print(f"Loading birdnet aggregate: {birdnet_path}")
    raw_df = load_aggregate(birdnet_path)

    # Strip _processed_only marker rows
    if _PROCESSED_ONLY_COL in raw_df.columns:
        raw_df = raw_df[raw_df[_PROCESSED_ONLY_COL] != True].drop(columns=[_PROCESSED_ONLY_COL])

    if raw_df.empty:
        print("ERROR: Birdnet aggregate has no detection data.",
              file=sys.stderr)
        return pd.DataFrame()

    print(f"Total raw detections: {len(raw_df)}")

    # --- Apply 3-step filtering ---
    ebird_file = EBIRD_SPECIES_FILE
    if not os.path.exists(ebird_file):
        # Try script directory fallback
        ebird_file = os.path.join(_SCRIPT_DIR, "Sanjay_Van_Birds.txt")

    filtered = filter_detections(raw_df, ebird_file=ebird_file)

    if filtered.empty:
        print("WARNING: No detections survived filtering.")
        return pd.DataFrame()

    # --- Apply date range and spot filters ---
    if args.start_date or args.end_date or args.spots:
        filtered = filter_aggregate_for_output(
            filtered, args.start_date, args.end_date, args.spots
        )

    print(f"Final
 filtered detections: {len(filtered)} detections, {filtered['common_name'].nunique()} species")
    return filtered


# =============================================================================
# V2 BACKWARD COMPATIBILITY — old-style constants & filter wrapper
# =============================================================================
# These mirror the original v2 config.py so existing scripts work unchanged.
# For new webapp pipeline usage, prefer parse_common_args() + CLI args.

from datetime import date as _date

# --- INPUTS (change per run) ---
SPOT_NAMES: list = []
DATE_START  = _date(2025, 11, 1)
DATE_END    = _date(2025, 12, 31)
INPUT_FILE_LIST: list = []

# --- PATHS (set once for your environment) ---
INPUT_DIRECTORIES: list = [
    r"path/to/audio_dir_1",
    r"path/to/audio_dir_2",
]

AGGREGATE_FILE     = r"path/to/birdnet_aggregate.csv"
PROCESSED_FILE     = r"path/to/processed_files.txt"
OUTPUT_CSV         = r"path/to/birdnet_output.csv"

EBIRD_FILE         = EBIRD_SPECIES_FILE  # alias to new name

OUTPUT_DIR_02_HEATMAPS   = r"path/to/output/02_heatmaps"
OUTPUT_DIR_03_TEMPORAL    = r"path/to/output/03_temporal_stickiness"
OUTPUT_DIR_04_SPATIAL     = r"path/to/output/04_spatial_stickiness"
OUTPUT_DIR_07_MIGRATORY   = r"path/to/output/07_migratory"
OUTPUT_DIR_08_SOLAR       = r"path/to/output/08_solar"
OUTPUT_DIR_09_TIMESERIES  = r"path/to/output/09_timeseries"

# --- CONSTANTS ---
SNR_DB              = 18.0
LOCATION_NAME       = "Sanjay Van"
MIN_CONFIDENCE      = 0.25
TOP_N_SPECIES       = 25
TOP_N_TEMPORAL      = 80
SCI_THRESHOLD       = 0.9
KURTOSIS_THRESHOLD  = 15.0
PMR_THRESHOLD       = 50.0
WINDOW_SIZE         = 60
EPSILON             = 1e-6
MIN_SOLAR_DAYS      = 5
MAX_TIMESERIES_SP   = 50
SPECIES_TO_PLOT     = None


# --- Old-style filter_detections wrapper (path-based) ---
_original_filter_detections = filter_detections

def _legacy_filter_detections(aggregate_path, ebird_file=None,
                              date_start=None, date_end=None,
                              spot_names=None):
    """Backward-compatible wrapper: accepts CSV path, returns filtered DataFrame."""
    df = pd.read_csv(aggregate_path)
    print(f"Loaded aggregate: {len(df)} rows")
    filtered = _original_filter_detections(df, ebird_file=ebird_file)

    # Date range filter
    if "Date_Only" in filtered.columns:
        if date_start is not None:
            filtered = filtered[filtered["Date_Only"] >= date_start]
        if date_end is not None:
            filtered = filtered[filtered["Date_Only"] <= date_end]

    # Spot filter
    if spot_names:
        filtered = filtered[filtered["Spot"].isin([s.lower() for s in spot_names])]

    print(f"Final: {len(filtered)} detections, {filtered['common_name'].nunique()} species")
    return filtered

# Override with legacy wrapper so v2 scripts work unchanged
filter_detections = _legacy_filter_detections

