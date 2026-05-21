"""
Shared Configuration for All Acoustic Analysis Scripts
========================================================
Dual-mode: works standalone (folder walking) AND with watcher (CLI args).

Standalone mode:
    Edit AUDIO_ROOT and output dirs below. Scripts import and use
    discover_audio_folders() to find WAV files.

Watcher mode:
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

# =============================================================================
# ROOT PATHS — Used in STANDALONE mode. Ignored when watcher passes CLI args.
# =============================================================================

AUDIO_ROOT = r"E:\Sanjay_van_data\raw_data\audio_raw"
CLASSIFICATION_OUTPUT_DIR = r"E:\Sanjay_van_data\analysis-pipeline-hridayansh\results"
INDICES_OUTPUT_DIR = r"E:\Sanjay_van_data\analysis-pipeline-hridayansh\results"

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
# STANDALONE: Folder discovery (spot → date_range → duty_cycle → wavs)
# =============================================================================

SPOT_DIRS = {
    "spot_1": os.path.join(AUDIO_ROOT, "spot_1_original_spot"),
    "spot_2": os.path.join(AUDIO_ROOT, "spot_2_peacock_spot"),
    "spot_3": os.path.join(AUDIO_ROOT, "spot_3_investigation_spot"),
    "spot_4": os.path.join(AUDIO_ROOT, "spot_4_yoga_spot"),
}


def discover_audio_folders():
    """Walk the audio root and find all date_range/duty_cycle folders with .wav files."""
    folders = {}
    for spot_key, spot_dir in SPOT_DIRS.items():
        folders[spot_key] = []
        if not os.path.isdir(spot_dir):
            print(f"  WARNING: {spot_dir} not found")
            continue
        for date_range in sorted(os.listdir(spot_dir)):
            date_path = os.path.join(spot_dir, date_range)
            if not os.path.isdir(date_path):
                continue
            found_sub = False
            for sub in os.listdir(date_path):
                sub_path = os.path.join(date_path, sub)
                if os.path.isdir(sub_path):
                    wavs = [f for f in os.listdir(sub_path) if f.lower().endswith('.wav')]
                    if wavs:
                        folders[spot_key].append({
                            "date_range": date_range,
                            "duty_cycle": sub,
                            "path": sub_path,
                            "wav_count": len(wavs)
                        })
                        found_sub = True
            if not found_sub:
                wavs = [f for f in os.listdir(date_path) if f.lower().endswith('.wav')]
                if wavs:
                    folders[spot_key].append({
                        "date_range": date_range,
                        "duty_cycle": "unknown",
                        "path": date_path,
                        "wav_count": len(wavs)
                    })
    return folders


def get_all_audio_paths():
    """Return flat list of all audio folder paths across all spots."""
    folders = discover_audio_folders()
    return [entry["path"] for fl in folders.values() for entry in fl]


def get_classification_csv_path(spot_key, date_range, duty_cycle):
    """Generate expected classification CSV filename for a given recording session."""
    return os.path.join(
        CLASSIFICATION_OUTPUT_DIR,
        f"{spot_key}_{date_range}_{duty_cycle}_classification.csv"
    )


def get_all_classification_csvs():
    """Return list of all expected classification CSV paths."""
    folders = discover_audio_folders()
    csvs = []
    for spot_key, folder_list in folders.items():
        for entry in folder_list:
            csvs.append(get_classification_csv_path(
                spot_key, entry["date_range"], entry["duty_cycle"]
            ))
    return csvs


def get_existing_classification_csvs():
    """Return only classification CSVs that actually exist on disk."""
    return [p for p in get_all_classification_csvs() if os.path.exists(p)]


# =============================================================================
# WATCHER INTEGRATION: Common CLI arg parser
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
    parser.add_argument("--skip-list", type=str, default="",
                        help="Path to JSON list of already-processed filenames")

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
    """Find the master detections CSV. Priority: CLI arg > project DB > CWD."""
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

    # Standalone fallback
    if os.path.exists("filtered_detections.csv"):
        return "filtered_detections.csv"

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


# =============================================================================
# AGGREGATE HELPERS (for Tier 3 scripts consuming upstream aggregates)
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


def _atomic_csv_write(df, path):
    """Write CSV atomically via temp file + rename. Prevents corruption on crash."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    shutil.move(tmp_path, path)


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
            segments.append(audio