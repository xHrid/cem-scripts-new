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
# QUICK SUMMARY (run this file directly to verify setup)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO DATA STRUCTURE DISCOVERY")
    print("=" * 60)
    folders = discover_audio_folders()
    total_wavs = 0
    for spot_key, folder_list in folders.items():
        print(f"\n{spot_key}:")
        for entry in folder_list:
            print(f"  {entry['date_range']}/{entry['duty_cycle']}: "
                  f"{entry['wav_count']} wav files")
            total_wavs += entry["wav_count"]
    print(f"\nTotal audio folders: {sum(len(v) for v in folders.values())}")
    print(f"Total wav files: {total_wavs}")

    print(f"\n{'=' * 60}")
    print("EXISTING CLASSIFICATION CSVs")
    print("=" * 60)
    existing = get_existing_classification_csvs()
    if existing:
        for p in existing:
            print(f"  {os.path.basename(p)}")
    else:
        print("  None found yet. Run Script 00b first.")

    print(f"\nConfig OK. Ready to run pipeline.")
