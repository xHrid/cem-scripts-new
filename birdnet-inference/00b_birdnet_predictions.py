"""
Script 00b: BirdNET Species Classification from Raw Audio
==========================================================
Processes raw WAV recordings through:
  1. Static noise removal (SNR-scaled subtraction + spectral gating)
  2. Rain noise removal (same technique, different reference)
  3. BirdNET species identification via birdnetlib
  4. Export per-session classification CSV

Source: BirdNet_Predictions.ipynb
  - Cell 7: analyze_bird_audio(), remove_static_noise(), remove_rain_noise()
  - Cell 4: TFLite GPU delegate setup
  - Batch loop pattern from calculate_indices.ipynb Cell 8 (iterate WAVs in folder)

Output CSVs contain columns:
  filename, common_name, scientific_name, label, confidence, start_time, end_time, hour

These CSVs feed into Script 01 (filtering pipeline).

Performance notes (vs original notebook code):
  - BirdNET Analyzer loaded ONCE, reused across all files (was per-file = ~60s waste each)
  - RecordingBuffer used instead of temp-file + Recording (skips redundant librosa.load)
  - TFLite interpreter threads increased to match available CPU cores
  - Files within each session processed in parallel via ProcessPoolExecutor
  - Verbose birdnetlib console output suppressed for cleaner progress bars
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress verbose TF/birdnetlib logging BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')

# --- birdnetlib imports ---
from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

# --- Shared config ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_SR = config.TARGET_SR
SNR_DB = 18
STATIC_NOISE_PATH = config.STATIC_NOISE_PATH
RAIN_NOISE_PATH = config.RAIN_NOISE_PATH
OUTPUT_DIR = config.CLASSIFICATION_OUTPUT_DIR
LAT = config.LATITUDE
LON = config.LONGITUDE

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DENOISING FUNCTIONS (from BirdNet_Predictions.ipynb Cell 7 — exact copy)
# =============================================================================
def remove_static_noise(audio, noise_ref, sr=TARGET_SR, snr_db=SNR_DB):
    if len(noise_ref) > len(audio):
        noise_ref = noise_ref[:len(audio)]
    else:
        noise_ref = np.pad(noise_ref, (0, len(audio) - len(noise_ref)), 'wrap')
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_ref ** 2)
    desired_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_ref_scaled = noise_ref * np.sqrt(desired_noise_power / noise_power)
    audio_td = audio - noise_ref_scaled
    stft = librosa.stft(audio_td, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_stft = librosa.stft(noise_ref, n_fft=2048, hop_length=512)
    noise_mag = np.abs(noise_stft)
    noise_threshold = np.mean(noise_mag, axis=1, keepdims=True) * 1.2
    gated_mag = np.where(magnitude > noise_threshold, magnitude, 0)
    cleaned_stft = gated_mag * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=512)


def remove_rain_noise(audio, noise_ref, sr=TARGET_SR, snr_db=SNR_DB):
    if len(noise_ref) > len(audio):
        noise_ref = noise_ref[:len(audio)]
    else:
        noise_ref = np.pad(noise_ref, (0, len(audio) - len(noise_ref)), 'wrap')
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_ref ** 2)
    desired_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_ref_scaled = noise_ref * np.sqrt(desired_noise_power / noise_power)
    audio_td = audio - noise_ref_scaled
    stft = librosa.stft(audio_td, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_stft = librosa.stft(noise_ref, n_fft=2048, hop_length=512)
    noise_mag = np.abs(noise_stft)
    noise_threshold = np.mean(noise_mag, axis=1, keepdims=True) * 1.2
    gated_mag = np.where(magnitude > noise_threshold, magnitude, 0)
    cleaned_stft = gated_mag * np.exp(1j * phase)
    audio_cleaned = librosa.istft(cleaned_stft, hop_length=512)
    return audio_cleaned


# =============================================================================
# BirdNET ANALYSIS FUNCTION (from BirdNet_Predictions.ipynb Cell 7)
# =============================================================================
def analyze_bird_audio(audio_path, lat, lon, noise_clip, rain_noise_clip, analyzer=None):
    """Load audio, denoise (static + rain), run BirdNET, return detections DataFrame.

    Parameters
    ----------
    analyzer : birdnetlib.analyzer.Analyzer, optional
        Pre-loaded BirdNET analyzer instance.  When supplied the heavy
        model-load step (~60 s) is skipped.  **Always pass a shared
        instance from the calling loop.**

    Optimizations vs notebook code:
      - Uses RecordingBuffer to pass denoised audio as numpy array directly,
        avoiding the temp-file write + redundant librosa.load() round-trip
        that doubled I/O time for every file.
    """
    audio_raw, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != TARGET_SR:
        audio_raw = librosa.resample(y=audio_raw, orig_sr=orig_sr, target_sr=TARGET_SR)

    final_sound_temp = remove_static_noise(audio_raw, noise_clip, sr=TARGET_SR, snr_db=SNR_DB)
    final_sound = remove_rain_noise(final_sound_temp, rain_noise_clip, sr=TARGET_SR, snr_db=SNR_DB)

    # Use RecordingBuffer to pass the denoised numpy array directly to BirdNET.
    # This skips: temp-file write (sf.write) → temp-file read (librosa.load inside Recording).
    if analyzer is None:
        analyzer = Analyzer()
    recording = RecordingBuffer(
        analyzer,
        final_sound,
        TARGET_SR,
        lat=lat,
        lon=lon,
    )
    # Suppress birdnetlib's per-chunk/per-species verbose prints
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        recording.analyze()
    return pd.DataFrame(recording.detections)


def _process_single_file(args):
    """Worker function for parallel processing. Each worker loads its own Analyzer
    (unavoidable — TFLite interpreters aren't picklable across processes) but
    processes many files with it."""
    filepath, filename, lat, lon, noise_clip, rain_noise_clip, hour = args
    try:
        # Each worker gets a thread-local analyzer (loaded once per worker via initializer)
        analyzer = _get_worker_analyzer()
        detections_df = analyze_bird_audio(filepath, lat, lon, noise_clip, rain_noise_clip, analyzer=analyzer)
        if not detections_df.empty:
            detections_df['filename'] = filename
            detections_df['hour'] = hour
            return detections_df
    except Exception as e:
        print(f"\n  ERROR processing {filename}: {e}")
    return None


# --- Worker-local Analyzer (one per process, reused across files) ---
_worker_analyzer = None

def _init_worker(num_threads):
    """Called once per worker process to create a long-lived Analyzer."""
    global _worker_analyzer
    # Suppress all TF warnings + birdnetlib prints in workers
    import io, contextlib, warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _worker_analyzer = Analyzer()
    # Increase TFLite threads for this worker
    if num_threads > 1:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _worker_analyzer.interpreter = type(_worker_analyzer.interpreter)(
                    model_path=_worker_analyzer.model_path, num_threads=num_threads
                )
            _worker_analyzer.interpreter.allocate_tensors()
            _worker_analyzer.input_details = _worker_analyzer.interpreter.get_input_details()
            _worker_analyzer.output_details = _worker_analyzer.interpreter.get_output_details()
            _worker_analyzer.input_layer_index = _worker_analyzer.input_details[0]["index"]
            _worker_analyzer.output_layer_index = _worker_analyzer.output_details[0]["index"]
        except Exception:
            pass  # Fall back to default 1-thread if re-init fails

def _get_worker_analyzer():
    """Return the process-local Analyzer."""
    return _worker_analyzer


# =============================================================================
# DATETIME EXTRACTION (from calculate_indices.ipynb Cell 2)
# =============================================================================
def extract_hour_from_filename(filename):
    """Extract hour from filename pattern: ..._YYYYMMDD_HHMMSS.wav"""
    match = re.search(r'_(\d{6})\.wav$', filename)
    if match:
        return int(match.group(1)[:2])
    return None


def extract_path_info(dataset_path):
    """Extract spot name, date range, and duty cycle code from path.
    Source: calculate_indices.ipynb Cell 5"""
    parts = dataset_path.replace('\\', '/').split('/')
    code = parts[-1] if len(parts) > 0 else ""
    date_range = parts[-2] if len(parts) > 1 else ""
    spot_folder = parts[-3] if len(parts) > 2 else ""
    spot_match = re.search(r'spot[_\s]*(\d+)', spot_folder, re.IGNORECASE)
    spot = f"spot_{spot_match.group(1)}" if spot_match else "spot_unknown"
    return spot, date_range, code


# =============================================================================
# MAIN PROCESSING LOOP
# (Batch pattern from calculate_indices.ipynb Cell 8, calling analyze_bird_audio
#  from BirdNet_Predictions.ipynb Cell 7)
# =============================================================================
def _run_watcher_mode():
    """Entry point when launched by the watcher with CLI args."""
    import multiprocessing

    args = config.parse_common_args("BirdNET species classification from raw audio")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Resolve noise files
    noise_path = config.resolve_noise_path(args)
    if not noise_path:
        print("ERROR: No static_noise.wav found. Cannot denoise.")
        sys.exit(1)

    rain_noise_path = RAIN_NOISE_PATH
    if args.noise_path:
        # Derive rain noise path from same directory as static noise
        rain_candidate = os.path.join(os.path.dirname(args.noise_path), "rain_noise.wav")
        if os.path.exists(rain_candidate):
            rain_noise_path = rain_candidate

    print("Loading noise reference clips...")
    noise_clip, _ = librosa.load(noise_path, sr=TARGET_SR)
    rain_noise_clip, _ = librosa.load(rain_noise_path, sr=TARGET_SR)

    # Skip-list: files already processed in previous jobs
    skip_set = config.load_skip_list(args.skip_list)
    print(f"Skip-list: {len(skip_set)} already-processed files")

    # Date filter
    start_val = int(args.start_date) if args.start_date else None
    end_val = int(args.end_date) if args.end_date else None

    # Parallelism
    total_cpus = multiprocessing.cpu_count()
    N_WORKERS = max(1, min(total_cpus // 2, 4))
    THREADS_PER_WORKER = max(1, total_cpus // N_WORKERS)
    print(f"Parallelism: {N_WORKERS} workers × {THREADS_PER_WORKER} TFLite threads "
          f"(detected {total_cpus} logical CPUs)")

    all_detections = []
    processed_files = []

    for dataset_dir in args.datasets:
        if not os.path.isdir(dataset_dir):
            print(f"WARNING: Dataset dir not found: {dataset_dir}")
            continue

        wav_files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith('.wav')])
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_dir} ({len(wav_files)} WAV files)")

        # Build task list, filtering by skip-list and date range
        tasks = []
        for filename in wav_files:
            if filename in skip_set:
                continue
            if not config.filter_by_date(filename, start_val, end_val):
                continue
            filepath = os.path.join(dataset_dir, filename)
            hour = config.extract_hour_from_filename(filename)
            tasks.append((filepath, filename, args.lat, args.lon,
                          noise_clip, rain_noise_clip, hour))

        if not tasks:
            print("  All files skipped (cached or out of date range)")
            continue

        print(f"  Processing {len(tasks)} files (skipped {len(wav_files) - len(tasks)})")

        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_worker,
            initargs=(THREADS_PER_WORKER,),
        ) as executor:
            futures = {executor.submit(_process_single_file, t): t[1] for t in tasks}
            with tqdm(total=len(tasks), desc=f"  BirdNET") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_detections.append(result)
                    pbar.update(1)

        # Track ALL wav files in this dir as processed (including ones that
        # produced no detections — so we don't re-scan them next time)
        processed_files.extend([t[1] for t in tasks])

    # Combine and save
    if all_detections:
        results_df = pd.concat(all_detections, ignore_index=True)

        if 'common_name' in results_df.columns and 'label' not in results_df.columns:
            results_df['label'] = results_df['common_name']

        output_csv = os.path.join(output_dir, "birdnet_results.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(results_df)} detections to: {output_csv}")
    else:
        print("\nWARNING: No detections found across all datasets.")

    # Write processed.json for watcher cache update
    config.save_processed_list(output_dir, processed_files)
    print(f"Processed {len(processed_files)} files total.")


def _run_standalone_mode():
    """Original standalone entry point — walks folder structure from 00_config."""
    import multiprocessing

    print("Loading noise reference clips...")
    noise_clip, _ = librosa.load(STATIC_NOISE_PATH, sr=TARGET_SR)
    rain_noise_clip, _ = librosa.load(RAIN_NOISE_PATH, sr=TARGET_SR)

    total_cpus = multiprocessing.cpu_count()
    N_WORKERS = max(1, min(total_cpus // 2, 4))
    THREADS_PER_WORKER = max(1, total_cpus // N_WORKERS)
    print(f"Parallelism: {N_WORKERS} workers × {THREADS_PER_WORKER} TFLite threads "
          f"(detected {total_cpus} logical CPUs)")

    audio_folders = config.discover_audio_folders()
    total_folders = sum(len(v) for v in audio_folders.values())
    print(f"\nFound {total_folders} recording sessions across {len(audio_folders)} spots")

    for spot_key, folder_list in audio_folders.items():
        for entry in folder_list:
            dataset_path = entry["path"]
            date_range = entry["date_range"]
            duty_cycle = entry["duty_cycle"]

            output_filename = f"{spot_key}_{date_range}_{duty_cycle}_classification.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            if os.path.exists(output_path):
                print(f"\nSKIPPING (already exists): {output_filename}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {spot_key} / {date_range} / {duty_cycle}")
            print(f"  Path: {dataset_path}")

            wav_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith('.wav')])
            tasks = []
            for filename in wav_files:
                filepath = os.path.join(dataset_path, filename)
                hour = extract_hour_from_filename(filename)
                tasks.append((filepath, filename, LAT, LON, noise_clip, rain_noise_clip, hour))

            all_detections = []
            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_worker,
                initargs=(THREADS_PER_WORKER,),
            ) as executor:
                futures = {executor.submit(_process_single_file, t): t[1] for t in tasks}
                with tqdm(total=len(tasks), desc=f"  {spot_key}/{date_range}") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            all_detections.append(result)
                        pbar.update(1)

            if all_detections:
                results_df = pd.concat(all_detections, ignore_index=True)
                if 'common_name' in results_df.columns and 'label' not in results_df.columns:
                    results_df['label'] = results_df['common_name']
                results_df.to_csv(output_path, index=False)
                print(f"  Saved {len(results_df)} detections to: {output_filename}")
            else:
                print(f"  WARNING: No detections found.")

    print(f"\n{'='*60}")
    print("BirdNET prediction pipeline complete.")
    print(f"Classification CSVs saved to: {OUTPUT_DIR}")
    existing = config.get_existing_classification_csvs()
    print(f"Total classification CSVs: {len(existing)}")


if __name__ == "__main__":
    # Detect mode: if --output-dir is in sys.argv, watcher launched us
    if "--output-dir" in sys.argv:
        _run_watcher_mode()
    else:
        _run_standalone_mode()
