"""
Script 05: Acoustic Indices Computation
=========================================
Computes 6 acoustic indices from raw audio recordings:
  - ADI (Acoustic Diversity Index): Shannon entropy of frequency band energy
  - ACI (Acoustic Complexity Index): Mean normalized spectral amplitude difference
  - AEI (Acoustic Evenness Index): 1 - Gini coefficient of frequency energy
  - NDSI (Normalized Difference Soundscape Index): (biophony - anthrophony) ratio
  - MFC (Mid-Frequency Cover): Fraction of frames with dominant mid-band energy
  - CLS (Cluster Count): Mean peak count per spectral frame

Pipeline:
  1. Load audio → denoise (static noise removal) → segment
  2. Rainfall classification → exclude heavy rain segments
  3. Compute indices per segment
  4. Export CSV

Paper Reference: Section 3.2.2 (index formulas), Section 4.3.3 (results),
                 Figure 17 (NDSI box plots)
"""

import os
import re
import numpy as np
import pandas as pd
import librosa
import joblib
from scipy.signal import spectrogram, find_peaks
from scipy.stats import entropy
from tqdm import tqdm

# =============================================================================
# CONFIGURATION (imported from 00_config.py)
# =============================================================================
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

DATASET_PATHS = config.get_all_audio_paths()
STATIC_NOISE_PATH = config.STATIC_NOISE_PATH
TARGET_SR = config.TARGET_SR
OUTPUT_DIR = config.INDICES_OUTPUT_DIR
MODEL_PATH = config.RAINFALL_MODEL_PATH
ENCODER_PATH = config.RAINFALL_ENCODER_PATH

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_datetime_from_filename(filename):
    """Extract year, month, date, hour, minute from filename pattern: ..._YYYYMMDD_HHMMSS.wav"""
    match_date = re.search(r'_(\d{8})_', filename)
    match_time = re.search(r'_(\d{6})\.wav$', filename)
    if match_time and match_date:
        date_str = match_date.group(1)
        time_str = match_time.group(1)
        return date_str[:4], date_str[4:6], date_str[6:], int(time_str[:2]), int(time_str[2:4])
    return None, None, None, None, None


def remove_static_noise(audio, noise_ref, sr=TARGET_SR, snr_db=18):
    """Remove stationary background noise using SNR-scaled subtraction + spectral gating."""
    if len(noise_ref) > len(audio):
        noise_ref = noise_ref[:len(audio)]
    else:
        noise_ref = np.pad(noise_ref, (0, len(audio) - len(noise_ref)), 'wrap')

    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_ref ** 2)
    if noise_power == 0:
        return audio

    desired_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_ref_scaled = noise_ref * np.sqrt(desired_noise_power / noise_power)

    # Time-domain subtraction
    audio_td = audio - noise_ref_scaled

    # Spectral gating
    stft = librosa.stft(audio_td, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_stft = librosa.stft(noise_ref, n_fft=2048, hop_length=512)
    noise_mag = np.abs(noise_stft)
    noise_threshold = np.mean(noise_mag, axis=1, keepdims=True) * 1.2
    gated_mag = np.where(magnitude > noise_threshold, magnitude, 0)
    cleaned_stft = gated_mag * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=512)


def compute_acoustic_indices(y, sr):
    """
    Compute all 6 acoustic indices from an audio segment.

    Returns: ADI, ACI, AEI, NDSI, MFC, CLS
    """
    f, t, Sxx = spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    Sxx += 1e-10  # Avoid log(0)

    # --- ADI: Shannon entropy of frequency band energy ---
    S_norm = Sxx / Sxx.sum(axis=0, keepdims=True)
    ADI = np.mean(entropy(S_norm, axis=0))

    # --- AEI: 1 - normalized ADI (complement of diversity) ---
    max_entropy = np.log(Sxx.shape[0]) if Sxx.shape[0] > 1 else 1.0
    AEI = 1.0 - (ADI / max_entropy)

    # --- ACI: Mean normalized absolute spectral difference ---
    diff = np.abs(np.diff(Sxx, axis=1))
    col_sum = Sxx[:, :-1].sum(axis=0)
    col_sum[col_sum == 0] = 1e-10
    ACI = np.mean(diff.sum(axis=0) / col_sum)

    # --- NDSI: (biophony - anthrophony) / (biophony + anthrophony) ---
    freq_res = f[1] - f[0] if len(f) > 1 else 1.0
    anthro_mask = (f >= 1000) & (f <= 2000)
    bio_mask = (f >= 2000) & (f <= 11000)
    E_anthro = Sxx[anthro_mask, :].sum()
    E_bio = Sxx[bio_mask, :].sum()
    NDSI = (E_bio - E_anthro) / (E_bio + E_anthro + 1e-10)

    # --- MFC: Mid-frequency cover (2-8 kHz > 20% of total) ---
    mid_mask = (f >= 2000) & (f <= 8000)
    S_mid = Sxx[mid_mask, :].sum(axis=0)
    S_total = Sxx.sum(axis=0)
    MFC = np.mean(S_mid > 0.2 * S_total)

    # --- CLS: Cluster count (mean peak count per frame) ---
    # Vectorized normalization (all columns at once), then loop only find_peaks
    frame_maxes = Sxx.max(axis=0, keepdims=True) + 1e-10
    Sxx_norm = Sxx / frame_maxes
    peak_counts = np.empty(Sxx_norm.shape[1], dtype=np.int32)
    for col in range(Sxx_norm.shape[1]):
        peaks, _ = find_peaks(Sxx_norm[:, col], height=0.5)
        peak_counts[col] = len(peaks)
    CLS = peak_counts.mean()

    return ADI, ACI, AEI, NDSI, MFC, CLS


def segment_audio(audio, folder_type="2R4W", fs=48000):
    """Segment audio based on recording duty cycle."""
    two_min_samples = int(120 * fs)
    segments = []

    if "2R4W" in folder_type:
        if len(audio) >= two_min_samples:
            segments.append(audio[:two_min_samples])
    elif "5R5W" in folder_type:
        if len(audio) >= two_min_samples:
            segments.append(audio[:two_min_samples])
    elif "30R30W" in folder_type:
        num_chunks = 10
        for start in range(0, len(audio), two_min_samples):
            end = start + two_min_samples
            if end <= len(audio):
                segments.append(audio[start:end])
            if len(segments) >= num_chunks:
                break

    return segments if segments else None


def predict_is_heavy_rain(segment_audio_data, sr, model, le):
    """Check if audio segment contains heavy rain using trained classifier."""
    MODEL_SR = 22050
    SEG_DUR = 10
    N_MFCC = 40

    try:
        HEAVY_RAIN_INDEX = list(le.classes_).index('H')
    except ValueError:
        return False

    resampled = librosa.resample(segment_audio_data, orig_sr=sr, target_sr=MODEL_SR)
    chunk_samples = SEG_DUR * MODEL_SR
    num_chunks = int(len(resampled) // chunk_samples)

    for i in range(num_chunks):
        chunk = resampled[i * chunk_samples:(i + 1) * chunk_samples]
        mfccs = librosa.feature.mfcc(y=chunk, sr=MODEL_SR, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
        prediction = model.predict(mfccs_mean)[0]
        if prediction == HEAVY_RAIN_INDEX:
            return True
    return False


def extract_path_info(dataset_path):
    """Extract spot name, date range, and duty cycle code from path."""
    parts = dataset_path.replace('\\', '/').split('/')
    code = parts[-1] if len(parts) > 0 else ""
    date_range = parts[-2] if len(parts) > 1 else ""
    spot_folder = parts[-3] if len(parts) > 2 else ""
    spot_match = re.search(r'spot[_\s]*(\d+)', spot_folder, re.IGNORECASE)
    spot = f"spot{spot_match.group(1)}" if spot_match else "spot_unknown"
    return spot, date_range, code


# =============================================================================
# CORE PROCESSING (shared by both modes)
# =============================================================================

def process_wav_files(datasets, noise_path, output_dir, skip_set=None,
                      start_date=None, end_date=None, snr_db=18):
    """Process WAV files from dataset dirs → acoustic indices CSV.

    Returns (results_list, processed_filenames_list).
    """
    if skip_set is None:
        skip_set = set()

    start_val = int(start_date) if start_date else None
    end_val = int(end_date) if end_date else None

    # Load noise reference
    print("Loading noise reference...")
    noise_clip_static, _ = librosa.load(noise_path, sr=TARGET_SR)

    # Load rainfall model (optional)
    rain_model, rain_encoder = None, None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_p = os.path.join(script_dir, "rainfall_model.joblib")
    encoder_p = os.path.join(script_dir, "label_encoder.joblib")
    if os.path.exists(model_p) and os.path.exists(encoder_p):
        rain_model = joblib.load(model_p)
        rain_encoder = joblib.load(encoder_p)
        print("Rainfall classifier loaded.")
    else:
        print("WARNING: Rainfall model not found. Skipping rain filtering.")

    all_results = []
    processed_files = []

    for dataset_path in datasets:
        if not os.path.isdir(dataset_path):
            print(f"  WARNING: {dataset_path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {dataset_path}")

        wav_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith('.wav')])
        print(f"Found {len(wav_files)} WAV files")

        for filename in tqdm(wav_files, desc=f"  {os.path.basename(dataset_path)}"):
            if filename in skip_set:
                continue

            # Date filtering
            if not config.filter_by_date(filename, start_val, end_val):
                continue

            year, month, date, hour, minute = config.extract_datetime_from_filename(filename)
            spot = config.extract_spot_from_filename(filename)
            filepath = os.path.join(dataset_path, filename)

            try:
                audio, sr = librosa.load(filepath, sr=TARGET_SR)
            except Exception as e:
                print(f"  ERROR loading {filename}: {e}")
                continue

            # Denoise
            audio_denoised = remove_static_noise(audio, noise_clip_static, snr_db=snr_db)

            # Segment (duration-based, no folder_type needed)
            segments = config.segment_audio(audio_denoised, sr=sr)
            if segments is None:
                continue

            for i, segment in enumerate(segments):
                # Rain filtering
                if rain_model and rain_encoder:
                    if predict_is_heavy_rain(segment, sr, rain_model, rain_encoder):
                        continue

                ADI, ACI, AEI, NDSI, MFC, CLS = compute_acoustic_indices(segment, sr)
                all_results.append({
                    "filename": filename,
                    "spot": spot or "",
                    "Segment": i + 1,
                    "Year": year,
                    "Month": month,
                    "Date": date,
                    "Hour": hour,
                    "Minute": minute,
                    "ADI": ADI,
                    "ACI": ACI,
                    "AEI": AEI,
                    "NDSI": NDSI,
                    "MFC": MFC,
                    "CLS": CLS
                })

            processed_files.append(filename)

    return all_results, processed_files


# =============================================================================
# WATCHER MODE
# =============================================================================

def _run_watcher_mode():
    """Called when watcher passes --output-dir (CLI mode)."""
    args = config.parse_common_args(description="05 – Acoustic Indices (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    noise_path = config.resolve_noise_path(args)
    if not noise_path:
        print("ERROR: No noise reference WAV found. Exiting.")
        sys.exit(1)

    skip_set = config.load_skip_list(args.skip_list)

    results, processed = process_wav_files(
        datasets=args.datasets,
        noise_path=noise_path,
        output_dir=args.output_dir,
        skip_set=skip_set,
        start_date=args.start_date,
        end_date=args.end_date,
        snr_db=args.snr_db,
    )

    if results:
        out_csv = os.path.join(args.output_dir, "acoustic_indices.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved {len(results)} rows → {out_csv}")

    config.save_processed_list(args.output_dir, processed)
    print(f"Done. {len(processed)} files processed.")


# =============================================================================
# STANDALONE MODE (original folder-walking behaviour)
# =============================================================================

def _run_standalone_mode():
    """Original standalone execution using 00_config folder discovery."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    noise_clip_static, _ = librosa.load(STATIC_NOISE_PATH, sr=TARGET_SR)

    rain_model, rain_encoder = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        rain_model = joblib.load(MODEL_PATH)
        rain_encoder = joblib.load(ENCODER_PATH)
        print("Rainfall classifier loaded.")
    else:
        print("WARNING: Rainfall model not found. Skipping rain filtering.")

    for dataset_path in DATASET_PATHS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_path}")
        spot, date_range, code = extract_path_info(dataset_path)
        output_filename = f"{spot}_{date_range}_{code}_indices.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        results = []
        wav_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith('.wav')])
        print(f"Found {len(wav_files)} WAV files")

        for filename in tqdm(wav_files, desc=f"  {spot}"):
            year, month, date, hour, minute = extract_datetime_from_filename(filename)
            filepath = os.path.join(dataset_path, filename)

            try:
                audio, sr = librosa.load(filepath, sr=TARGET_SR)
            except Exception as e:
                print(f"  ERROR loading {filename}: {e}")
                continue

            audio_denoised = remove_static_noise(audio, noise_clip_static)
            segments = segment_audio(audio_denoised, folder_type=code)
            if segments is None:
                continue

            for i, segment in enumerate(segments):
                if rain_model and rain_encoder:
                    if predict_is_heavy_rain(segment, sr, rain_model, rain_encoder):
                        continue
                ADI, ACI, AEI, NDSI, MFC, CLS = compute_acoustic_indices(segment, sr)
                results.append({
                    "filename": filename,
                    "Segment": i + 1,
                    "Year": year,
                    "Month": month,
                    "Date": date,
                    "Hour": hour,
                    "Minute": minute,
                    "ADI": ADI,
                    "ACI": ACI,
                    "AEI": AEI,
                    "NDSI": NDSI,
                    "MFC": MFC,
                    "CLS": CLS
                })

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"  Saved {len(results_df)} rows to: {output_path}")

    print("\nDone. All acoustic indices computed.")


# =============================================================================
# ENTRY POINT — auto-detect mode
# =============================================================================
if __name__ == "__main__":
    import json
    if "--output-dir" in sys.argv:
        _run_watcher_mode()
    else:
        _run_standalone_mode()
