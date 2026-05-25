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

STATIC_NOISE_PATH = config.STATIC_NOISE_PATH
TARGET_SR = config.TARGET_SR
MODEL_PATH = config.RAINFALL_MODEL_PATH
ENCODER_PATH = config.RAINFALL_ENCODER_PATH

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
    """Aggregate-aware watcher mode.

    Flow:
      1. Load aggregate → get processed filenames
      2. Process only NEW files from datasets
      3. Append results to aggregate + mark empty files
      4. Output date-filtered subset to job output dir
    """
    args = config.parse_common_args(description="05 – Acoustic Indices (watcher)")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Aggregate ---
    aggregate_path = config.resolve_aggregate_path(args, "acoustic_indices.csv")
    print(f"Aggregate file: {aggregate_path}")

    existing_aggregate = config.load_aggregate(aggregate_path)
    already_processed = config.get_processed_filenames(existing_aggregate)
    print(f"Aggregate contains {len(already_processed)} already-processed files")

    noise_path = config.resolve_noise_path(args)
    if not noise_path:
        print("ERROR: No noise reference WAV found. Exiting.")
        sys.exit(1)

    # Process only NEW files (skip_set = already in aggregate)
    results, processed = process_wav_files(
        datasets=args.datasets,
        noise_path=noise_path,
        output_dir=args.output_dir,
        skip_set=already_processed,
        snr_db=args.snr_db,
    )

    # --- Update aggregate ---
    files_with_results = set()
    if results:
        new_df = pd.DataFrame(results)
        config.append_to_aggregate(new_df, aggregate_path)
        files_with_results = set(new_df["filename"].unique())
        print(f"Appended {len(new_df)} new rows to aggregate")

    # Mark files with no indices (e.g. too short to segment)
    empty_files = [f for f in processed if f not in files_with_results]
    if empty_files:
        config.mark_empty_files(empty_files, aggregate_path)
        print(f"Marked {len(empty_files)} files as empty")

    # --- Output filtered subset ---
    full_aggregate = config.load_aggregate(aggregate_path)
    filtered = config.filter_aggregate_for_output(
        full_aggregate,
        start_date=args.start_date,
        end_date=args.end_date,
        spots=args.spots,
    )

    if not filtered.empty:
        out_csv = os.path.join(args.output_dir, "acoustic_indices.csv")
        filtered.to_csv(out_csv, index=False)
        print(f"Output {len(filtered)} rows for requested range → {out_csv}")
    else:
        print("WARNING: No indices in requested date range / spots.")

    config.save_processed_list(args.output_dir, processed)
    print(f"Done. {len(processed)} new files processed.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    _run_watcher_mode()
