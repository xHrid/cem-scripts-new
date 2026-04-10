import argparse
import os
import re
import sys
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram, find_peaks
from scipy.stats import entropy
import joblib
from tqdm import tqdm

TARGET_SR = 48000

def predict_is_heavy_rain(audio_segment, sr, model, label_index):
    return False

def extract_metadata_from_filename(filename):
    match = re.search(r'(SPOT\d+)_(\d{8})_(\d{6})\.wav$', filename, re.IGNORECASE)
    if match:
        spot = match.group(1).upper()
        date_str = match.group(2)
        time_str = match.group(3)
        year, month, day = date_str[:4], date_str[4:6], date_str[6:]
        hour, minute = int(time_str[:2]), int(time_str[2:4])
        return spot, year, month, day, hour, minute
    return None, None, None, None, None, None

def remove_static_noise(audio, noise_ref, sr=TARGET_SR, snr_db=18):
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

def compute_acoustic_indices(y, sr):
    f, t, Sxx = spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    Sxx += 1e-10 
    S_norm = Sxx / Sxx.sum(axis=0, keepdims=True)
    ADI = np.mean(entropy(S_norm, axis=0))
    AEI = 1.0 - (ADI / np.log(Sxx.shape[0])) if Sxx.shape[0] > 1 else 1.0
    delta = np.abs(np.diff(Sxx, axis=1))
    ACI_vals = np.sum(delta, axis=1) / (np.sum(Sxx[:, :-1], axis=1))
    ACI_total = np.mean(ACI_vals)
    bio_band = np.logical_and(f >= 2000, f <= 11000)
    anthro_band = np.logical_and(f >= 100, f <= 2000)
    B = np.sum(Sxx[bio_band, :])
    A = np.sum(Sxx[anthro_band, :])
    NDSI = (B - A) / (B + A)
    mid_band = np.logical_and(f >= 2000, f <= 8000)
    mid_band_energy = np.sum(Sxx[mid_band, :], axis=0)
    total_energy = np.sum(Sxx, axis=0)
    threshold = 0.2 * total_energy
    MFC = np.mean(mid_band_energy > threshold)
    CLS_list = []
    for frame in Sxx.T: 
        norm_frame = frame / (np.max(frame))
        peaks, _ = find_peaks(norm_frame, height=0.5)
        CLS_list.append(len(peaks))
    CLS = np.mean(CLS_list)
    return ADI, ACI_total, AEI, NDSI, MFC, CLS

def segment_audio(audio, folder_type, fs=48000):
    segments = []
    total_samples = len(audio)
    two_min_samples = int(120 * fs)

    if "2R4W" in folder_type:
        if total_samples >= two_min_samples:
            segments.append(audio[:two_min_samples])
    elif "5R5W" in folder_type:
        if "first_last" in folder_type:
            segments.append(audio[:two_min_samples])
            if total_samples >= two_min_samples:
                segments.append(audio[-two_min_samples:])
        elif "central" in folder_type:
            start = (total_samples // 2) - (two_min_samples // 2)
            end = start + two_min_samples
            if start >= 0 and end <= total_samples:
                segments.append(audio[start:end])
    elif "30R30W" in folder_type:
        num_chunks = 10 
        if total_samples >= (num_chunks * two_min_samples):
            gap = (total_samples - (num_chunks * two_min_samples)) // (num_chunks - 1)
            for i in range(num_chunks):
                start = i * (two_min_samples + gap)
                end = start + two_min_samples
                if end <= total_samples:
                    segments.append(audio[start:end])
        else:
            for start in range(0, total_samples, two_min_samples):
                end = start + two_min_samples
                if end <= total_samples:
                    segments.append(audio[start:end])
    return segments if segments else None

def main(datasets, static_noise_path, output_dir, sampling_rule_base, model_path, encoder_path, spots, start_date, end_date):
    os.makedirs(output_dir, exist_ok=True)
    
    # Process UI filters
    valid_spots = [s.upper().strip() for s in spots.split(',')] if spots else None
    start_val = int(start_date) if start_date else None
    end_val = int(end_date) if end_date else None

    model, le, HEAVY_RAIN_INDEX = None, None, -1
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)
            HEAVY_RAIN_INDEX = list(le.classes_).index('H')
        except Exception as e:
            pass
    
    noise_clip_static, _ = librosa.load(static_noise_path, sr=TARGET_SR)
    all_results = []
    
    for dataset_dir in datasets:
        if not os.path.isdir(dataset_dir): continue
        
        file_counter_5r5w = 0
        wav_files = [f for f in sorted(os.listdir(dataset_dir)) if f.lower().endswith(".wav")]
        
        for filename in tqdm(wav_files, desc=f"Scanning {os.path.basename(dataset_dir)}"):
            spot, year, month, date, hour, minute = extract_metadata_from_filename(filename)
            if not spot: continue

            # Apply UI Filters immediately to skip irrelevant files
            if valid_spots and spot.upper() not in valid_spots:
                continue
            file_date = int(f"{year}{month}{date}")
            if start_val and file_date < start_val:
                continue
            if end_val and file_date > end_val:
                continue

            filepath = os.path.join(dataset_dir, filename)
            try:
                audio, sr = librosa.load(filepath, sr=TARGET_SR)
            except Exception:
                continue
                
            sampling_rule = sampling_rule_base
            if sampling_rule_base == "5R5W":
                sampling_rule = "5R5W_first_last" if file_counter_5r5w in [0, 2] else "5R5W_central"
                file_counter_5r5w = (file_counter_5r5w + 1) % 3
            
            audio_denoised = remove_static_noise(audio, noise_clip_static)
            segments = segment_audio(audio_denoised, sampling_rule, fs=sr)
            
            if not segments: continue
            
            for i, segment in enumerate(segments):
                ADI, ACI, AEI, NDSI, MFC, CLS = compute_acoustic_indices(segment.flatten(), sr)
                all_results.append({
                    "filename": filename,
                    "spot": spot,
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
    
    if all_results:
        output_path = os.path.join(output_dir, "acoustic_indices.csv")
        pd.DataFrame(all_results).to_csv(output_path, index=False)
        print(f"\n✅ Saved indices for {len(all_results)} segments to {output_path}")
    else:
        print("\n⚠️ No files processed successfully. Check spot/date filters.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--noise-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling-rule", required=True, choices=["2R4W", "5R5W", "30R30W"])
    parser.add_argument("--model-path", default='rainfall_model.joblib')
    parser.add_argument("--encoder-path", default='label_encoder.joblib')
    
    # Added UI filter arguments
    parser.add_argument("--spots", type=str, default="")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    
    args = parser.parse_args()
    
    main(args.datasets, args.noise_path, args.output_dir, args.sampling_rule, 
         args.model_path, args.encoder_path, args.spots, args.start_date, args.end_date)