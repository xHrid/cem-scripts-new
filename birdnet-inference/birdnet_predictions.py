import os
import re
import sys
import argparse
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram
import tempfile
from tqdm import tqdm
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# Monkey patch for compatibility
np.complex = complex  

TARGET_SR = 48000
SEGMENT_DURATION = 120.0  

def extract_metadata_from_filename(filename):
    """Safely extract metadata from the standard CEM filename convention."""
    match = re.search(r'(SPOT\d+)_(\d{8})_(\d{6})\.wav$', filename, re.IGNORECASE)
    if match:
        spot = match.group(1).upper()
        date_str = match.group(2)
        time_str = match.group(3)
        year, month, day = date_str[:4], date_str[4:6], date_str[6:]
        hour, minute = int(time_str[:2]), int(time_str[2:4])
        return spot, year, month, day, hour, minute
    return None, None, None, None, None, None

def segment_audio(audio, folder_type, fs=48000):
    """Extracts 2-minute segments based on recording schedules."""
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

def remove_static_noise(audio, noise_ref, sr=48000, snr_db=18):
    """Remove static noise using time-domain subtraction and spectral gating."""
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

def analyze_bird_audio(audio_path, lat, lon):
    """Analyze audio with BirdNET."""
    analyzer = Analyzer()
    recording = Recording(analyzer, audio_path, lat=lat, lon=lon)
    recording.analyze()
    df = pd.DataFrame.from_records(recording.detections)

    if not df.empty:
        df = df.rename(columns={"start": "start_time", "end": "end_time"})
    return df

def main(datasets, static_noise_path, output_dir, sampling_rule_base, lat=28.53, lon=77.18, target_sr=48000, snr_db=18):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(static_noise_path):
        print(f"ERROR: Noise file missing at {static_noise_path}")
        sys.exit(1)
        
    noise_clip, _ = librosa.load(static_noise_path, sr=target_sr)
    all_detections = []
    
    for dataset_dir in datasets:
        if not os.path.isdir(dataset_dir):
            print(f"WARNING: Dataset path {dataset_dir} is not a directory. Skipping.")
            continue
            
        print(f"\nScanning directory: {dataset_dir} with rule {sampling_rule_base}")
        file_counter_5r5w = 0
        
        wav_files = [f for f in sorted(os.listdir(dataset_dir)) if f.lower().endswith(".wav")]
        
        for fname in tqdm(wav_files, desc=f"Processing {os.path.basename(dataset_dir)}"):
            spot, year, month, day, hour, minute = extract_metadata_from_filename(fname)
            
            if not spot:
                print(f"Skipping {fname} - Does not match expected CEM filename convention.")
                continue

            filepath = os.path.join(dataset_dir, fname)
            try:
                audio, sr = librosa.load(filepath, sr=target_sr)
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
                continue

            sampling_rule = sampling_rule_base
            if sampling_rule_base == "5R5W":
                sampling_rule = "5R5W_first_last" if file_counter_5r5w in [0, 2] else "5R5W_central"
                file_counter_5r5w = (file_counter_5r5w + 1) % 3

            audio_denoised = remove_static_noise(audio, noise_clip, sr=sr, snr_db=snr_db)
            segments = segment_audio(audio_denoised, sampling_rule, fs=sr)

            if not segments:
                continue
            
            for j, segment in enumerate(segments):
                temp_segment_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        temp_segment_path = tmp.name
                    sf.write(temp_segment_path, segment, target_sr)

                    # Suppress BirdNet verbose output
                    import io
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    
                    detections_df = analyze_bird_audio(temp_segment_path, lat, lon)
                    
                    sys.stdout = old_stdout

                    if not detections_df.empty:
                        detections_df["filename"] = fname
                        detections_df["spot"] = spot
                        detections_df["segment_index"] = j
                        detections_df["sampling_rule"] = sampling_rule
                        detections_df["year"] = year
                        detections_df["month"] = month
                        detections_df["day"] = day
                        detections_df["hour"] = hour
                        detections_df["minute"] = minute
                        all_detections.append(detections_df)
                        
                except Exception as e:
                    sys.stdout = old_stdout # ensure restore
                    print(f"Error on {fname} segment {j}: {e}")
                finally:
                    if temp_segment_path and os.path.exists(temp_segment_path):
                        os.unlink(temp_segment_path)
    
    if all_detections:
        output_path = os.path.join(output_dir, "birdnet_results.csv")
        final_df = pd.concat(all_detections, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Success! Saved {len(final_df)} detections to {output_path}")
    else:
        print("\n⚠️ No detections processed across any datasets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--noise-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling-rule", required=True, choices=["2R4W", "5R5W", "30R30W"], help="Sampling rule from UI")
    parser.add_argument("--lat", type=float, default=28.53)
    parser.add_argument("--lon", type=float, default=77.18)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--snr-db", type=int, default=18)
    args = parser.parse_args()
    
    main(args.datasets, args.noise_path, args.output_dir, args.sampling_rule, args.lat, args.lon, args.sample_rate, args.snr_db)