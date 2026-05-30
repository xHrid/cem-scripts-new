"""
00b: BirdNET Predictions Pipeline
===================================
Three-part pipeline:
  1. File listing  — discover, filter, deduplicate WAV files
  2. Main pipeline — run BirdNET on new files, append to aggregate CSV
  3. Output CSV    — filtered subset of aggregate for requested range
"""

import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from datetime import date
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import contextlib
import multiprocessing

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

import config as cfg

# Filename pattern: SPOTNAME_YYYYMMDD_HHMMSS.wav
_FILENAME_RE = re.compile(
    r"^(?P<spot>[A-Za-z][A-Za-z0-9_-]*)_"
    r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\.wav$",
    re.IGNORECASE,
)


# =============================================================================
# PART 1 — FILE LISTING
# =============================================================================
def parse_filename(filename: str) -> dict | None:
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    month, day = int(m.group("month")), int(m.group("day"))
    hour, minute, second = int(m.group("hour")), int(m.group("minute")), int(m.group("second"))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    try:
        file_date = date(int(m.group("year")), month, day)
    except ValueError:
        return None
    return {"spot": m.group("spot"), "date": file_date, "hour": hour, "filename": filename}


def list_files(
    input_directories: list[str],
    date_start: date,
    date_end: date,
    processed_files: set[str],
    input_file_list: list[str] | None = None,
) -> list[str]:
    discovered: dict[str, str] = {}

    for directory in input_directories:
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            print(f"WARNING: input directory not found: {directory}")
            continue
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                parsed = parse_filename(fname)
                if parsed is None:
                    continue
                if not (date_start <= parsed["date"] <= date_end):
                    continue
                if fname not in discovered:
                    discovered[fname] = os.path.join(root, fname)

    if input_file_list:
        for fpath in input_file_list:
            fpath = os.path.abspath(fpath)
            fname = os.path.basename(fpath)
            if fname not in discovered and os.path.isfile(fpath):
                discovered[fname] = fpath

    for pf in processed_files:
        discovered.pop(pf, None)

    result = sorted(discovered.values())
    print(f"File listing: {len(result)} to process ({len(processed_files)} already processed)")
    return result


# =============================================================================
# PART 2 — MAIN PIPELINE
# =============================================================================
def _denoise(audio: np.ndarray, noise_ref: np.ndarray,
             sr: int | None = None, snr_db: float | None = None) -> np.ndarray:
    # Read from config at call-time so CLI overrides (e.g. --snr-db) take effect.
    sr = cfg.TARGET_SR if sr is None else sr
    snr_db = cfg.SNR_DB if snr_db is None else snr_db
    if len(noise_ref) > len(audio):
        noise_ref = noise_ref[:len(audio)]
    else:
        noise_ref = np.pad(noise_ref, (0, len(audio) - len(noise_ref)), "wrap")

    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_ref ** 2)
    if noise_power == 0:
        return audio
    desired_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_ref_scaled = noise_ref * np.sqrt(desired_noise_power / noise_power)

    audio_td = audio - noise_ref_scaled
    n_fft, hop = 2048, 512
    stft = librosa.stft(audio_td, n_fft=n_fft, hop_length=hop)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_stft = librosa.stft(noise_ref, n_fft=n_fft, hop_length=hop)
    noise_threshold = np.mean(np.abs(noise_stft), axis=1, keepdims=True) * 1.2
    gated_mag = np.where(magnitude > noise_threshold, magnitude, 0)
    return librosa.istft(gated_mag * np.exp(1j * phase), hop_length=hop)


def _analyze_file(filepath, analyzer, noise_clip, rain_clip):
    audio_raw, orig_sr = sf.read(filepath, dtype="float32")
    if audio_raw.ndim > 1:
        audio_raw = audio_raw.mean(axis=1)
    if orig_sr != cfg.TARGET_SR:
        audio_raw = librosa.resample(y=audio_raw, orig_sr=orig_sr, target_sr=cfg.TARGET_SR)

    audio_clean = _denoise(audio_raw, noise_clip)
    audio_clean = _denoise(audio_clean, rain_clip)

    recording = RecordingBuffer(
        analyzer, audio_clean, cfg.TARGET_SR,
        lat=cfg.LATITUDE, lon=cfg.LONGITUDE, min_conf=cfg.MIN_CONFIDENCE,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        recording.analyze()
    return pd.DataFrame(recording.detections)


_worker_analyzer = None
_worker_noise = None
_worker_rain = None


def _init_worker(noise_path, rain_path, tflite_threads):
    global _worker_analyzer, _worker_noise, _worker_rain

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _worker_analyzer = Analyzer()

    if tflite_threads > 1:
        try:
            interp = _worker_analyzer.interpreter
            with contextlib.redirect_stderr(io.StringIO()):
                new_interp = type(interp)(model_path=_worker_analyzer.model_path, num_threads=tflite_threads)
            new_interp.allocate_tensors()
            _worker_analyzer.interpreter = new_interp
            _worker_analyzer.input_details = new_interp.get_input_details()
            _worker_analyzer.output_details = new_interp.get_output_details()
            _worker_analyzer.input_layer_index = _worker_analyzer.input_details[0]["index"]
            _worker_analyzer.output_layer_index = _worker_analyzer.output_details[0]["index"]
        except Exception:
            pass

    _worker_noise, _ = sf.read(noise_path, dtype="float32")
    _worker_rain, _ = sf.read(rain_path, dtype="float32")
    if _worker_noise.ndim > 1:
        _worker_noise = _worker_noise.mean(axis=1)
    if _worker_rain.ndim > 1:
        _worker_rain = _worker_rain.mean(axis=1)

    noise_sr = sf.info(noise_path).samplerate
    rain_sr = sf.info(rain_path).samplerate
    if noise_sr != cfg.TARGET_SR:
        _worker_noise = librosa.resample(y=_worker_noise, orig_sr=noise_sr, target_sr=cfg.TARGET_SR)
    if rain_sr != cfg.TARGET_SR:
        _worker_rain = librosa.resample(y=_worker_rain, orig_sr=rain_sr, target_sr=cfg.TARGET_SR)


def _process_single_file(filepath):
    filename = os.path.basename(filepath)
    parsed = parse_filename(filename)
    hour = parsed["hour"] if parsed else None
    try:
        df = _analyze_file(filepath, _worker_analyzer, _worker_noise, _worker_rain)
        if not df.empty:
            df["filename"] = filename
            df["filepath"] = filepath
            df["hour"] = hour
            df["spot"] = parsed["spot"] if parsed else ""
            if "common_name" in df.columns and "label" not in df.columns:
                df["label"] = df["common_name"]
            return filename, df
        return filename, None
    except Exception as e:
        print(f"\n  ERROR processing {filename}: {e}")
        return filename, None


def load_processed_files(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def save_processed_files(path: str, filenames: set[str]):
    with open(path, "w") as f:
        for fname in sorted(filenames):
            f.write(fname + "\n")


def run_pipeline(file_list, aggregate_path, processed_files_path):
    if not file_list:
        print("No new files to process.")
        return pd.DataFrame()

    total_cpus = multiprocessing.cpu_count()
    n_workers = max(1, min(total_cpus // 2, 4))
    threads_per = max(1, total_cpus // n_workers)
    print(f"Parallelism: {n_workers} workers × {threads_per} TFLite threads ({total_cpus} CPUs)")

    all_detections = []
    processed_this_run = set()
    already_processed = load_processed_files(processed_files_path)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(cfg.STATIC_NOISE_PATH, cfg.RAIN_NOISE_PATH, threads_per),
    ) as executor:
        futures = {executor.submit(_process_single_file, fp): fp for fp in file_list}
        with tqdm(total=len(file_list), desc="BirdNET") as pbar:
            for future in as_completed(futures):
                filename, result = future.result()
                processed_this_run.add(filename)
                if result is not None:
                    all_detections.append(result)
                pbar.update(1)

    new_df = pd.DataFrame()
    if all_detections:
        new_df = pd.concat(all_detections, ignore_index=True)
        header = not os.path.isfile(aggregate_path)
        new_df.to_csv(aggregate_path, mode="a", header=header, index=False)
        print(f"Appended {len(new_df)} detections to {aggregate_path}")
    else:
        print("No detections in this batch.")

    already_processed.update(processed_this_run)
    save_processed_files(processed_files_path, already_processed)
    print(f"Marked {len(processed_this_run)} files as processed (total: {len(already_processed)})")
    return new_df


# =============================================================================
# PART 3 — OUTPUT CSV
# =============================================================================
def write_output_csv(aggregate_path, output_path, input_directories, date_start, date_end):
    if not os.path.isfile(aggregate_path):
        print("No aggregate file found.")
        return

    df = pd.read_csv(aggregate_path)
    if df.empty:
        print("Aggregate file is empty.")
        return

    if "filepath" in df.columns:
        abs_dirs = [os.path.abspath(d) for d in input_directories]
        df = df[df["filepath"].apply(
            lambda fp: not pd.isna(fp) and any(os.path.abspath(str(fp)).startswith(d + os.sep) for d in abs_dirs)
        )]

    if "filename" in df.columns:
        df = df[df["filename"].apply(lambda fn: (p := parse_filename(str(fn))) is not None and date_start <= p["date"] <= date_end)]

    if df.empty:
        print("No detections match requested directories + date range.")
        return

    df.to_csv(output_path, index=False)
    print(f"Output: {len(df)} detections → {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg.apply_overrides()
    processed_set = load_processed_files(cfg.PROCESSED_FILE)
    files_to_process = list_files(
        input_directories=cfg.INPUT_DIRECTORIES,
        date_start=cfg.DATE_START,
        date_end=cfg.DATE_END,
        processed_files=processed_set,
        input_file_list=cfg.INPUT_FILE_LIST,
    )

    run_pipeline(
        file_list=files_to_process,
        aggregate_path=cfg.AGGREGATE_FILE,
        processed_files_path=cfg.PROCESSED_FILE,
    )

    write_output_csv(
        aggregate_path=cfg.AGGREGATE_FILE,
        output_path=cfg.OUTPUT_CSV,
        input_directories=cfg.INPUT_DIRECTORIES,
        date_start=cfg.DATE_START,
        date_end=cfg.DATE_END,
    )


if __name__ == "__main__":
    main()
