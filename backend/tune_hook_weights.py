#!/usr/bin/env python3
"""
Viral Hook Detection — Advanced Structural Tuning Engine
=========================================================

Replaces the naive 15-second sliding window with Foote's Audio Novelty
Segmentation to discover natural musical boundaries (e.g. Verse → Chorus).
Each structural segment is then scored with a composite of five signals:

  1. ML model probability   – XGBoost ensemble hit prediction
  2. Energy deviation       – RMS energy relative to track median
  3. Danceability           – rhythmic suitability for movement
  4. Onset novelty          – beat-drop intensity (spectral flux)
  5. Repetition score (NEW) – harmonic recurrence across the track

References
----------
  [1] Foote, J. (2000). "Automatic Audio Segmentation Using a Measure
      of Audio Novelty." Proc. IEEE Int. Conf. Multimedia and Expo (ICME).
  [2] Bartsch, M. & Wakefield, G. (2001). "To Catch a Chorus: Using
      Chroma-Based Representations for Audio Thumbnailing." Proc. IEEE
      Workshop on Applications of Signal Processing to Audio and Acoustics.

Usage
-----
  1. Place validation MP3/WAV files in  ./validation_audio/
  2. Run:  python tune_hook_weights.py
"""

import os
import glob
import json
import traceback
import warnings

import librosa
import numpy as np
import scipy.spatial.distance as distance
from scipy.signal import find_peaks

from app import extract_features_from_array, predictor, load_model_globally

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SR = 22050              # Sample rate (Hz)
HOP_LENGTH = 512        # ~23 ms per frame at 22050 Hz
TARGET_WINDOW = 15.0    # TikTok-friendly clip length (seconds)
MIN_SEGMENT = 5.0       # Ignore structural segments shorter than this

# Default composite weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    'model_prob':   0.30,
    'energy_dev':   0.20,
    'danceability': 0.15,
    'novelty':      0.10,
    'repetition':   0.25,      # NEW — Bartsch-inspired
}


# ═══════════════════════════════════════════════════════════════════════════
# Foote's Audio Novelty Segmentation  [1]
# ═══════════════════════════════════════════════════════════════════════════

def compute_ssm(frames):
    """
    Self-Similarity Matrix  S(i,j) = cos_sim(v_i, v_j).

    Foote [1], Section II-A:
        S(i,j) = (v_i · v_j) / (‖v_i‖ · ‖v_j‖)

    Parameters
    ----------
    frames : np.ndarray, shape (num_frames, feature_dim)

    Returns
    -------
    ssm : np.ndarray, shape (num_frames, num_frames)
    """
    pairwise_dist = distance.cdist(frames, frames, metric='cosine')
    return 1.0 - pairwise_dist


def build_checkerboard_kernel(size=64, sigma=32):
    """
    Gaussian-tapered checkerboard kernel  C.

    Foote [1], Section II-B / Figure 3.

    Quadrant layout:
        +1  −1
        −1  +1
    tapered by a radially-symmetric 2-D Gaussian with σ.

    Parameters
    ----------
    size  : int – kernel width (L in the paper)
    sigma : float – Gaussian standard deviation

    Returns
    -------
    kernel : np.ndarray, shape (size, size)
    """
    half = size // 2
    rows = np.where(np.arange(size) >= half, 1, -1)
    cols = np.where(np.arange(size) >= half, 1, -1)
    kernel = np.outer(rows, cols).astype(np.float64)

    # Radial Gaussian taper
    ax = np.arange(size) - half + 0.5
    x, y = np.meshgrid(ax, ax)
    gaussian = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))

    return kernel * gaussian


def compute_foote_novelty(y, sr=SR, kernel_size=64, sigma=32):
    """
    Foote's novelty curve:

        N(i) = Σ_m Σ_n  C(m, n) · S(i + m, i + n)

    The kernel is slid along the main diagonal of the SSM.  Peaks in N
    correspond to structural transition points (e.g. Verse → Chorus).

    Parameters
    ----------
    y           : np.ndarray – audio time series
    sr          : int        – sample rate
    kernel_size : int        – width L of the checkerboard kernel
    sigma       : float      – Gaussian taper σ

    Returns
    -------
    ssm           : np.ndarray   – Self-Similarity Matrix
    novelty_curve : np.ndarray   – per-frame novelty score (normalised 0-1)
    segments      : list[dict]   – structural segments
                                   [{'start', 'end', 'duration'}, …]
    """
    # Chroma CENS — robust to dynamics & local tempo changes
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH)
    frames = chroma.T                        # (num_frames, 12)
    num_frames = frames.shape[0]

    # ---- Self-Similarity Matrix ----
    # Downsample by 2 to speed up SSM and Novelty without losing much precision
    ds_factor = 2
    frames_ds = frames[::ds_factor]
    num_frames_ds = frames_ds.shape[0]
    
    ssm_ds = compute_ssm(frames_ds)

    # ---- Checkerboard kernel ----
    kernel_size_ds = max(16, kernel_size // ds_factor)
    sigma_ds = max(8, sigma / ds_factor)
    kernel = build_checkerboard_kernel(kernel_size_ds, sigma_ds)
    k_half = kernel_size_ds // 2

    # ---- Correlate kernel along the main diagonal of S ----
    novelty_curve_ds = np.zeros(num_frames_ds)
    for i in range(k_half, num_frames_ds - k_half):
        sub = ssm_ds[i - k_half : i + k_half, i - k_half : i + k_half]
        novelty_curve_ds[i] = np.sum(sub * kernel)

    # Interpolate back to original frame rate
    novelty_curve = np.interp(
        np.arange(num_frames), 
        np.arange(num_frames_ds) * ds_factor, 
        novelty_curve_ds
    )
    
    # Upsample SSM back to original size for repetition score computation
    ssm = np.kron(ssm_ds, np.ones((ds_factor, ds_factor)))
    ssm = ssm[:num_frames, :num_frames]

    # Normalise to [0, 1]
    nc_min, nc_max = novelty_curve.min(), novelty_curve.max()
    if nc_max - nc_min > 1e-10:
        novelty_curve = (novelty_curve - nc_min) / (nc_max - nc_min)

    # ---- Peak detection → structural boundaries ----
    # Minimum gap = 5 s to avoid micro-chopping
    min_frames = int(5.0 * sr / HOP_LENGTH)
    peaks, _ = find_peaks(novelty_curve, distance=min_frames, prominence=0.10)

    timestamps = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH)
    total_dur  = librosa.get_duration(y=y, sr=sr)
    boundaries = np.concatenate([[0.0], timestamps, [total_dur]])

    segments = []
    for idx in range(len(boundaries) - 1):
        s, e = float(boundaries[idx]), float(boundaries[idx + 1])
        segments.append({
            'start':    round(s, 2),
            'end':      round(e, 2),
            'duration': round(e - s, 2),
        })

    return ssm, novelty_curve, segments


# ═══════════════════════════════════════════════════════════════════════════
# Bartsch & Wakefield Repetition Score  [2]
# ═══════════════════════════════════════════════════════════════════════════

def compute_repetition_score(ssm, start_frame, end_frame):
    """
    Measures how strongly a segment's chroma profile recurs across the
    rest of the track.

    Bartsch & Wakefield [2] showed that chorus sections exhibit high
    average cosine similarity to the rest of the song because they
    repeat.  We compute:

        rep = mean( S[segment_rows, non_segment_cols] )

    High score  ⇒  segment's harmonic content appears elsewhere
                    ⇒  likely a chorus / hook.

    Parameters
    ----------
    ssm         : np.ndarray – full-track Self-Similarity Matrix
    start_frame : int        – segment start (frame index)
    end_frame   : int        – segment end   (frame index)

    Returns
    -------
    float – repetition score clipped to [0, 1]
    """
    n = ssm.shape[0]
    if end_frame <= start_frame or n == 0:
        return 0.0

    seg_rows = ssm[start_frame:end_frame, :]

    # Mask out the segment itself so we only measure cross-similarity
    mask = np.ones(n, dtype=bool)
    mask[start_frame:end_frame] = False
    if not np.any(mask):
        return 0.0

    cross_sim = np.mean(seg_rows[:, mask])
    return float(np.clip(cross_sim, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Sub-window extraction for long segments
# ═══════════════════════════════════════════════════════════════════════════

def find_peak_subwindow(y_segment, sr, window_sec=TARGET_WINDOW):
    """
    For structural segments longer than *window_sec*, slide a window
    across the segment and return the sub-window with the highest mean
    RMS energy (the most intense 15 s).

    Parameters
    ----------
    y_segment  : np.ndarray – audio for the segment
    sr         : int
    window_sec : float

    Returns
    -------
    (offset_start, offset_end) in seconds relative to segment start
    """
    seg_dur = len(y_segment) / sr
    if seg_dur <= window_sec:
        return 0.0, seg_dur

    rms = librosa.feature.rms(y=y_segment, hop_length=HOP_LENGTH)[0]
    win_frames = int(window_sec * sr / HOP_LENGTH)

    if len(rms) > win_frames:
        cumsum = np.cumsum(np.insert(rms, 0, 0))
        window_means = (cumsum[win_frames:] - cumsum[:-win_frames]) / win_frames
        best_start = int(np.argmax(window_means))
    else:
        best_start = 0

    start_sec = best_start * HOP_LENGTH / sr
    end_sec   = min(start_sec + window_sec, seg_dur)
    return start_sec, end_sec


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_track_segments(file_path, weights):
    """
    Analyze a single track using Foote structural segmentation + composite
    scoring.  Returns the highest-scoring segment dict, or None.

    Pipeline
    --------
    1. Load audio
    2. compute_foote_novelty()  →  natural structural boundaries
    3. For each structural segment:
       a. If > ~20 s, crop to peak-energy 15 s sub-window
       b. Extract librosa features
       c. Run ML prediction, compute energy/danceability/novelty/repetition
       d. Weighted composite score
    4. Return best segment
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        if len(y) == 0:
            return None

        total_duration = librosa.get_duration(y=y, sr=sr)
        if total_duration < 10.0:
            print(f"  [SKIP] {os.path.basename(file_path)}: track too short ({total_duration:.1f}s)")
            return None

        # ── Foote Structural Segmentation ──
        ssm, novelty_curve, natural_segments = compute_foote_novelty(y, sr)
        print(f"  [FOOTE] Found {len(natural_segments)} structural segments")

        # ── Global energy baseline ──
        rms_global = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        global_median_energy = np.median(rms_global)
        global_max_energy    = np.max(rms_global) + 1e-10

        scored_segments = []

        for seg in natural_segments:
            if seg['duration'] < MIN_SEGMENT:
                continue

            seg_start_sample = int(seg['start'] * sr)
            seg_end_sample   = int(seg['end']   * sr)

            # If segment is longer than ~20 s, crop to the most intense 15 s
            if seg['duration'] > TARGET_WINDOW * 1.3:
                y_full_seg = y[seg_start_sample:seg_end_sample]
                sub_start, sub_end = find_peak_subwindow(y_full_seg, sr)
                chunk_start = seg['start'] + sub_start
                chunk_end   = seg['start'] + sub_end
            else:
                chunk_start = seg['start']
                chunk_end   = seg['end']

            start_sample = int(chunk_start * sr)
            end_sample   = int(chunk_end   * sr)
            y_chunk = y[start_sample:end_sample]

            if len(y_chunk) < sr:       # < 1 s after clipping
                continue

            # ── Feature extraction ──
            chunk_features = extract_features_from_array(y_chunk, sr)
            if not chunk_features:
                continue
            chunk_features['duration_ms'] = int(total_duration * 1000)
            chunk_features['target_year'] = 2024

            # ── 1. ML probability ──
            pred_result = predictor.predict_song_hit_probability(chunk_features)
            if not pred_result:
                continue
            model_prob = pred_result['hit_probability']

            # ── 2. Energy deviation ──
            chunk_rms = librosa.feature.rms(y=y_chunk, hop_length=HOP_LENGTH)[0]
            chunk_mean_energy = np.mean(chunk_rms)
            energy_dev = np.clip(
                (chunk_mean_energy - global_median_energy) / global_max_energy,
                0, 1,
            )

            # ── 3. Danceability ──
            danceability = chunk_features.get('danceability', 0.5)

            # ── 4. Onset novelty (spectral flux) ──
            chunk_onset = librosa.onset.onset_strength(y=y_chunk, sr=sr)
            onset_novelty = np.clip(np.mean(chunk_onset) / 10.0, 0, 1)

            # ── 5. Repetition score (Bartsch) ──
            seg_start_frame = int(seg['start'] * sr / HOP_LENGTH)
            seg_end_frame   = min(int(seg['end'] * sr / HOP_LENGTH), ssm.shape[0])
            repetition = compute_repetition_score(ssm, seg_start_frame, seg_end_frame)

            # ── Composite score ──
            score = (
                weights.get('model_prob',   0) * model_prob    +
                weights.get('energy_dev',   0) * energy_dev    +
                weights.get('danceability', 0) * danceability  +
                weights.get('novelty',      0) * onset_novelty +
                weights.get('repetition',   0) * repetition
            )

            scored_segments.append({
                'start':        round(chunk_start, 2),
                'end':          round(chunk_end,   2),
                'score':        round(score, 4),
                'model_prob':   round(model_prob, 4),
                'energy_dev':   round(energy_dev, 4),
                'danceability': round(danceability, 4),
                'novelty':      round(onset_novelty, 4),
                'repetition':   round(repetition, 4),
                'structural_segment': f"{seg['start']:.1f}s-{seg['end']:.1f}s",
            })

        if not scored_segments:
            return None

        # Return best segment
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        return scored_segments[0]

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Tuning engine
# ═══════════════════════════════════════════════════════════════════════════

def tune_weights(audio_dir, parameter_grid):
    """
    Empirical grid search: test each weight configuration against a set
    of validation tracks and report the best-performing combination.
    """
    print("Loading Global ML Model...")
    load_model_globally()

    audio_files = (
        glob.glob(os.path.join(audio_dir, '*.mp3')) +
        glob.glob(os.path.join(audio_dir, '*.wav'))
    )
    if not audio_files:
        print(f"No audio files found in {audio_dir}. Add some tracks and re-run.")
        return

    print(f"Found {len(audio_files)} validation tracks.\n")

    results = []

    for config in parameter_grid:
        label   = config.get('_label', 'Custom')
        weights = {k: v for k, v in config.items() if k != '_label'}

        print(f"--- [{label}] ---")
        print(f"    Weights: {weights}")
        config_scores = []

        for fp in audio_files[:5]:          # cap at 5 for speed
            filename = os.path.basename(fp)
            best = analyze_track_segments(fp, weights)
            if best:
                print(
                    f"    {filename:35s} -> Hook @ "
                    f"{best['start']:.1f}s-{best['end']:.1f}s  "
                    f"(score={best['score']:.3f}  "
                    f"rep={best['repetition']:.2f}  "
                    f"energy={best['energy_dev']:.2f}  "
                    f"dance={best['danceability']:.2f})"
                )
                config_scores.append(best['score'])

        if config_scores:
            avg = float(np.mean(config_scores))
            results.append({
                'label':     label,
                'weights':   weights,
                'avg_score': avg,
            })
            print(f"    => Average Hook Confidence: {avg:.3f}\n")

    if not results:
        print("No results produced. Check your audio files.")
        return

    results.sort(key=lambda x: x['avg_score'], reverse=True)
    best = results[0]

    print("=" * 60)
    print("  BEST CONFIGURATION FOUND")
    print("=" * 60)
    print(f"  Label : {best['label']}")
    print(f"  Score : {best['avg_score']:.3f}")
    print(f"  Weights:")
    print(json.dumps(best['weights'], indent=4))

    # Save result to JSON
    output_path = os.path.join(audio_dir, 'tuning_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Parameter grid — each entry is a named weight configuration.
    # All weights within a config should sum to 1.0.
    grid = [
        # Balanced baseline
        {
            '_label':       'Balanced',
            'model_prob':   0.25,
            'energy_dev':   0.20,
            'danceability': 0.15,
            'novelty':      0.10,
            'repetition':   0.30,
        },
        # Chorus hunter — repetition-dominant (Bartsch & Wakefield)
        {
            '_label':       'Chorus Hunter',
            'model_prob':   0.15,
            'energy_dev':   0.15,
            'danceability': 0.10,
            'novelty':      0.10,
            'repetition':   0.50,
        },
        # Drop detector — energy + onset spike
        {
            '_label':       'Drop Detector',
            'model_prob':   0.15,
            'energy_dev':   0.40,
            'danceability': 0.10,
            'novelty':      0.25,
            'repetition':   0.10,
        },
        # TikTok-optimised — danceability + repetition
        {
            '_label':       'TikTok Optimised',
            'model_prob':   0.10,
            'energy_dev':   0.15,
            'danceability': 0.35,
            'novelty':      0.10,
            'repetition':   0.30,
        },
        # ML trust — lean on the trained model
        {
            '_label':       'ML Trust',
            'model_prob':   0.50,
            'energy_dev':   0.10,
            'danceability': 0.10,
            'novelty':      0.10,
            'repetition':   0.20,
        },
    ]

    print("=" * 60)
    print("  Viral Hook Detection - Structural Tuning Engine")
    print("  Foote (2000) + Bartsch & Wakefield (2001)")
    print("=" * 60)
    print()

    tuning_dir = "./validation_audio"
    if not os.path.exists(tuning_dir):
        os.makedirs(tuning_dir)
        print(f"[INFO] Created '{tuning_dir}/'.")
        print(f"       Place validation MP3/WAV files there and re-run.")
    else:
        tune_weights(tuning_dir, grid)
