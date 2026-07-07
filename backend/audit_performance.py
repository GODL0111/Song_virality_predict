import time
import librosa
import numpy as np
from app import extract_temporal_hooks, extract_features_from_array, predictor, load_model_globally
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_audit(audio_path):
    print(f"--- Performance Audit for {audio_path} ---")
    
    # Load Model
    t0 = time.time()
    load_model_globally()
    t_model = time.time() - t0
    print(f"[1] Model Loading: {t_model:.3f}s")
    
    # Load Audio
    t0 = time.time()
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    t_load = time.time() - t0
    print(f"[2] MP3 Loading (librosa.load): {t_load:.3f}s")
    
    # Global Feature Extraction
    t0 = time.time()
    features = extract_features_from_array(y, sr)
    t_global = time.time() - t0
    print(f"[3] Global Feature Extraction: {t_global:.3f}s")
    
    # Global XGBoost
    t0 = time.time()
    features['duration_ms'] = librosa.get_duration(y=y, sr=sr) * 1000
    features['target_year'] = 2024
    result = predictor.predict_song_hit_probability(features)
    t_xgb = time.time() - t0
    print(f"[4] Global XGBoost Prediction: {t_xgb:.3f}s")
    
    # Temporal Hooks
    print("\n--- Starting Extract Temporal Hooks ---")
    total_duration_sec = librosa.get_duration(y=y, sr=sr)
    
    t0 = time.time()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    t_beat = time.time() - t0
    print(f"[5] Beat Tracking: {t_beat:.3f}s")
    
    t0 = time.time()
    rms = librosa.feature.rms(y=y)[0]
    global_median_energy = np.median(rms)
    global_max_energy = np.max(rms) + 1e-10
    t_rms = time.time() - t0
    print(f"[6] Global RMS Calculation: {t_rms:.3f}s")
    
    print(f"Expected number of loops: {int((total_duration_sec - 15) / 5)}")
    
    loop_times = []
    extract_times = []
    xgb_times = []
    onset_times = []
    
    segments = []
    window_sec = 15.0
    stride_sec = 5.0
    
    t_loop_start = time.time()
    count = 0
    
    if len(beat_times) < 10:
        beat_times = np.arange(0, total_duration_sec, 0.5)
        
    for start_sec in np.arange(0, total_duration_sec - window_sec, stride_sec):
        count += 1
        t_iter = time.time()
        
        start_idx = np.argmin(np.abs(beat_times - start_sec))
        snapped_start = beat_times[start_idx]
        end_idx = np.argmin(np.abs(beat_times - (snapped_start + window_sec)))
        snapped_end = beat_times[end_idx]
        
        if snapped_end - snapped_start < 5.0:
            continue
            
        y_chunk = y[int(snapped_start * sr):int(snapped_end * sr)]
        
        # Per segment feature extraction
        te0 = time.time()
        chunk_features = extract_features_from_array(y_chunk, sr)
        extract_times.append(time.time() - te0)
        
        if chunk_features:
            chunk_features['target_year'] = 2024
            chunk_features['duration_ms'] = total_duration_sec * 1000
            
            tx0 = time.time()
            seg_result = predictor.predict_song_hit_probability(chunk_features)
            xgb_times.append(time.time() - tx0)
            
            to0 = time.time()
            chunk_onset = librosa.onset.onset_strength(y=y_chunk, sr=sr)
            novelty = np.clip(np.mean(chunk_onset) / 10.0, 0, 1)
            onset_times.append(time.time() - to0)
            
        loop_times.append(time.time() - t_iter)
        
        if count <= 5:
            print(f"  Iteration {count}: {loop_times[-1]:.3f}s (Extract: {extract_times[-1]:.3f}s, XGB: {xgb_times[-1]:.3f}s)")
            
    t_loop_total = time.time() - t_loop_start
    print(f"\n[7] Segment Loop Total ({count} chunks): {t_loop_total:.3f}s")
    if count > 0:
        print(f"  - Average iter time: {np.mean(loop_times):.3f}s")
        print(f"  - Average per-segment extract: {np.mean(extract_times):.3f}s")
        print(f"  - Average per-segment XGB: {np.mean(xgb_times):.3f}s")
        print(f"  - Average per-segment onset: {np.mean(onset_times):.3f}s")
    
    print("\n--- Summary ---")
    print(f"Total Audit Time: {t_load + t_global + t_xgb + t_beat + t_rms + t_loop_total:.3f}s")
    print(f"Estimated Time for 3-minute song: {t_load + t_global + t_xgb + t_beat + t_rms + (33 * np.mean(loop_times)):.3f}s")

if __name__ == "__main__":
    import soundfile as sf
    print("Creating 180s dummy audio with simulated beats...")
    sr = 22050
    duration = 180
    t = np.arange(duration * sr) / float(sr)
    # Base noise
    y = np.random.randn(duration * sr) * 0.1
    # Add simulated beats (every 0.5s = 120BPM)
    beat_freq = 120 / 60.0
    envelope = np.abs(np.sin(2 * np.pi * beat_freq * t))
    y = y * envelope
    sf.write("dummy_test.wav", y, sr)
    run_audit("dummy_test.wav")
