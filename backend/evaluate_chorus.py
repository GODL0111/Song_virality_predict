import librosa
import numpy as np

def calculate_chorus_similarity(y, sr, start_sample, end_sample, full_chroma):
    chunk_chroma = librosa.feature.chroma_stft(y=y[start_sample:end_sample], sr=sr)
    chunk_mean = np.mean(chunk_chroma, axis=1)
    full_mean = np.mean(full_chroma, axis=1)
    
    dot = np.dot(chunk_mean, full_mean)
    norm = np.linalg.norm(chunk_mean) * np.linalg.norm(full_mean)
    if norm == 0:
        return 0.0
    return max(0.0, float(dot / norm))

y, sr = librosa.load("dummy_test.wav", sr=22050, mono=True)
total_duration_sec = librosa.get_duration(y=y, sr=sr)

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo_track, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

full_chroma = librosa.feature.chroma_stft(y=y, sr=sr)

window_sec = 15.0
stride_sec = 5.0

raw_segments = []

for start_sec in np.arange(0, total_duration_sec - window_sec, stride_sec):
    start_idx = np.argmin(np.abs(beat_times - start_sec))
    snapped_start = beat_times[start_idx]
    
    end_idx = np.argmin(np.abs(beat_times - (snapped_start + window_sec)))
    snapped_end = beat_times[end_idx]
    
    if snapped_end - snapped_start < 5.0:
        continue
        
    start_sample = int(snapped_start * sr)
    end_sample = int(snapped_end * sr)
    
    chorus_sim = calculate_chorus_similarity(y, sr, start_sample, end_sample, full_chroma)
    y_chunk = y[start_sample:end_sample]
    chunk_rms = librosa.feature.rms(y=y_chunk)[0]
    chunk_onset = librosa.onset.onset_strength(y=y_chunk, sr=sr)
    
    raw_segments.append({
        'start_time': round(snapped_start, 1),
        'end_time': round(snapped_end, 1),
        'chorus_sim': chorus_sim,
        'energy_raw': np.mean(chunk_rms),
        'loudness_raw': 20 * np.log10(np.mean(chunk_rms) + 1e-10),
    })

# Normalize chorus sim
vals = [seg['chorus_sim'] for seg in raw_segments]
min_v, max_v = min(vals), max(vals)
for seg in raw_segments:
    seg['norm_chorus_sim'] = (seg['chorus_sim'] - min_v) / (max_v - min_v)

# Sort by normalized chorus sim
sorted_segs = sorted(raw_segments, key=lambda x: x['norm_chorus_sim'], reverse=True)

print(f"Total Segments: {len(raw_segments)}")
print("--- TOP 5 CHORUS CANDIDATES ---")
for i, seg in enumerate(sorted_segs[:5]):
    print(f"#{i+1}: {seg['start_time']}s - {seg['end_time']}s | Norm Sim: {seg['norm_chorus_sim']:.3f} | Raw Sim: {seg['chorus_sim']:.3f} | Energy: {seg['energy_raw']:.3f}")

p75 = np.percentile([seg['norm_chorus_sim'] for seg in raw_segments], 75)
print(f"\n75th Percentile Threshold: {p75:.3f}")
count_p75 = sum(1 for seg in raw_segments if seg['norm_chorus_sim'] >= p75)
print(f"Segments passing 75th percentile: {count_p75}")
