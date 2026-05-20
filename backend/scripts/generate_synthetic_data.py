"""
AffectiSense — Synthetic Dataset Generator (v2 — Enhanced Features)

Generates a statistically correlated synthetic dataset for multimodal depression analysis.
Produces cached `.npy` feature vectors matching the enhanced pipeline dimensions:
  - Audio: 119 features (MFCCs, spectral, prosodic, jitter, shimmer, HNR, pause, tremor, energy)
  - Video: 38 features (AU dynamics, gaze, asymmetry, smile, micro-expressions, head stability)

Correlations established:
- Depressed (PHQ > 10): 
    Audio: Lower F0, lower speech rate, lower energy, higher jitter/shimmer, more pauses, voice tremor
    Video: Lower expressiveness, gaze avoidance, higher asymmetry, fewer smiles, psychomotor retardation
- Control (PHQ < 10):
    Audio: Normal pitch variability, normal energy, low jitter/shimmer
    Video: Normal facial movement, stable gaze, symmetric expressions, frequent smiles
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

AUDIO_DIM = 119
VIDEO_DIM = 38

def generate_dataset(output_dir: str, num_samples: int = 200):
    base_dir = Path(output_dir)
    cache_dir = base_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    print(f"Generating {num_samples} synthetic patient records (Audio:{AUDIO_DIM}d, Video:{VIDEO_DIM}d)...")
    
    for i in tqdm(range(num_samples)):
        participant_id = f"SYNTH_{i:04d}"
        
        # 30% chance of being depressed
        is_depressed = np.random.rand() < 0.3
        
        if is_depressed:
            phq8_score = np.random.randint(10, 25)
            binary = 1
            
            # === Depressed Audio Profile (119 dims) ===
            audio_feats = np.random.normal(loc=-0.2, scale=0.3, size=AUDIO_DIM)
            # MFCCs (0-59): slightly lower energy pattern
            audio_feats[:60] = np.random.normal(loc=-0.3, scale=0.4, size=60)
            # Spectral (60-87): flatter spectrum
            audio_feats[60:88] = np.random.normal(loc=0.1, scale=0.2, size=28)
            # F0 features (88-91): low variability
            audio_feats[88] = np.random.normal(15.0, 2.0)   # f0_std low (monotone)
            audio_feats[89] = np.random.normal(0.005, 0.001) # rms_mean low
            audio_feats[90] = np.random.normal(600, 100)     # speech rate slow
            # New clinical features (91-118):
            audio_feats[91] = np.random.normal(0.04, 0.01)   # jitter HIGH
            audio_feats[92] = np.random.normal(0.08, 0.02)   # shimmer HIGH
            audio_feats[93] = np.random.normal(8.0, 2.0)     # hnr_db LOW
            audio_feats[94] = np.random.normal(12.0, 3.0)    # pause_count HIGH
            audio_feats[95] = np.random.normal(8.0, 2.0)     # pause_total HIGH
            audio_feats[96] = np.random.normal(0.7, 0.1)     # pause_mean HIGH
            audio_feats[97] = np.random.normal(5.5, 0.5)     # tremor_freq present
            audio_feats[98] = np.random.normal(0.3, 0.1)     # tremor_magnitude HIGH
            audio_feats[99] = np.random.normal(0.002, 0.001)  # energy_std LOW
            audio_feats[100] = np.random.normal(0.005, 0.001) # energy_range LOW
            audio_feats[101] = np.random.normal(5.0, 1.0)     # energy_dynamic_range LOW
            
            # === Depressed Video Profile (38 dims) ===
            video_feats = np.random.normal(loc=0.0, scale=0.05, size=VIDEO_DIM)
            # Original features (0-15)
            video_feats[0] = np.random.normal(10.0, 2.0)    # blink_rate altered
            video_feats[1] = np.random.normal(0.01, 0.002)   # eye_openness low
            video_feats[12] = np.random.normal(0.002, 0.001)  # expressiveness LOW
            video_feats[14] = np.random.normal(0.0005, 0.0001) # au_change_rate LOW
            # New features (16-37): gaze, asymmetry, smile, micro-expr, head stability
            video_feats[16] = np.random.normal(0.2, 0.05)    # gaze_h: off-center
            video_feats[17] = np.random.normal(-0.15, 0.05)  # gaze_v: downward
            video_feats[18] = np.random.normal(0.15, 0.03)   # gaze_h_std: unstable
            video_feats[19] = np.random.normal(0.12, 0.03)   # gaze_v_std
            video_feats[20] = np.random.normal(0.08, 0.02)   # gaze_stability LOW
            video_feats[21] = np.random.normal(0.4, 0.1)     # gaze_avoidance HIGH
            video_feats[22] = np.random.normal(0.15, 0.03)   # eye_asymmetry HIGH
            video_feats[23] = np.random.normal(0.12, 0.03)   # brow_asymmetry HIGH
            video_feats[24] = np.random.normal(0.1, 0.03)    # mouth_asymmetry HIGH
            video_feats[25] = np.random.normal(0.12, 0.03)   # overall_asymmetry HIGH
            video_feats[26] = np.random.normal(0.01, 0.005)  # smile_intensity LOW
            video_feats[27] = np.random.normal(0.005, 0.002) # smile_std LOW
            video_feats[28] = np.random.normal(1.0, 0.5)     # smile_frequency LOW
            video_feats[29] = np.random.normal(2.0, 1.0)     # smile_duration LOW
            video_feats[30] = np.random.normal(1.0, 0.5)     # micro_expr_count LOW
            video_feats[31] = np.random.normal(0.05, 0.02)   # micro_expr_rate LOW
            video_feats[32] = np.random.normal(0.01, 0.005)  # head_pitch_std LOW (rigid)
            video_feats[33] = np.random.normal(0.01, 0.005)  # head_yaw_std LOW
            video_feats[34] = np.random.normal(0.005, 0.002) # head_roll_std LOW
            video_feats[35] = np.random.normal(0.03, 0.01)   # head_pitch_range LOW
            video_feats[36] = np.random.normal(0.03, 0.01)   # head_yaw_range LOW
            video_feats[37] = np.random.normal(0.02, 0.005)  # head_roll_range LOW
            
        else:
            phq8_score = np.random.randint(0, 10)
            binary = 0
            
            # === Control Audio Profile (119 dims) ===
            audio_feats = np.random.normal(loc=0.3, scale=0.4, size=AUDIO_DIM)
            audio_feats[:60] = np.random.normal(loc=0.2, scale=0.5, size=60)
            audio_feats[60:88] = np.random.normal(loc=0.4, scale=0.3, size=28)
            audio_feats[88] = np.random.normal(30.0, 5.0)    # f0_std high (expressive)
            audio_feats[89] = np.random.normal(0.02, 0.005)   # rms_mean normal
            audio_feats[90] = np.random.normal(1500, 300)      # speech rate normal
            audio_feats[91] = np.random.normal(0.01, 0.003)    # jitter LOW
            audio_feats[92] = np.random.normal(0.02, 0.005)    # shimmer LOW
            audio_feats[93] = np.random.normal(18.0, 3.0)      # hnr_db HIGH
            audio_feats[94] = np.random.normal(4.0, 1.5)       # pause_count LOW
            audio_feats[95] = np.random.normal(2.0, 0.5)       # pause_total LOW
            audio_feats[96] = np.random.normal(0.3, 0.05)      # pause_mean LOW
            audio_feats[97] = np.random.normal(0.0, 0.5)       # tremor_freq absent
            audio_feats[98] = np.random.normal(0.05, 0.02)     # tremor_magnitude LOW
            audio_feats[99] = np.random.normal(0.008, 0.002)   # energy_std normal
            audio_feats[100] = np.random.normal(0.02, 0.005)   # energy_range normal
            audio_feats[101] = np.random.normal(15.0, 3.0)     # energy_dynamic_range normal
            
            # === Control Video Profile (38 dims) ===
            video_feats = np.random.normal(loc=0.3, scale=0.15, size=VIDEO_DIM)
            video_feats[0] = np.random.normal(18.0, 3.0)
            video_feats[1] = np.random.normal(0.03, 0.005)
            video_feats[12] = np.random.normal(0.015, 0.005)
            video_feats[14] = np.random.normal(0.005, 0.001)
            video_feats[16] = np.random.normal(0.0, 0.03)     # gaze centered
            video_feats[17] = np.random.normal(0.0, 0.03)
            video_feats[18] = np.random.normal(0.05, 0.02)
            video_feats[19] = np.random.normal(0.04, 0.02)
            video_feats[20] = np.random.normal(0.15, 0.03)    # gaze_stability HIGH
            video_feats[21] = np.random.normal(0.1, 0.03)     # gaze_avoidance LOW
            video_feats[22] = np.random.normal(0.03, 0.01)    # eye_asymmetry LOW
            video_feats[23] = np.random.normal(0.03, 0.01)
            video_feats[24] = np.random.normal(0.02, 0.01)
            video_feats[25] = np.random.normal(0.03, 0.01)
            video_feats[26] = np.random.normal(0.06, 0.02)    # smile_intensity HIGH
            video_feats[27] = np.random.normal(0.02, 0.005)
            video_feats[28] = np.random.normal(8.0, 2.0)      # smile_frequency HIGH
            video_feats[29] = np.random.normal(10.0, 3.0)     # smile_duration HIGH
            video_feats[30] = np.random.normal(5.0, 1.5)      # micro_expr count
            video_feats[31] = np.random.normal(0.2, 0.05)
            video_feats[32] = np.random.normal(0.04, 0.01)    # head_pitch_std normal
            video_feats[33] = np.random.normal(0.04, 0.01)
            video_feats[34] = np.random.normal(0.02, 0.005)
            video_feats[35] = np.random.normal(0.1, 0.03)
            video_feats[36] = np.random.normal(0.1, 0.03)
            video_feats[37] = np.random.normal(0.05, 0.01)
            
        # Simulate missing modalities (10% chance each)
        audio_missing = np.random.rand() < 0.1
        video_missing = np.random.rand() < 0.1
        if audio_missing and video_missing:
            audio_missing = False
            
        if not audio_missing:
            np.save(cache_dir / f"{participant_id}_audio.npy", audio_feats.astype(np.float32))
        if not video_missing:
            np.save(cache_dir / f"{participant_id}_video.npy", video_feats.astype(np.float32))
            
        records.append({
            "participant_id": participant_id,
            "phq8_score": phq8_score,
            "binary_diagnosis": binary
        })
        
    df = pd.DataFrame(records)
    csv_path = base_dir / "labels.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nDataset generation complete!")
    print(f"  Path: {base_dir}")
    print(f"  Audio dim: {AUDIO_DIM}, Video dim: {VIDEO_DIM}")
    print(f"  Total: {len(df)} (Depressed: {df['binary_diagnosis'].sum()}, Control: {len(df) - df['binary_diagnosis'].sum()})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="synthetic_dataset")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.num_samples)
