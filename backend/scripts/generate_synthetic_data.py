"""
AffectiSense — Synthetic Dataset Generator

Generates a statistically correlated synthetic dataset for multimodal depression analysis.
Instead of generating fake .mp4 files (which would require rendering photorealistic human 
faces for MediaPipe to detect), this directly generates the cached `.npy` feature vectors 
that `train.py` will load.

Correlations established:
- Depressed (PHQ > 10): 
    - Audio: Lower F0 (pitch), lower speech rate, lower energy.
    - Video: Lower facial expressiveness, lower AU change rate.
- Control (PHQ < 10):
    - Audio: Higher/normal pitch variability, normal energy.
    - Video: Normal facial movement, active Action Units.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_dataset(output_dir: str, num_samples: int = 100):
    base_dir = Path(output_dir)
    cache_dir = base_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    print(f"Generating {num_samples} synthetic patient records...")
    
    for i in tqdm(range(num_samples)):
        participant_id = f"SYNTH_{i:04d}"
        
        # 30% chance of being depressed
        is_depressed = np.random.rand() < 0.3
        
        if is_depressed:
            phq8_score = np.random.randint(10, 25) # Moderate to Severe
            binary = 1
            
            # Depressed Audio Profile (136 dims)
            audio_feats = np.random.normal(loc=0.0, scale=0.5, size=136)
            audio_feats[0] = np.random.normal(15.0, 2.0)  # Low f0_std (monotone)
            audio_feats[1] = np.random.normal(0.005, 0.001) # Low rms_mean
            audio_feats[2] = np.random.normal(600, 100) # Slow speech rate
            
            # Depressed Video Profile (16 dims)
            video_feats = np.random.normal(loc=0.0, scale=0.1, size=16)
            video_feats[0] = np.random.normal(0.002, 0.001) # Low facial expressiveness
            video_feats[1] = np.random.normal(10.0, 2.0) # Altered blink rate
            video_feats[3] = np.random.normal(0.002, 0.001) # Minimal head movement
            video_feats[4] = np.random.normal(0.0005, 0.0001) # Low AU change rate
            
        else:
            phq8_score = np.random.randint(0, 10) # None to Mild
            binary = 0
            
            # Control Audio Profile (136 dims)
            audio_feats = np.random.normal(loc=0.5, scale=0.5, size=136)
            audio_feats[0] = np.random.normal(30.0, 5.0)  # High f0_std (expressive)
            audio_feats[1] = np.random.normal(0.02, 0.005) # Normal rms_mean
            audio_feats[2] = np.random.normal(1500, 300) # Normal speech rate
            
            # Control Video Profile (16 dims)
            video_feats = np.random.normal(loc=0.5, scale=0.2, size=16)
            video_feats[0] = np.random.normal(0.015, 0.005) # Normal facial expressiveness
            video_feats[1] = np.random.normal(20.0, 4.0) # Normal blink rate
            video_feats[3] = np.random.normal(0.015, 0.005) # Normal head movement
            video_feats[4] = np.random.normal(0.005, 0.001) # Normal AU change rate
            
        # Simulate missing modalities (10% chance video is missing, 10% audio is missing)
        audio_missing = np.random.rand() < 0.1
        video_missing = np.random.rand() < 0.1
        if audio_missing and video_missing:
            # Prevent both from being missing
            audio_missing = False
            
        if not audio_missing:
            np.save(cache_dir / f"{participant_id}_audio.npy", audio_feats)
        if not video_missing:
            np.save(cache_dir / f"{participant_id}_video.npy", video_feats)
            
        records.append({
            "participant_id": participant_id,
            "phq8_score": phq8_score,
            "binary_diagnosis": binary
        })
        
    # Save CSV
    df = pd.DataFrame(records)
    csv_path = base_dir / "labels.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nDataset generation complete!")
    print(f"Dataset Path: {base_dir}")
    print(f"Total Samples: {len(df)} (Depressed: {df['binary_diagnosis'].sum()}, Control: {len(df) - df['binary_diagnosis'].sum()})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="synthetic_dataset")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.num_samples)
