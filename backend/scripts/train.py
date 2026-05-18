"""
AffectiSense — Training Pipeline

This script provides the production-ready infrastructure to train the 
AffectiSense multimodal attention bottleneck fusion model on a clinical dataset 
(e.g., DAIC-WOZ, AVEC).

Usage:
  python scripts/train.py --data_dir /path/to/dataset --epochs 50 --batch_size 16

Note:
  Ensure your dataset directory has a `labels.csv` with:
  participant_id, phq8_score, binary_diagnosis
  And subfolders:
  - data_dir/audio/ (containing <participant_id>.wav)
  - data_dir/video/ (containing <participant_id>.mp4)
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import f1_score, mean_squared_error

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.pipelines.audio import AudioPipeline
from app.pipelines.video import VideoPipeline
from app.models.fusion import AffectiSenseModel


class MultimodalDepressionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Audio/Video pairs.
    Pre-extracts or dynamically extracts features using the CPU pipelines.
    """
    def __init__(self, data_dir: Path, labels_df: pd.DataFrame, cache_dir: Path):
        self.data_dir = data_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_pipeline = AudioPipeline()
        self.video_pipeline = VideoPipeline()

    def __len__(self):
        return len(self.labels_df)

    def _get_cached_or_extract(self, participant_id: str, modality: str):
        cache_path = self.cache_dir / f"{participant_id}_{modality}.npy"
        if cache_path.exists():
            return torch.tensor(np.load(cache_path), dtype=torch.float32)

        # File paths
        if modality == "audio":
            file_path = self.data_dir / "audio" / f"{participant_id}.wav"
            if file_path.exists():
                feats = self.audio_pipeline.process(file_path).to_vector()
                np.save(cache_path, feats)
                return torch.tensor(feats, dtype=torch.float32)
        elif modality == "video":
            file_path = self.data_dir / "video" / f"{participant_id}.mp4"
            if file_path.exists():
                feats = self.video_pipeline.process(file_path).to_vector()
                np.save(cache_path, feats)
                return torch.tensor(feats, dtype=torch.float32)
                
        return None

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        participant_id = str(row['participant_id'])
        
        # Load features
        audio_feats = self._get_cached_or_extract(participant_id, "audio")
        video_feats = self._get_cached_or_extract(participant_id, "video")
        
        # Labels
        severity = torch.tensor([row['phq8_score'] / 24.0], dtype=torch.float32) # Normalize 0-24 -> 0-1
        binary = torch.tensor([row['binary_diagnosis']], dtype=torch.float32)
        
        return {
            "participant_id": participant_id,
            "audio_features": audio_feats if audio_feats is not None else torch.zeros(136),
            "video_features": video_feats if video_feats is not None else torch.zeros(16),
            "audio_avail": torch.tensor([audio_feats is not None], dtype=torch.bool),
            "video_avail": torch.tensor([video_feats is not None], dtype=torch.bool),
            "severity": severity,
            "binary": binary
        }


def train_model(args):
    device = torch.device(args.device)
    logger.info(f"Starting training on {device}")
    
    # 1. Initialize Model
    model = AffectiSenseModel(
        audio_input_dim=136,
        video_input_dim=16,
        embed_dim=settings.MODEL_EMBED_DIM,
        n_bottleneck=settings.MODEL_N_BOTTLENECK_TOKENS,
        n_heads=settings.MODEL_N_ATTENTION_HEADS,
        dropout=settings.MODEL_DROPOUT,
        modality_dropout_prob=settings.MODALITY_DROPOUT_PROB,
    ).to(device)

    # 2. Setup Data (Mocked paths for script completeness)
    data_dir = Path(args.data_dir)
    labels_csv = data_dir / "labels.csv"
    if not labels_csv.exists():
        logger.error(f"Labels CSV not found at {labels_csv}. Please ensure dataset is prepared.")
        return

    df = pd.read_csv(labels_csv)
    # Split 80/20
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = MultimodalDepressionDataset(data_dir, train_df, data_dir / "cache")
    val_dataset = MultimodalDepressionDataset(data_dir, val_df, data_dir / "cache")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Loss & Optimizer
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            optimizer.zero_grad()
            
            # Simulate modality missingness via dataset flags if needed, 
            # but model internally handles Modality Dropout (p=0.3) natively in forward pass!
            audio = batch["audio_features"].to(device)
            video = batch["video_features"].to(device)
            
            # Apply missing mask if files weren't found
            audio[~batch["audio_avail"].squeeze()] = 0.0
            video[~batch["video_avail"].squeeze()] = 0.0

            out_binary, out_severity = model(audio, video)
            
            # Joint Loss: 70% Classification, 30% Severity Regression
            loss_bin = criterion_bce(out_binary, batch["binary"].to(device))
            loss_sev = criterion_mse(out_severity, batch["severity"].to(device))
            loss = 0.7 * loss_bin + 0.3 * loss_sev
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_sev_preds = []
        all_sev_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                audio = batch["audio_features"].to(device)
                video = batch["video_features"].to(device)
                
                audio[~batch["audio_avail"].squeeze()] = 0.0
                video[~batch["video_avail"].squeeze()] = 0.0

                out_binary, out_severity = model(audio, video)
                
                loss_bin = criterion_bce(out_binary, batch["binary"].to(device))
                loss_sev = criterion_mse(out_severity, batch["severity"].to(device))
                loss = 0.7 * loss_bin + 0.3 * loss_sev
                val_loss += loss.item()
                
                all_preds.extend((out_binary > 0.5).cpu().numpy().tolist())
                all_labels.extend(batch["binary"].numpy().tolist())
                all_sev_preds.extend(out_severity.cpu().numpy().tolist())
                all_sev_labels.extend(batch["severity"].numpy().tolist())
                
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds)
        val_rmse = np.sqrt(mean_squared_error(all_sev_labels, all_sev_preds))
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "affectisense_best_weights.pt")
            logger.info("Saved new best model weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to DAIC-WOZ or custom dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    train_model(args)
