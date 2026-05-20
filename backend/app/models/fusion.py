"""
AffectiSense — Multimodal Fusion Model

Implements Attention Bottleneck Fusion with Modality Dropout for
modality-resilient depression screening. Supports any combination
of available modalities (Audio, Video, or both).

Architecture:
  Per-modality encoders → Modality tokens + availability embeddings
  → Cross-attention bottleneck → Classification + Severity + Confidence heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from backend.app.core.config import settings


class ModalityEncoder(nn.Module):
    """Lightweight MLP encoder for a single modality's feature vector."""

    def __init__(self, input_dim: int, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBottleneckFusion(nn.Module):
    """
    Cross-modal attention bottleneck fusion with modality dropout.

    Forces information through a constrained set of bottleneck tokens,
    distilling complementary cross-modal signals while remaining resilient
    to missing modalities via learned availability embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_bottleneck: int = 32,
        n_heads: int = 8,
        n_modalities: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities

        # Learnable bottleneck tokens
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(n_bottleneck, embed_dim) * 0.02
        )

        # Per-modality type tokens
        self.modality_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, embed_dim) * 0.02)
            for _ in range(n_modalities)
        ])

        # Availability embeddings (0 = missing, 1 = present)
        self.availability_embedding = nn.Embedding(2, embed_dim)

        # Cross-attention: bottleneck queries attend to modality tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Self-attention within bottleneck
        self.self_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        embeddings: list[Optional[torch.Tensor]],
        available_mask: list[bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse available modality embeddings through attention bottleneck.

        Args:
            embeddings: List of [B, D] tensors per modality (None if missing)
            available_mask: Boolean mask for which modalities are present

        Returns:
            fused: [B, D] fused representation (mean-pooled bottleneck)
            attn_weights: [B, n_bottleneck, n_tokens] attention weights
        """
        batch_size = None
        for emb in embeddings:
            if emb is not None:
                batch_size = emb.shape[0]
                break
        assert batch_size is not None, "At least one modality must be available"

        # Build input tokens: modality_embedding + type_token + availability_emb
        tokens = []
        for i in range(self.n_modalities):
            avail = torch.tensor(
                [1 if available_mask[i] else 0],
                device=self.bottleneck_tokens.device,
            )
            avail_emb = self.availability_embedding(avail)  # [1, D]

            if available_mask[i] and embeddings[i] is not None:
                # Use actual modality embedding
                token = (
                    embeddings[i]
                    + self.modality_tokens[i].expand(batch_size, -1)
                    + avail_emb.expand(batch_size, -1)
                )
            else:
                # Use learned placeholder for missing modality
                token = (
                    self.modality_tokens[i].expand(batch_size, -1)
                    + avail_emb.expand(batch_size, -1)
                )
            tokens.append(token.unsqueeze(1))  # [B, 1, D]

        # Concatenate all modality tokens: [B, n_modalities, D]
        kv = torch.cat(tokens, dim=1)

        # Bottleneck queries: [B, n_bottleneck, D]
        q = self.bottleneck_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: bottleneck attends to modality tokens
        attn_out, attn_weights = self.cross_attention(q, kv, kv)
        q = self.cross_norm(q + attn_out)

        # Self-attention within bottleneck
        self_attn_out, _ = self.self_attention(q, q, q)
        q = self.self_norm(q + self_attn_out)

        # FFN
        q = self.ffn_norm(q + self.ffn(q))

        # Pool bottleneck tokens to single vector
        fused = q.mean(dim=1)  # [B, D]

        return fused, attn_weights


class AffectiSenseModel(nn.Module):
    """
    Full AffectiSense multimodal depression screening model.

    Components:
      - Per-modality encoders (Audio, Video)
      - Attention bottleneck fusion
      - Classification head (binary: depressed/control)
      - Severity regression head (continuous 0-1 score)
      - Confidence head (epistemic uncertainty via MC Dropout)
    """

    def __init__(
        self,
        audio_input_dim: int = 119,
        video_input_dim: int = 38,
        embed_dim: int = 256,
        n_bottleneck: int = 32,
        n_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob

        # Per-modality encoders
        self.audio_encoder = ModalityEncoder(audio_input_dim, embed_dim, dropout)
        self.video_encoder = ModalityEncoder(video_input_dim, embed_dim, dropout)

        # Fusion core
        self.fusion = AttentionBottleneckFusion(
            embed_dim=embed_dim,
            n_bottleneck=n_bottleneck,
            n_heads=n_heads,
            n_modalities=2,
            dropout=dropout,
        )

        # Classification head (binary: depressed / control)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        # Severity regression head (PHQ-8 normalized to 0-1)
        self.severity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional modality dropout.

        Args:
            audio_features: [B, audio_dim] or None
            video_features: [B, video_dim] or None

        Returns:
            Dict with 'logits', 'severity', 'fused', 'attn_weights',
            'available_mask'
        """
        available = [audio_features is not None, video_features is not None]

        # Modality dropout during training
        if self.training:
            available, audio_features, video_features = self._apply_modality_dropout(
                available, audio_features, video_features
            )

        # Encode available modalities
        embeddings = [None, None]
        if available[0] and audio_features is not None:
            embeddings[0] = self.audio_encoder(audio_features)
        if available[1] and video_features is not None:
            embeddings[1] = self.video_encoder(video_features)

        # Fuse
        fused, attn_weights = self.fusion(embeddings, available)

        # Predict
        logits = self.classifier(fused).squeeze(-1)
        severity = self.severity_head(fused).squeeze(-1)

        return {
            "logits": logits,
            "severity": severity,
            "fused": fused,
            "attn_weights": attn_weights,
            "available_mask": available,
        }

    def _apply_modality_dropout(self, available, audio_features, video_features):
        """Randomly drop modalities during training to build resilience."""
        if np.random.random() < self.modality_dropout_prob:
            n_available = sum(available)
            if n_available > 1:
                drop_idx = np.random.randint(0, len(available))
                available[drop_idx] = False
                if drop_idx == 0:
                    audio_features = None
                else:
                    video_features = None
        return available, audio_features, video_features

    @torch.no_grad()
    def predict_with_confidence(
        self,
        audio_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        n_samples: int = 20,
    ) -> dict:
        """
        Monte Carlo Dropout inference for calibrated confidence.

        Runs N forward passes with dropout enabled, computing mean prediction
        and epistemic uncertainty from the variance.
        """
        self.train()  # Enable dropout for MC sampling

        predictions = []
        severities = []
        for _ in range(n_samples):
            out = self.forward(audio_features, video_features)
            predictions.append(torch.sigmoid(out["logits"]))
            severities.append(out["severity"])

        self.eval()

        preds = torch.stack(predictions)
        sevs = torch.stack(severities)

        mean_pred = preds.mean(0)
        epistemic_uncertainty = preds.std(0)

        n_available = sum([
            audio_features is not None,
            video_features is not None,
        ])
        completeness = n_available / 2.0
        confidence = (1.0 - epistemic_uncertainty) * completeness

        return {
            "depression_probability": mean_pred,
            "severity": sevs.mean(0),
            "confidence": confidence,
            "epistemic_uncertainty": epistemic_uncertainty,
            "modality_completeness": completeness,
        }
