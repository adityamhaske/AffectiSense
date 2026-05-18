"""
AffectiSense — Inference Service

Orchestrates the full inference pipeline: file handling → preprocessing →
model inference → clinical interpretation → structured response.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

from backend.app.core.config import settings
from backend.app.pipelines.audio import AudioPipeline, AudioFeatures
from backend.app.pipelines.video import VideoPipeline, VideoFeatures
from backend.app.models.fusion import AffectiSenseModel
from backend.app.schemas.prediction import (
    PredictionResponse, ModalityScore, Modality, SeverityLevel,
)


def _severity_from_score(score: float) -> SeverityLevel:
    """Map continuous severity score (0-1) to clinical category."""
    if score < 0.1:
        return SeverityLevel.NONE
    elif score < 0.25:
        return SeverityLevel.MINIMAL
    elif score < 0.4:
        return SeverityLevel.MILD
    elif score < 0.6:
        return SeverityLevel.MODERATE
    elif score < 0.8:
        return SeverityLevel.MODERATELY_SEVERE
    else:
        return SeverityLevel.SEVERE


def _generate_clinical_summary(
    prediction: str,
    severity: SeverityLevel,
    confidence: float,
    audio_indicators: list[str],
    video_indicators: list[str],
) -> str:
    """Generate a natural language clinical summary."""
    modalities_text = []
    if audio_indicators:
        modalities_text.append(f"vocal analysis ({', '.join(audio_indicators[:3])})")
    if video_indicators:
        modalities_text.append(f"facial expression analysis ({', '.join(video_indicators[:3])})")

    source = " and ".join(modalities_text) if modalities_text else "available modalities"

    if prediction == "control":
        return (
            f"Based on {source}, the subject shows indicators consistent with "
            f"non-depressed affect. Confidence: {confidence:.0%}. "
            f"This assessment should be reviewed alongside clinical interview data."
        )
    else:
        return (
            f"Based on {source}, the subject shows indicators consistent with "
            f"{severity.value.replace('_', ' ')} depressive affect. "
            f"Confidence: {confidence:.0%}. "
            f"Clinical follow-up is recommended for comprehensive evaluation."
        )

def _generate_untrained_summary(audio_indicators: list[str], video_indicators: list[str]) -> str:
    """Generate a summary explicitly stating only raw biomarkers are provided."""
    return (
        "Model Untrained. Diagnostics disabled. Displaying raw clinical biomarkers "
        "extracted from audio and/or video. Please refer to the risk and protective "
        "factors below for analysis of speech patterns and facial dynamics."
    )


def _analyze_audio_indicators(features: AudioFeatures) -> tuple[list[str], list[str]]:
    """Derive human-readable risk/protective indicators from audio features."""
    risk = []
    protective = []

    if features.f0_std < 20.0:
        risk.append("Reduced pitch variability (monotone speech)")
    else:
        protective.append("Normal pitch variability")

    if features.speech_rate_proxy < 800:
        risk.append("Slow speech rate")
    elif features.speech_rate_proxy > 2500:
        risk.append("Pressured speech rate")
    else:
        protective.append("Normal speech rate")

    if features.rms_mean < 0.01:
        risk.append("Low vocal energy")
    else:
        protective.append("Adequate vocal energy")

    if features.voiced_fraction < 0.4:
        risk.append("Long pauses / low speech activity")

    if features.spectral_flatness_mean > 0.1:
        risk.append("Breathy vocal quality")

    return risk, protective


def _analyze_video_indicators(features: VideoFeatures) -> tuple[list[str], list[str]]:
    """Derive human-readable risk/protective indicators from video features."""
    risk = []
    protective = []

    if features.facial_expressiveness < 0.005:
        risk.append("Flat affect (reduced facial expressiveness)")
    else:
        protective.append("Normal facial expressiveness")

    if features.blink_rate > 30:
        risk.append("Elevated blink rate")
    elif features.blink_rate < 5:
        risk.append("Reduced blink rate")

    if features.avg_mouth_openness < 0.005:
        risk.append("Reduced mouth movement")

    if features.head_movement_magnitude < 0.005:
        risk.append("Minimal head movement (psychomotor retardation)")
    else:
        protective.append("Normal head movement")

    if features.au_change_rate < 0.001:
        risk.append("Low facial animation rate")

    return risk, protective


class InferenceService:
    """
    Orchestrates multimodal depression screening inference.

    Handles modality routing, model loading, feature extraction,
    prediction, confidence estimation, and clinical interpretation.
    """

    def __init__(self):
        self.audio_pipeline = AudioPipeline()
        self.video_pipeline = VideoPipeline()
        self.model: Optional[AffectiSenseModel] = None
        self.device = torch.device(settings.DEVICE)
        self._model_loaded = False

    def load_model(self, checkpoint_path: Optional[Path] = None):
        """Load or initialize the fusion model."""
        self.model = AffectiSenseModel(
            audio_input_dim=136,
            video_input_dim=16,
            embed_dim=settings.MODEL_EMBED_DIM,
            n_bottleneck=settings.MODEL_N_BOTTLENECK_TOKENS,
            n_heads=settings.MODEL_N_ATTENTION_HEADS,
            dropout=settings.MODEL_DROPOUT,
            modality_dropout_prob=settings.MODALITY_DROPOUT_PROB,
        ).to(self.device)

        if checkpoint_path and checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._model_loaded = True
            logger.info(f"Loaded model checkpoint: {checkpoint_path}")
        else:
            self._model_loaded = False
            self.model = None
            logger.warning("No checkpoint found — entering Ethical Fallback mode (Biomarkers Only).")

    @property
    def is_ready(self) -> bool:
        return self._model_loaded

    async def predict(
        self,
        audio_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
    ) -> PredictionResponse:
        """
        Run full multimodal inference pipeline.

        Args:
            audio_path: Path to audio file, or None if unavailable.
            video_path: Path to video file, or None if unavailable.

        Returns:
            PredictionResponse with prediction, confidence, and explanations.
        """
        # Attempt to load model if not already checked
        if not self._model_loaded and self.model is None:
            self.load_model()

        start_time = time.time()
        modalities_used = []

        # --- Extract features ---
        audio_features_obj = None
        video_features_obj = None
        audio_tensor = None
        video_tensor = None
        audio_risk, audio_protective = [], []
        video_risk, video_protective = [], []

        if audio_path and audio_path.exists():
            try:
                audio_features_obj = self.audio_pipeline.process(audio_path)
                audio_vec = audio_features_obj.to_vector()
                audio_tensor = torch.tensor(audio_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                modalities_used.append(Modality.AUDIO)
                audio_risk, audio_protective = _analyze_audio_indicators(audio_features_obj)
            except Exception as e:
                logger.error(f"Audio processing failed: {e}")

        if video_path and video_path.exists():
            try:
                video_features_obj = self.video_pipeline.process(video_path)
                video_vec = video_features_obj.to_vector()
                video_tensor = torch.tensor(video_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                modalities_used.append(Modality.VIDEO)
                video_risk, video_protective = _analyze_video_indicators(video_features_obj)
            except Exception as e:
                logger.error(f"Video processing failed: {e}")

        if not modalities_used:
            raise ValueError("No modalities could be processed. Please provide valid audio or video input.")

        # --- Ethical Fallback (Untrained) ---
        if not self._model_loaded:
            processing_time = (time.time() - start_time) * 1000
            
            # Empty modality scores without prediction data
            modality_scores = []
            if Modality.AUDIO in modalities_used:
                modality_scores.append(ModalityScore(
                    modality=Modality.AUDIO, available=True, prediction_score=None, confidence=None, key_indicators=audio_risk[:5]
                ))
            if Modality.VIDEO in modalities_used:
                modality_scores.append(ModalityScore(
                    modality=Modality.VIDEO, available=True, prediction_score=None, confidence=None, key_indicators=video_risk[:5]
                ))

            all_risk = audio_risk + video_risk
            all_protective = audio_protective + video_protective

            return PredictionResponse(
                is_model_trained=False,
                prediction=None,
                depression_probability=None,
                severity_level=None,
                severity_score=None,
                overall_confidence=0.0,
                modality_completeness=1.0 if len(modalities_used) == 2 else 0.5,
                modality_scores=modality_scores,
                clinical_summary=_generate_untrained_summary(audio_risk, video_risk),
                risk_factors=all_risk[:10],
                protective_factors=all_protective[:10],
                modalities_used=modalities_used,
                processing_time_ms=round(processing_time, 2),
                model_version=settings.APP_VERSION,
            )

        # --- Model inference with confidence ---
        with torch.no_grad():
            result = self.model.predict_with_confidence(
                audio_features=audio_tensor,
                video_features=video_tensor,
                n_samples=settings.MC_DROPOUT_SAMPLES,
            )

        depression_prob = float(result["depression_probability"].cpu().item())
        severity_score = float(result["severity"].cpu().item())
        confidence = float(result["confidence"].cpu().item())
        completeness = result["modality_completeness"]

        prediction = "depressed" if depression_prob > 0.5 else "control"
        severity_level = _severity_from_score(severity_score)

        # --- Build per-modality scores ---
        modality_scores = []

        modality_scores.append(ModalityScore(
            modality=Modality.AUDIO,
            available=Modality.AUDIO in modalities_used,
            prediction_score=depression_prob if Modality.AUDIO in modalities_used else None,
            confidence=confidence if Modality.AUDIO in modalities_used else None,
            key_indicators=audio_risk[:5],
        ))

        modality_scores.append(ModalityScore(
            modality=Modality.VIDEO,
            available=Modality.VIDEO in modalities_used,
            prediction_score=depression_prob if Modality.VIDEO in modalities_used else None,
            confidence=confidence if Modality.VIDEO in modalities_used else None,
            key_indicators=video_risk[:5],
        ))

        # --- Clinical interpretation ---
        all_risk = audio_risk + video_risk
        all_protective = audio_protective + video_protective

        clinical_summary = _generate_clinical_summary(
            prediction, severity_level, confidence,
            audio_risk, video_risk,
        )

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            is_model_trained=True,
            prediction=prediction,
            depression_probability=round(depression_prob, 4),
            severity_level=severity_level,
            severity_score=round(severity_score, 4),
            overall_confidence=round(max(0.0, min(1.0, confidence)), 4),
            modality_completeness=completeness,
            modality_scores=modality_scores,
            clinical_summary=clinical_summary,
            risk_factors=all_risk[:10],
            protective_factors=all_protective[:10],
            modalities_used=modalities_used,
            processing_time_ms=round(processing_time, 2),
            model_version=settings.APP_VERSION,
        )
