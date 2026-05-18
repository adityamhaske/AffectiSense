# AffectiSense — Pydantic Response/Request Schemas
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Modality(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    EEG = "eeg"


class SeverityLevel(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"


class ModalityScore(BaseModel):
    """Per-modality prediction breakdown."""
    modality: Modality
    available: bool
    prediction_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Raw model output probability")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated confidence for this modality")
    key_indicators: list[str] = Field(default_factory=list, description="Human-readable feature explanations")


class PredictionResponse(BaseModel):
    """Full prediction response from the AffectiSense engine."""
    # --- Core Prediction ---
    is_model_trained: bool = Field(default=True, description="True if trained weights were used, False if only biomarkers are returned")
    prediction: Optional[str] = Field(default=None, description="Binary label: 'depressed' or 'control'")
    depression_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Fused probability of depression")
    severity_level: Optional[SeverityLevel] = Field(default=None, description="Estimated severity category")
    severity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Continuous severity estimate (0=none, 1=severe)")

    # --- Confidence ---
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence in the prediction")
    modality_completeness: float = Field(ge=0.0, le=1.0, description="Fraction of modalities available (0.5 for 1/2, 1.0 for 2/2)")

    # --- Per-Modality Breakdown ---
    modality_scores: list[ModalityScore] = Field(description="Per-modality prediction details")

    # --- Explainability ---
    clinical_summary: str = Field(description="Natural language clinical interpretation")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk indicators")
    protective_factors: list[str] = Field(default_factory=list, description="Identified protective indicators")

    # --- Metadata ---
    modalities_used: list[Modality] = Field(description="Which modalities were actually used")
    processing_time_ms: float = Field(description="Total inference time in milliseconds")
    model_version: str = Field(default="1.0.0")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    models_loaded: bool
    available_modalities: list[str]


class AnalysisRequest(BaseModel):
    """Request body for analysis (when not using file upload)."""
    modalities: list[Modality] = Field(description="Which modalities to analyze")
    session_id: Optional[str] = None
