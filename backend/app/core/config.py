# NeuroSense AI — Application Configuration
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Central configuration for the NeuroSense AI platform."""

    # --- Application ---
    APP_NAME: str = "NeuroSense AI"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Multimodal Mental Health Assessment Platform"
    DEBUG: bool = True

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    MODEL_DIR: Path = BASE_DIR / "data" / "models"
    TEMP_DIR: Path = BASE_DIR / "data" / "temp"

    # --- Audio Processing ---
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_MAX_DURATION_SEC: int = 300  # 5 minutes max
    AUDIO_MIN_DURATION_SEC: int = 5
    AUDIO_N_MFCC: int = 20
    AUDIO_N_MELS: int = 128
    AUDIO_HOP_LENGTH: int = 512
    AUDIO_N_FFT: int = 2048

    # --- Video Processing ---
    VIDEO_MAX_DURATION_SEC: int = 300
    VIDEO_FPS_TARGET: int = 5  # Downsample to 5 FPS for efficiency
    VIDEO_WINDOW_SEC: float = 2.0  # 2-second temporal windows
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5

    # --- Model ---
    MODEL_EMBED_DIM: int = 256
    MODEL_N_BOTTLENECK_TOKENS: int = 32
    MODEL_N_ATTENTION_HEADS: int = 8
    MODEL_DROPOUT: float = 0.1
    MODALITY_DROPOUT_PROB: float = 0.3  # During training
    MC_DROPOUT_SAMPLES: int = 20  # For confidence estimation

    # --- Inference ---
    DEVICE: str = "cpu"
    MAX_BATCH_SIZE: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
