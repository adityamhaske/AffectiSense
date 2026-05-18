"""
AffectiSense — FastAPI Application

Main API server with endpoints for:
  - Health checks
  - Multimodal file upload (audio + video)
  - Depression screening inference
  - Model status
"""

import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    PredictionResponse, HealthResponse, Modality,
)
from backend.app.services.inference import InferenceService
from backend.app.pipelines.linguistic import LinguisticPipeline
import json

# --- Globals ---
inference_service = InferenceService()
linguistic_pipeline = LinguisticPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    inference_service.load_model()
    logger.info("Model loaded and ready for inference")
    yield
    logger.info("Shutting down AffectiSense")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Utility ---
def _save_upload(upload: UploadFile, suffix: str) -> Path:
    """Save an uploaded file to a temporary location and return the path."""
    file_id = uuid.uuid4().hex[:12]
    dest = settings.TEMP_DIR / f"{file_id}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with platform info."""
    return {
        "platform": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inference_service.is_ready else "loading",
        version=settings.APP_VERSION,
        models_loaded=inference_service.is_ready,
        available_modalities=["audio", "video"],
    )


@app.post("/api/v1/analyze", response_model=PredictionResponse, tags=["inference"])
async def analyze(
    audio: Optional[UploadFile] = File(None, description="Audio file (.wav, .mp3, .flac)"),
    video: Optional[UploadFile] = File(None, description="Video file (.mp4, .avi, .mov)"),
):
    """
    Multimodal depression screening analysis.

    Upload audio and/or video files for analysis. The system supports
    any combination of modalities and will produce calibrated predictions
    with confidence scores indicating reliability.

    **Supported modality combinations:**
    - Audio only
    - Video only
    - Audio + Video (best accuracy)
    """
    if audio is None and video is None:
        raise HTTPException(
            status_code=400,
            detail="At least one modality (audio or video) must be provided.",
        )

    audio_path = None
    video_path = None

    try:
        # Save uploaded files
        if audio is not None:
            audio_ext = Path(audio.filename or "audio.wav").suffix or ".wav"
            audio_path = _save_upload(audio, audio_ext)
            logger.info(f"Audio uploaded: {audio.filename} → {audio_path}")

        if video is not None:
            video_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
            video_path = _save_upload(video, video_ext)
            logger.info(f"Video uploaded: {video.filename} → {video_path}")

        # Run inference
        result = await inference_service.predict(
            audio_path=audio_path,
            video_path=video_path,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Clean up temporary files
        if audio_path and audio_path.exists():
            audio_path.unlink()
        if video_path and video_path.exists():
            video_path.unlink()


@app.get("/api/v1/modalities", tags=["info"])
async def list_modalities():
    """List supported modalities and their current status."""
    return {
        "modalities": [
            {
                "name": "audio",
                "status": "active",
                "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                "max_duration_sec": settings.AUDIO_MAX_DURATION_SEC,
                "features": "MFCC, spectral, prosodic, pitch (F0)",
            },
            {
                "name": "video",
                "status": "active",
                "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
                "max_duration_sec": settings.VIDEO_MAX_DURATION_SEC,
                "features": "FaceMesh landmarks, AU dynamics, blink rate, head pose",
            },
            {
                "name": "eeg",
                "status": "coming_soon",
                "supported_formats": [".edf", ".mat"],
                "features": "EEGNet + Graph Attention Network (Phase 2)",
            },
        ]
    }


@app.post("/api/v1/interview/process", tags=["interview"])
async def process_interview_segment(
    audio: UploadFile = File(..., description="Audio segment of the user's response"),
    history: str = Form("[]", description="JSON string of conversation history [{'role': '...', 'content': '...'}]")
):
    """
    Process a conversational segment using the Linguistic Pipeline (Whisper + LLM).
    Returns the transcript, sentiment, clinical themes, and the generated next question.
    """
    audio_path = None
    try:
        context_history = json.loads(history)
        
        audio_ext = Path(audio.filename or "audio.wav").suffix or ".wav"
        audio_path = _save_upload(audio, audio_ext)
        logger.info(f"Interview Audio uploaded: {audio.filename} → {audio_path}")
        
        # Run linguistic pipeline
        features = linguistic_pipeline.process(str(audio_path), context_history)
        
        return {
            "transcript": features.transcript,
            "sentiment_label": features.sentiment_label,
            "sentiment_score": float(features.sentiment_score),
            "clinical_themes": features.clinical_themes,
            "next_question": features.next_question
        }
        
    except Exception as e:
        logger.exception(f"Interview processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink()
