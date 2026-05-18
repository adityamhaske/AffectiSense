"""
AffectiSense — Linguistic Pipeline (Fully Local)

Integrates ASR (Speech-to-Text) and NLP (LLM & Sentiment) entirely locally.
- ASR: OpenAI Whisper (base model)
- LLM: HuggingFace Transformers (TinyLlama or similar lightweight chat model)
- Sentiment: DistilBERT sentiment-analysis
- Theme Detection: Zero-shot classification for clinical themes
"""

import torch
import whisper
from transformers import pipeline
from typing import Dict, Any, List
from loguru import logger
from pydantic import BaseModel

from backend.app.core.config import settings

class LinguisticFeatures(BaseModel):
    transcript: str
    sentiment_label: str
    sentiment_score: float
    clinical_themes: Dict[str, float]
    next_question: str

class LinguisticPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model_loaded = False
        
        # We lazy-load these to save RAM until a conversation actually starts
        self.asr_model = None
        self.sentiment_model = None
        self.theme_model = None
        self.chat_model = None
        
    def _load_models(self):
        """Loads models on-demand to save initial startup RAM."""
        if self._model_loaded:
            return
            
        logger.info("Loading Local Linguistic Models (Whisper + Transformers)...")
        # 1. ASR Model
        self.asr_model = whisper.load_model("base", device=self.device)
        
        # 2. Sentiment
        self.sentiment_model = pipeline("sentiment-analysis", device=-1 if self.device == "cpu" else 0)
        
        # 3. Clinical Theme Detection (Zero-shot)
        self.theme_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1 if self.device == "cpu" else 0)
        
        # 4. LLM Chat Generator (TinyLlama is lightweight ~1.1B params)
        # Using a highly compressed/quantized version or just a standard text-generation pipeline.
        self.chat_model = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=-1 if self.device == "cpu" else 0,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            pad_token_id=50256 # Default eos for most
        )
        self._model_loaded = True
        logger.info("Local Linguistic Models Loaded!")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        if not self._model_loaded:
            self._load_models()
        
        logger.info(f"Transcribing {audio_path}...")
        result = self.asr_model.transcribe(audio_path, fp16=False)
        return result["text"].strip()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Run sentiment and clinical theme detection on the transcript."""
        if not text:
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "themes": {}}
            
        if not self._model_loaded:
            self._load_models()
            
        # 1. Sentiment
        sentiment_out = self.sentiment_model(text[:512])[0] # Truncate to avoid length errors
        
        # 2. Clinical Themes
        candidate_labels = ["hopelessness", "fatigue", "anxiety", "worthlessness", "loss of interest", "insomnia"]
        theme_out = self.theme_model(text[:512], candidate_labels)
        
        themes = {label: score for label, score in zip(theme_out["labels"], theme_out["scores"])}
        
        return {
            "sentiment": sentiment_out,
            "themes": themes
        }
        
    def generate_next_question(self, context_history: List[Dict[str, str]], current_transcript: str) -> str:
        """
        Use the local LLM to generate an empathetic follow-up question.
        context_history: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        if not self._model_loaded:
            self._load_models()
            
        # Build prompt using TinyLlama chat template format
        prompt = "<|system|>\nYou are an empathetic clinical assistant conducting a mental health screening. Respond with a single, concise follow-up question based on the user's input. Do not diagnose.</s>\n"
        
        for msg in context_history[-3:]: # Keep context small for memory
            role = msg.get("role", "user")
            prompt += f"<|{role}|>\n{msg['content']}</s>\n"
            
        prompt += f"<|user|>\n{current_transcript}</s>\n<|assistant|>\n"
        
        out = self.chat_model(
            prompt,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False,
            truncation=True,
            pad_token_id=self.chat_model.tokenizer.eos_token_id
        )
        
        generated_text = out[0]["generated_text"].strip()
        
        # Cleanup potential generation artifacts
        if "</s>" in generated_text:
            generated_text = generated_text.split("</s>")[0]
            
        return generated_text

    def process(self, audio_path: str, context_history: List[Dict[str, str]] = None) -> LinguisticFeatures:
        """Run full linguistic pipeline on an audio file."""
        if context_history is None:
            context_history = []
            
        transcript = self.transcribe(audio_path)
        
        if transcript:
            analysis = self.analyze_text(transcript)
            next_question = self.generate_next_question(context_history, transcript)
        else:
            analysis = {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "themes": {}}
            next_question = "I didn't quite catch that. Could you tell me how you're feeling today?"
            
        return LinguisticFeatures(
            transcript=transcript,
            sentiment_label=analysis["sentiment"]["label"],
            sentiment_score=analysis["sentiment"]["score"],
            clinical_themes=analysis["themes"],
            next_question=next_question
        )
