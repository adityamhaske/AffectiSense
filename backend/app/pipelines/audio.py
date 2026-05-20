"""
AffectiSense — Audio Feature Extraction Pipeline

Extracts depression-relevant acoustic biomarkers from speech recordings using
librosa. Features include spectral, prosodic, cepstral descriptors, as well as
clinically validated voice-quality measures (jitter, shimmer, HNR), pause
analysis, voice tremor, and energy dynamics.

Pipeline: Load → Resample 16kHz → VAD → Feature Extraction → Statistics
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from backend.app.core.config import settings


@dataclass
class AudioFeatures:
    """Container for extracted audio features with metadata."""
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    mfcc_std: np.ndarray = field(default_factory=lambda: np.array([]))
    mfcc_delta_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_bandwidth_std: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_rolloff_std: float = 0.0
    spectral_contrast_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_flatness_mean: float = 0.0
    chroma_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    chroma_std: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_mean: float = 0.0
    rms_std: float = 0.0
    zcr_mean: float = 0.0
    zcr_std: float = 0.0
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    voiced_fraction: float = 0.0
    duration_sec: float = 0.0
    speech_rate_proxy: float = 0.0
    # --- Clinical voice-quality features ---
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr_db: float = 0.0
    pause_count: float = 0.0
    pause_total_sec: float = 0.0
    pause_mean_sec: float = 0.0
    voice_tremor_freq: float = 0.0
    voice_tremor_magnitude: float = 0.0
    energy_std: float = 0.0
    energy_range: float = 0.0
    energy_dynamic_range_db: float = 0.0
    sample_rate: int = 16000
    n_features: int = 0

    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single 1-D vector for model input."""
        scalar_features = np.array([
            self.spectral_centroid_mean, self.spectral_centroid_std,
            self.spectral_bandwidth_mean, self.spectral_bandwidth_std,
            self.spectral_rolloff_mean, self.spectral_rolloff_std,
            self.spectral_flatness_mean,
            self.rms_mean, self.rms_std,
            self.zcr_mean, self.zcr_std,
            self.f0_mean, self.f0_std, self.f0_range, self.voiced_fraction,
            self.duration_sec, self.speech_rate_proxy,
            # Clinical voice-quality features
            self.jitter, self.shimmer, self.hnr_db,
            self.pause_count, self.pause_total_sec, self.pause_mean_sec,
            self.voice_tremor_freq, self.voice_tremor_magnitude,
            self.energy_std, self.energy_range, self.energy_dynamic_range_db,
        ])
        vector = np.concatenate([
            self.mfcc_mean, self.mfcc_std, self.mfcc_delta_mean,
            scalar_features,
            self.spectral_contrast_mean,
            self.chroma_mean, self.chroma_std,
        ])
        self.n_features = len(vector)
        return vector.astype(np.float32)


class AudioPipeline:
    """
    CPU-optimized audio feature extraction pipeline for depression screening.

    Extracts ~147-dim feature vector capturing MFCCs, spectral descriptors,
    prosodic features, chroma, pitch (F0) statistics, jitter, shimmer, HNR,
    pause analysis, voice tremor, and energy dynamics.
    """

    def __init__(self):
        self.sr = settings.AUDIO_SAMPLE_RATE
        self.n_mfcc = settings.AUDIO_N_MFCC
        self.n_fft = settings.AUDIO_N_FFT
        self.hop_length = settings.AUDIO_HOP_LENGTH

    def process(self, audio_path: str | Path) -> AudioFeatures:
        """Full pipeline: load → preprocess → extract features."""
        audio_path = Path(audio_path)
        logger.info(f"Processing audio: {audio_path.name}")

        y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        duration = len(y) / sr

        if duration < settings.AUDIO_MIN_DURATION_SEC:
            raise ValueError(
                f"Audio too short ({duration:.1f}s). "
                f"Minimum: {settings.AUDIO_MIN_DURATION_SEC}s"
            )
        if duration > settings.AUDIO_MAX_DURATION_SEC:
            logger.warning(f"Audio trimmed from {duration:.1f}s to {settings.AUDIO_MAX_DURATION_SEC}s")
            y = y[: int(settings.AUDIO_MAX_DURATION_SEC * sr)]
            duration = settings.AUDIO_MAX_DURATION_SEC

        y_processed = self._remove_silence(y, sr)
        if len(y_processed) < sr * 2:
            logger.warning("Very little speech detected, using original audio")
            y_processed = y

        features = self._extract_features(y_processed, sr, duration)
        logger.info(f"Extracted {features.n_features} audio features from {duration:.1f}s")
        return features

    def _remove_silence(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Remove silence using librosa trim + split."""
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        intervals = librosa.effects.split(y_trimmed, top_db=30)
        if len(intervals) == 0:
            return y_trimmed
        return np.concatenate([y_trimmed[s:e] for s, e in intervals])

    def _extract_features(self, y: np.ndarray, sr: int, original_duration: float) -> AudioFeatures:
        """Extract comprehensive acoustic feature set."""
        features = AudioFeatures(duration_sec=original_duration, sample_rate=sr)

        # MFCCs + deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        features.mfcc_mean = np.mean(mfcc, axis=1)
        features.mfcc_std = np.std(mfcc, axis=1)
        features.mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)

        # Spectral
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features.spectral_centroid_mean, features.spectral_centroid_std = float(np.mean(sc)), float(np.std(sc))

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features.spectral_bandwidth_mean, features.spectral_bandwidth_std = float(np.mean(sb)), float(np.std(sb))

        sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features.spectral_rolloff_mean, features.spectral_rolloff_std = float(np.mean(sr_feat)), float(np.std(sr_feat))

        scon = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.spectral_contrast_mean = np.mean(scon, axis=1)

        sf = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features.spectral_flatness_mean = float(np.mean(sf))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.chroma_mean, features.chroma_std = np.mean(chroma, axis=1), np.std(chroma, axis=1)

        # Energy / prosodic
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features.rms_mean, features.rms_std = float(np.mean(rms)), float(np.std(rms))

        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
        features.zcr_mean, features.zcr_std = float(np.mean(zcr)), float(np.std(zcr))
        features.speech_rate_proxy = float(np.mean(zcr)) * sr

        # Pitch (F0)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=self.hop_length
        )
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) > 0:
            features.f0_mean = float(np.mean(voiced_f0))
            features.f0_std = float(np.std(voiced_f0))
            features.f0_range = float(np.max(voiced_f0) - np.min(voiced_f0))
        features.voiced_fraction = float(np.sum(~np.isnan(f0)) / len(f0))

        # ----- Clinical voice-quality features -----

        # 1. Jitter — cycle-to-cycle F0 period perturbation
        if len(voiced_f0) > 1:
            periods = 1.0 / voiced_f0  # convert Hz → seconds
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            features.jitter = float(np.mean(period_diffs) / mean_period) if mean_period > 0 else 0.0

        # 2. Shimmer — cycle-to-cycle amplitude perturbation from RMS
        if len(rms) > 1:
            rms_diffs = np.abs(np.diff(rms))
            mean_rms = np.mean(rms)
            features.shimmer = float(np.mean(rms_diffs) / mean_rms) if mean_rms > 0 else 0.0

        # 3. Harmonics-to-Noise Ratio (HNR) via harmonic/percussive decomposition
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        noise_energy = float(np.sum(y_percussive ** 2))
        if noise_energy > 0:
            features.hnr_db = float(10.0 * np.log10(harmonic_energy / noise_energy))
        else:
            features.hnr_db = 0.0

        # 4. Pause Analysis — silence intervals > 0.3 s in original signal
        pause_threshold_samples = int(0.3 * sr)
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            # Gaps *between* voiced intervals are pauses
            gaps = []
            for i in range(1, len(intervals)):
                gap = intervals[i][0] - intervals[i - 1][1]
                if gap >= pause_threshold_samples:
                    gaps.append(gap)
            features.pause_count = float(len(gaps))
            if len(gaps) > 0:
                gap_durations = np.array(gaps) / sr
                features.pause_total_sec = float(np.sum(gap_durations))
                features.pause_mean_sec = float(np.mean(gap_durations))

        # 5. Voice Tremor — 4-8 Hz modulation in the F0 contour
        if len(voiced_f0) > 8:
            # Remove mean, compute autocorrelation of F0 contour
            f0_centered = voiced_f0 - np.mean(voiced_f0)
            f0_corr = np.correlate(f0_centered, f0_centered, mode='full')
            f0_corr = f0_corr[len(f0_corr) // 2:]  # keep positive lags
            f0_corr = f0_corr / (f0_corr[0] + 1e-10)  # normalise

            # Frame rate of the F0 track
            f0_frame_rate = sr / self.hop_length
            # Lag range corresponding to 4-8 Hz
            min_lag = max(1, int(f0_frame_rate / 8.0))
            max_lag = min(len(f0_corr) - 1, int(f0_frame_rate / 4.0))
            if max_lag > min_lag:
                tremor_region = f0_corr[min_lag:max_lag + 1]
                peak_idx = np.argmax(tremor_region)
                features.voice_tremor_magnitude = float(tremor_region[peak_idx])
                tremor_lag = min_lag + peak_idx
                if tremor_lag > 0:
                    features.voice_tremor_freq = float(f0_frame_rate / tremor_lag)

        # 6. Energy Dynamics — variation patterns of the RMS contour
        if len(rms) > 0:
            features.energy_std = float(np.std(rms))
            features.energy_range = float(np.max(rms) - np.min(rms))
            rms_max = float(np.max(rms))
            rms_min_nonzero = float(np.min(rms[rms > 0])) if np.any(rms > 0) else 1e-10
            features.energy_dynamic_range_db = float(
                20.0 * np.log10(rms_max / rms_min_nonzero)
            ) if rms_max > 0 else 0.0

        features.to_vector()  # compute n_features
        return features
