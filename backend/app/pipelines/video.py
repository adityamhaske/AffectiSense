"""
AffectiSense — Video Feature Extraction Pipeline

Extracts depression-relevant facial biomarkers from video recordings using
MediaPipe FaceMesh. Features include AU dynamics, gaze patterns, blink rate,
and head pose — all privacy-preserving (raw video is never stored).

Pipeline: Load → Frame sampling → Face detection → Landmark extraction → Temporal stats
"""

import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from backend.app.core.config import settings


# Key facial landmark indices for AU-related measurements
LANDMARK_INDICES = {
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "left_eyebrow_inner": 107, "left_eyebrow_outer": 70,
    "right_eyebrow_inner": 336, "right_eyebrow_outer": 300,
    "nose_tip": 1, "chin": 152,
    "left_mouth_corner": 61, "right_mouth_corner": 291,
    "upper_lip": 13, "lower_lip": 14,
    "forehead": 10,
    "left_cheek": 234, "right_cheek": 454,
}


@dataclass
class VideoFeatures:
    """Container for extracted video features."""
    # Eye features (blink rate, openness)
    blink_rate: float = 0.0
    avg_eye_openness: float = 0.0
    eye_openness_std: float = 0.0

    # Eyebrow features (AU4 - brow lowerer proxy)
    avg_brow_height: float = 0.0
    brow_height_std: float = 0.0

    # Mouth features (smile, lip movement)
    avg_mouth_openness: float = 0.0
    mouth_openness_std: float = 0.0
    avg_mouth_width: float = 0.0
    mouth_width_std: float = 0.0

    # Head pose
    avg_head_pitch: float = 0.0
    avg_head_yaw: float = 0.0
    head_movement_magnitude: float = 0.0

    # Overall facial expressiveness
    facial_expressiveness: float = 0.0
    expression_variability: float = 0.0

    # Temporal dynamics
    au_change_rate: float = 0.0

    # Metadata
    n_frames_processed: int = 0
    n_faces_detected: int = 0
    face_detection_rate: float = 0.0
    duration_sec: float = 0.0
    n_features: int = 0

    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single 1-D vector for model input."""
        vector = np.array([
            self.blink_rate, self.avg_eye_openness, self.eye_openness_std,
            self.avg_brow_height, self.brow_height_std,
            self.avg_mouth_openness, self.mouth_openness_std,
            self.avg_mouth_width, self.mouth_width_std,
            self.avg_head_pitch, self.avg_head_yaw, self.head_movement_magnitude,
            self.facial_expressiveness, self.expression_variability,
            self.au_change_rate, self.face_detection_rate,
        ], dtype=np.float32)
        self.n_features = len(vector)
        return vector


class VideoPipeline:
    """
    Privacy-preserving video feature extraction for depression screening.

    Uses MediaPipe FaceMesh to extract 468 facial landmarks per frame,
    then computes AU-proxy features, gaze patterns, and temporal dynamics.
    Raw video frames are never stored — only computed scalar features persist.
    """

    def __init__(self):
        self.target_fps = settings.VIDEO_FPS_TARGET
        self.mp_face_mesh = mp.solutions.face_mesh
        self._blink_threshold = 0.02  # eye aspect ratio threshold

    def process(self, video_path: str | Path) -> VideoFeatures:
        """Full pipeline: load → sample frames → extract landmarks → compute features."""
        video_path = Path(video_path)
        logger.info(f"Processing video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        frame_skip = max(1, int(original_fps / self.target_fps))

        logger.info(f"Video: {duration:.1f}s, {original_fps:.0f}fps, sampling every {frame_skip} frames")

        # Extract landmarks from sampled frames
        frame_data = self._extract_landmarks(cap, frame_skip)
        cap.release()

        if len(frame_data) == 0:
            logger.warning("No faces detected in video, returning zero features")
            features = VideoFeatures(duration_sec=duration)
            features.to_vector()
            return features

        features = self._compute_features(frame_data, duration)
        logger.info(f"Extracted {features.n_features} video features from {features.n_frames_processed} frames")
        return features

    def _extract_landmarks(self, cap: cv2.VideoCapture, frame_skip: int) -> list[dict]:
        """Extract facial landmarks from sampled video frames."""
        frame_data = []
        frame_idx = 0

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        ) as face_mesh:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    lm_array = np.array(
                        [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                    )
                    measurements = self._compute_frame_measurements(lm_array)
                    frame_data.append(measurements)

                frame_idx += 1

        return frame_data

    def _compute_frame_measurements(self, landmarks: np.ndarray) -> dict:
        """Compute per-frame facial measurements from 468 landmarks."""
        def dist(idx1: int, idx2: int) -> float:
            return float(np.linalg.norm(landmarks[idx1] - landmarks[idx2]))

        ix = LANDMARK_INDICES

        # Eye openness (Eye Aspect Ratio proxy)
        left_eye_open = dist(ix["left_eye_top"], ix["left_eye_bottom"])
        right_eye_open = dist(ix["right_eye_top"], ix["right_eye_bottom"])
        eye_openness = (left_eye_open + right_eye_open) / 2.0

        # Brow height (relative to nose bridge)
        left_brow = dist(ix["left_eyebrow_inner"], ix["nose_tip"])
        right_brow = dist(ix["right_eyebrow_inner"], ix["nose_tip"])
        brow_height = (left_brow + right_brow) / 2.0

        # Mouth measurements
        mouth_openness = dist(ix["upper_lip"], ix["lower_lip"])
        mouth_width = dist(ix["left_mouth_corner"], ix["right_mouth_corner"])

        # Head pose proxy (nose-to-chin vector angle)
        nose = landmarks[ix["nose_tip"]]
        chin = landmarks[ix["chin"]]
        forehead = landmarks[ix["forehead"]]
        face_vertical = forehead - chin
        head_pitch = float(np.arctan2(face_vertical[2], face_vertical[1]))

        left_cheek = landmarks[ix["left_cheek"]]
        right_cheek = landmarks[ix["right_cheek"]]
        face_horizontal = right_cheek - left_cheek
        head_yaw = float(np.arctan2(face_horizontal[2], face_horizontal[0]))

        return {
            "eye_openness": eye_openness,
            "brow_height": brow_height,
            "mouth_openness": mouth_openness,
            "mouth_width": mouth_width,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
        }

    def _compute_features(self, frame_data: list[dict], duration: float) -> VideoFeatures:
        """Aggregate per-frame measurements into temporal statistics."""
        features = VideoFeatures()
        features.n_frames_processed = len(frame_data)
        features.duration_sec = duration

        eye_openness = np.array([f["eye_openness"] for f in frame_data])
        brow_height = np.array([f["brow_height"] for f in frame_data])
        mouth_openness = np.array([f["mouth_openness"] for f in frame_data])
        mouth_width = np.array([f["mouth_width"] for f in frame_data])
        head_pitch = np.array([f["head_pitch"] for f in frame_data])
        head_yaw = np.array([f["head_yaw"] for f in frame_data])

        # Eye features
        features.avg_eye_openness = float(np.mean(eye_openness))
        features.eye_openness_std = float(np.std(eye_openness))
        blinks = np.sum(np.diff((eye_openness < self._blink_threshold).astype(int)) == 1)
        features.blink_rate = float(blinks) / max(duration / 60.0, 0.01)

        # Brow features
        features.avg_brow_height = float(np.mean(brow_height))
        features.brow_height_std = float(np.std(brow_height))

        # Mouth features
        features.avg_mouth_openness = float(np.mean(mouth_openness))
        features.mouth_openness_std = float(np.std(mouth_openness))
        features.avg_mouth_width = float(np.mean(mouth_width))
        features.mouth_width_std = float(np.std(mouth_width))

        # Head pose
        features.avg_head_pitch = float(np.mean(head_pitch))
        features.avg_head_yaw = float(np.mean(head_yaw))
        head_movement = np.sqrt(np.diff(head_pitch)**2 + np.diff(head_yaw)**2)
        features.head_movement_magnitude = float(np.mean(head_movement)) if len(head_movement) > 0 else 0.0

        # Expressiveness (total variance across all AU proxies)
        all_signals = np.stack([eye_openness, brow_height, mouth_openness, mouth_width])
        features.facial_expressiveness = float(np.mean(np.std(all_signals, axis=1)))
        features.expression_variability = float(np.std(np.std(all_signals, axis=1)))

        # AU change rate (mean absolute frame-to-frame change)
        deltas = np.abs(np.diff(all_signals, axis=1))
        features.au_change_rate = float(np.mean(deltas))

        features.n_faces_detected = len(frame_data)
        features.face_detection_rate = 1.0  # all frame_data entries have faces

        features.to_vector()
        return features
