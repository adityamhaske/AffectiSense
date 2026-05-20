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
    # Iris landmarks (available when refine_landmarks=True)
    "left_iris_center": 468, "left_iris_right": 469,
    "left_iris_top": 470, "left_iris_left": 471,
    "right_iris_center": 473, "right_iris_right": 474,
    "right_iris_top": 475, "right_iris_left": 476,
    # Additional eye corner landmarks for gaze reference frame
    "left_eye_inner": 133, "left_eye_outer": 33,
    "right_eye_inner": 362, "right_eye_outer": 263,
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

    # --- NEW: Gaze direction features ---
    avg_gaze_horizontal: float = 0.0
    avg_gaze_vertical: float = 0.0
    gaze_horizontal_std: float = 0.0
    gaze_vertical_std: float = 0.0
    gaze_stability: float = 0.0           # combined std of gaze angles
    gaze_avoidance_ratio: float = 0.0     # fraction of frames with gaze > threshold from center

    # --- NEW: Facial asymmetry features ---
    avg_eye_asymmetry: float = 0.0
    avg_brow_asymmetry: float = 0.0
    avg_mouth_asymmetry: float = 0.0
    overall_facial_asymmetry: float = 0.0

    # --- NEW: Smile intensity features ---
    avg_smile_intensity: float = 0.0
    smile_intensity_std: float = 0.0
    smile_frequency: float = 0.0          # smiles per minute
    avg_smile_duration_frames: float = 0.0

    # --- NEW: Micro-expression features ---
    micro_expression_count: float = 0.0
    micro_expression_rate: float = 0.0    # events per minute

    # --- NEW: Head pose stability features ---
    head_pitch_std: float = 0.0
    head_yaw_std: float = 0.0
    head_roll_std: float = 0.0
    head_pitch_range: float = 0.0
    head_yaw_range: float = 0.0
    head_roll_range: float = 0.0

    # Metadata
    n_frames_processed: int = 0
    n_faces_detected: int = 0
    face_detection_rate: float = 0.0
    duration_sec: float = 0.0
    n_features: int = 0

    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single 1-D vector for model input."""
        vector = np.array([
            # Original 16 features
            self.blink_rate, self.avg_eye_openness, self.eye_openness_std,
            self.avg_brow_height, self.brow_height_std,
            self.avg_mouth_openness, self.mouth_openness_std,
            self.avg_mouth_width, self.mouth_width_std,
            self.avg_head_pitch, self.avg_head_yaw, self.head_movement_magnitude,
            self.facial_expressiveness, self.expression_variability,
            self.au_change_rate, self.face_detection_rate,
            # Gaze direction (6)
            self.avg_gaze_horizontal, self.avg_gaze_vertical,
            self.gaze_horizontal_std, self.gaze_vertical_std,
            self.gaze_stability, self.gaze_avoidance_ratio,
            # Facial asymmetry (4)
            self.avg_eye_asymmetry, self.avg_brow_asymmetry,
            self.avg_mouth_asymmetry, self.overall_facial_asymmetry,
            # Smile intensity (4)
            self.avg_smile_intensity, self.smile_intensity_std,
            self.smile_frequency, self.avg_smile_duration_frames,
            # Micro-expression (2)
            self.micro_expression_count, self.micro_expression_rate,
            # Head pose stability (6)
            self.head_pitch_std, self.head_yaw_std, self.head_roll_std,
            self.head_pitch_range, self.head_yaw_range, self.head_roll_range,
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
        """Compute per-frame facial measurements from 478 landmarks (468 + 10 iris)."""
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

        # Head roll (tilt angle of the face-horizontal vector in the XY plane)
        head_roll = float(np.arctan2(face_horizontal[1], face_horizontal[0]))

        # --- NEW: Gaze direction estimation using iris landmarks ---
        # Compute gaze angle for each eye as iris displacement relative to eye center
        left_eye_inner = landmarks[ix["left_eye_inner"]]
        left_eye_outer = landmarks[ix["left_eye_outer"]]
        left_eye_center = (left_eye_inner + left_eye_outer) / 2.0
        left_iris = landmarks[ix["left_iris_center"]]
        left_eye_width = float(np.linalg.norm(left_eye_inner - left_eye_outer))

        right_eye_inner = landmarks[ix["right_eye_inner"]]
        right_eye_outer = landmarks[ix["right_eye_outer"]]
        right_eye_center = (right_eye_inner + right_eye_outer) / 2.0
        right_iris = landmarks[ix["right_iris_center"]]
        right_eye_width = float(np.linalg.norm(right_eye_inner - right_eye_outer))

        # Normalise iris offset by eye width to get scale-invariant gaze angle proxy
        eye_width_avg = max((left_eye_width + right_eye_width) / 2.0, 1e-6)
        left_gaze_h = (left_iris[0] - left_eye_center[0]) / max(left_eye_width, 1e-6)
        left_gaze_v = (left_iris[1] - left_eye_center[1]) / max(left_eye_width, 1e-6)
        right_gaze_h = (right_iris[0] - right_eye_center[0]) / max(right_eye_width, 1e-6)
        right_gaze_v = (right_iris[1] - right_eye_center[1]) / max(right_eye_width, 1e-6)

        gaze_horizontal = (left_gaze_h + right_gaze_h) / 2.0
        gaze_vertical = (left_gaze_v + right_gaze_v) / 2.0

        # --- NEW: Facial asymmetry ---
        eye_asymmetry = abs(left_eye_open - right_eye_open) / max(eye_openness, 1e-6)
        brow_asymmetry = abs(left_brow - right_brow) / max(brow_height, 1e-6)
        left_mouth_corner_y = landmarks[ix["left_mouth_corner"]][1]
        right_mouth_corner_y = landmarks[ix["right_mouth_corner"]][1]
        mouth_center_y = (left_mouth_corner_y + right_mouth_corner_y) / 2.0
        mouth_asymmetry = abs(left_mouth_corner_y - right_mouth_corner_y) / max(abs(mouth_center_y), 1e-6)

        # --- NEW: Smile intensity ---
        # Smile score: mouth corners rise (lower y = higher in image) relative to
        # mouth center (upper_lip/lower_lip midpoint), combined with mouth width.
        mouth_center = (landmarks[ix["upper_lip"]] + landmarks[ix["lower_lip"]]) / 2.0
        left_corner = landmarks[ix["left_mouth_corner"]]
        right_corner = landmarks[ix["right_mouth_corner"]]
        # Corner displacement: how far each corner is from mouth center
        corner_disp = (float(np.linalg.norm(left_corner - mouth_center)) +
                       float(np.linalg.norm(right_corner - mouth_center))) / 2.0
        # Cheek raise proxy (AU6): cheek-to-eye distance decrease
        left_cheek_eye_dist = dist(ix["left_cheek"], ix["left_eye_bottom"])
        right_cheek_eye_dist = dist(ix["right_cheek"], ix["right_eye_bottom"])
        cheek_raise = 1.0 / max((left_cheek_eye_dist + right_cheek_eye_dist) / 2.0, 1e-6)
        # Combine: higher corner displacement + higher cheek raise → stronger smile
        smile_intensity = corner_disp * cheek_raise

        return {
            "eye_openness": eye_openness,
            "left_eye_open": left_eye_open,
            "right_eye_open": right_eye_open,
            "brow_height": brow_height,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "mouth_openness": mouth_openness,
            "mouth_width": mouth_width,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "head_roll": head_roll,
            "gaze_horizontal": float(gaze_horizontal),
            "gaze_vertical": float(gaze_vertical),
            "eye_asymmetry": float(eye_asymmetry),
            "brow_asymmetry": float(brow_asymmetry),
            "mouth_asymmetry": float(mouth_asymmetry),
            "smile_intensity": float(smile_intensity),
        }

    def _compute_features(self, frame_data: list[dict], duration: float) -> VideoFeatures:
        """Aggregate per-frame measurements into temporal statistics."""
        features = VideoFeatures()
        features.n_frames_processed = len(frame_data)
        features.duration_sec = duration
        duration_min = max(duration / 60.0, 0.01)

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
        features.blink_rate = float(blinks) / duration_min

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

        # ===================================================================
        # NEW FEATURE AGGREGATIONS
        # ===================================================================

        # --- 1. Gaze direction features ---
        gaze_h = np.array([f["gaze_horizontal"] for f in frame_data])
        gaze_v = np.array([f["gaze_vertical"] for f in frame_data])
        features.avg_gaze_horizontal = float(np.mean(gaze_h))
        features.avg_gaze_vertical = float(np.mean(gaze_v))
        features.gaze_horizontal_std = float(np.std(gaze_h))
        features.gaze_vertical_std = float(np.std(gaze_v))
        features.gaze_stability = float(np.sqrt(np.std(gaze_h)**2 + np.std(gaze_v)**2))
        # Gaze avoidance: fraction of frames where gaze deviates > 0.3 (normalised) from center
        gaze_deviation = np.sqrt(gaze_h**2 + gaze_v**2)
        features.gaze_avoidance_ratio = float(np.mean(gaze_deviation > 0.3))

        # --- 2. Facial asymmetry features ---
        eye_asym = np.array([f["eye_asymmetry"] for f in frame_data])
        brow_asym = np.array([f["brow_asymmetry"] for f in frame_data])
        mouth_asym = np.array([f["mouth_asymmetry"] for f in frame_data])
        features.avg_eye_asymmetry = float(np.mean(eye_asym))
        features.avg_brow_asymmetry = float(np.mean(brow_asym))
        features.avg_mouth_asymmetry = float(np.mean(mouth_asym))
        features.overall_facial_asymmetry = float(np.mean([features.avg_eye_asymmetry,
                                                           features.avg_brow_asymmetry,
                                                           features.avg_mouth_asymmetry]))

        # --- 3. Smile intensity features ---
        smile = np.array([f["smile_intensity"] for f in frame_data])
        features.avg_smile_intensity = float(np.mean(smile))
        features.smile_intensity_std = float(np.std(smile))
        # Detect smile events: smile exceeds mean + 0.5*std (adaptive threshold)
        smile_threshold = float(np.mean(smile) + 0.5 * np.std(smile))
        smile_binary = (smile > smile_threshold).astype(int)
        smile_onsets = np.sum(np.diff(smile_binary) == 1)
        features.smile_frequency = float(smile_onsets) / duration_min
        # Average smile duration (in frames)
        if smile_onsets > 0:
            smile_frames = np.sum(smile_binary)
            features.avg_smile_duration_frames = float(smile_frames) / float(max(smile_onsets, 1))
        else:
            features.avg_smile_duration_frames = 0.0

        # --- 4. Micro-expression detection ---
        # A micro-expression is a rapid (<0.5s) transient spike in any AU proxy.
        # We detect frames where the frame-to-frame AU change exceeds 2× the
        # mean change rate AND the spike reverts within a short temporal window.
        n_frames = len(frame_data)
        if n_frames > 2:
            # Effective FPS of the sampled stream
            effective_fps = max(float(n_frames) / max(duration, 0.01), 1.0)
            # Maximum spike duration in frames for <0.5 s events
            max_spike_frames = max(int(0.5 * effective_fps), 1)
            # Compute frame-to-frame absolute changes for each AU signal
            au_deltas = np.abs(np.diff(all_signals, axis=1))  # shape: (4, n_frames-1)
            mean_delta = np.mean(au_deltas, axis=1, keepdims=True)  # per signal
            spike_threshold = 2.0 * mean_delta + 1e-8
            # A frame is a "spike" if any AU channel exceeds its threshold
            spike_any = np.any(au_deltas > spike_threshold, axis=0)  # shape: (n_frames-1,)
            # Count isolated spike bursts of length <= max_spike_frames
            micro_count = 0
            i = 0
            spike_indices = np.where(spike_any)[0]
            while i < len(spike_indices):
                # Find the end of this contiguous spike run
                j = i
                while j + 1 < len(spike_indices) and spike_indices[j + 1] - spike_indices[j] <= 1:
                    j += 1
                run_length = spike_indices[j] - spike_indices[i] + 1
                if run_length <= max_spike_frames:
                    micro_count += 1
                i = j + 1
            features.micro_expression_count = float(micro_count)
            features.micro_expression_rate = float(micro_count) / duration_min
        else:
            features.micro_expression_count = 0.0
            features.micro_expression_rate = 0.0

        # --- 5. Head pose stability features ---
        head_roll = np.array([f["head_roll"] for f in frame_data])
        features.head_pitch_std = float(np.std(head_pitch))
        features.head_yaw_std = float(np.std(head_yaw))
        features.head_roll_std = float(np.std(head_roll))
        features.head_pitch_range = float(np.ptp(head_pitch))
        features.head_yaw_range = float(np.ptp(head_yaw))
        features.head_roll_range = float(np.ptp(head_roll))

        # ===================================================================

        features.n_faces_detected = len(frame_data)
        features.face_detection_rate = 1.0  # all frame_data entries have faces

        features.to_vector()
        return features
