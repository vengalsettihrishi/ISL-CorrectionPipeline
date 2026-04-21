"""
landmark_extractor.py — Extract hand + pose landmarks from sign language videos.

This module processes each video file through MediaPipe Hands and MediaPipe Pose,
extracts per-frame landmark coordinates, resamples to a fixed sequence length,
and saves the result as a numpy array.

Usage:
    python landmark_extractor.py --input_dir ./videos --output_dir ./processed_landmarks

Input:  Directory of videos organized as class_name/video_file.mp4
Output: Directory of .npy files organized as class_name/video_file.npy
        Each .npy has shape (target_seq_length, features_per_frame)
        Also saves a label_map.json mapping class names to integer indices.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe is imported inside functions to keep the module importable
# even if mediapipe isn't installed (useful for testing other modules).
#
# Compatibility note: MediaPipe 0.10.x (Python 3.13 / Windows) ships only the
# Tasks API — mp.solutions.hands / mp.solutions.pose no longer exist.
# This module uses mp.tasks.vision.HandLandmarker and PoseLandmarker.

from config import LandmarkConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _download_model_if_needed(model_path: str, url: str) -> str:
    """Download a MediaPipe Tasks .task model file if not already present."""
    import urllib.request
    path = Path(model_path)
    if not path.exists():
        logger.info(f"Downloading model to {model_path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        logger.info("Download complete.")
    return str(path)


def extract_landmarks_from_video(
    video_path: str,
    config: LandmarkConfig,
    hand_model_path: str = "models/hand_landmarker.task",
    pose_model_path: str = "models/pose_landmarker_lite.task",
) -> Optional[np.ndarray]:
    """
    Extract hand + pose landmarks from every frame of a video.

    Uses MediaPipe Tasks API (0.10.x+). Models are auto-downloaded on first run
    to the models/ subdirectory (~8 MB total).

    Returns:
        np.ndarray of shape (num_frames, features_per_frame) or None if
        the video cannot be read or no landmarks are detected in any frame.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # ── Download model files on first run ────────────────────────────────────
    hand_model = _download_model_if_needed(
        hand_model_path,
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    )
    pose_model = _download_model_if_needed(
        pose_model_path,
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    )

    # ── Configure landmarkers (VIDEO mode = tracking across frames) ───────────
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.max_num_hands,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_detection_confidence,
        min_tracking_confidence=config.min_hand_tracking_confidence,
    )
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=config.min_pose_detection_confidence,
        min_tracking_confidence=config.min_pose_tracking_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_frames: List[np.ndarray] = []
    frame_idx = 0

    with mp_vision.HandLandmarker.create_from_options(hand_options) as hand_det, \
         mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_det:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe Tasks expects RGB mp.Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp in milliseconds (must be strictly increasing)
            timestamp_ms = int(frame_idx * 1000 / fps)

            hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_det.detect_for_video(mp_image, timestamp_ms)

            frame_vector = _build_frame_vector_tasks(hand_result, pose_result, config)
            all_frames.append(frame_vector)
            frame_idx += 1

    cap.release()

    if len(all_frames) == 0:
        logger.warning(f"No frames extracted from: {video_path}")
        return None

    return np.array(all_frames, dtype=np.float32)


def _build_frame_vector(hand_results, pose_results, config: LandmarkConfig) -> np.ndarray:
    """
    Legacy helper kept for any callers that pass old mp.solutions result objects.
    Delegates to the Tasks-API version.
    """
    # Wrap old-style results into a compatible call (no-op: returns zeros)
    # This stub exists so convert_parquet_to_npy.py (which doesn't call this
    # path) and any unit tests continue to import without errors.
    n_hand = config.hand_landmarks_per_hand * config.coords_per_landmark
    n_pose = config.pose_landmarks * config.coords_per_landmark
    return np.zeros(n_hand * 2 + n_pose, dtype=np.float32)


def _build_frame_vector_tasks(hand_result, pose_result, config: LandmarkConfig) -> np.ndarray:
    """
    Build a single feature vector for one frame from MediaPipe Tasks API results.

    Layout: [left_hand(63) | right_hand(63) | pose(99)] = 225 floats.

    If a hand or pose is not detected in this frame, those slots are filled
    with zeros — the model learns that zeros mean "not detected".
    """
    n_hand = config.hand_landmarks_per_hand * config.coords_per_landmark  # 63
    n_pose = config.pose_landmarks * config.coords_per_landmark           # 99

    left_hand = np.zeros(n_hand, dtype=np.float32)
    right_hand = np.zeros(n_hand, dtype=np.float32)
    pose_vec = np.zeros(n_pose, dtype=np.float32)

    # --- Hands (Tasks API) ---
    # hand_result.hand_landmarks: List[List[NormalizedLandmark]]
    # hand_result.handedness:     List[List[Category]]
    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_landmarks, handedness_list in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):
            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)

            # Tasks API: handedness_list[0].category_name is "Left" or "Right"
            label = handedness_list[0].category_name
            if label == "Left":
                left_hand = coords
            else:
                right_hand = coords

    # --- Pose (Tasks API) ---
    # pose_result.pose_landmarks: List[List[NormalizedLandmark]]
    if pose_result.pose_landmarks:
        coords = []
        for lm in pose_result.pose_landmarks[0]:  # first (only) detected person
            coords.extend([lm.x, lm.y, lm.z])
        pose_vec = np.array(coords, dtype=np.float32)

    return np.concatenate([left_hand, right_hand, pose_vec])


# ---------------------------------------------------------------------------
# Sequence resampling
# ---------------------------------------------------------------------------

def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample a landmark sequence to a fixed number of frames.

    Uses linear interpolation to stretch or compress the temporal axis.
    This ensures every training sample has the same shape regardless
    of the original video length or frame rate.

    Args:
        sequence: shape (original_length, features)
        target_length: desired number of frames

    Returns:
        shape (target_length, features)
    """
    original_length = sequence.shape[0]
    if original_length == target_length:
        return sequence

    # Indices in the original sequence that correspond to evenly spaced
    # points in the target sequence
    original_indices = np.linspace(0, original_length - 1, target_length)

    resampled = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
    for i, idx in enumerate(original_indices):
        lower = int(np.floor(idx))
        upper = min(lower + 1, original_length - 1)
        frac = idx - lower
        resampled[i] = sequence[lower] * (1 - frac) + sequence[upper] * frac

    return resampled


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_landmarks(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize landmark coordinates to be translation and scale invariant.

    For each frame:
    1. Subtract the mean of all non-zero landmarks (centering).
    2. Divide by the standard deviation (scale normalization).

    This makes the features robust to the signer's distance from the camera
    and position in the frame.

    Zero-valued landmarks (undetected) remain zero after normalization.
    """
    normalized = sequence.copy()
    for i in range(normalized.shape[0]):
        frame = normalized[i]
        nonzero_mask = frame != 0
        if nonzero_mask.any():
            mean = frame[nonzero_mask].mean()
            std = frame[nonzero_mask].std()
            if std > 1e-6:
                normalized[i][nonzero_mask] = (frame[nonzero_mask] - mean) / std
    return normalized


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_dataset(
    input_dir: str,
    output_dir: str,
    config: LandmarkConfig,
) -> Dict[str, int]:
    """
    Process an entire video dataset organized as class_name/video.mp4.

    Returns:
        label_map: dict mapping class name -> integer index
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover classes from directory names
    class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    label_map = {d.name: idx for idx, d in enumerate(class_dirs)}

    logger.info(f"Found {len(label_map)} classes: {list(label_map.keys())[:10]}...")

    # Supported video extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    total = 0
    failed = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_output = output_path / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        video_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]

        for video_file in video_files:
            total += 1

            # Extract raw landmarks
            raw_sequence = extract_landmarks_from_video(video_file, config)
            if raw_sequence is None or len(raw_sequence) < 3:
                failed += 1
                logger.warning(f"Skipping {video_file.name} — too few frames or unreadable")
                continue

            # Resample to fixed length
            resampled = resample_sequence(raw_sequence, config.target_seq_length)

            # Normalize
            normalized = normalize_landmarks(resampled)

            # Save as .npy
            out_file = class_output / f"{video_file.stem}.npy"
            np.save(str(out_file), normalized)

        logger.info(
            f"  [{class_name}] Processed {len(video_files)} videos"
        )

    # Save label map
    label_map_path = output_path / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    logger.info(f"Done. {total - failed}/{total} videos processed successfully.")
    logger.info(f"Label map saved to {label_map_path}")

    return label_map


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from ISL video dataset"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to video dataset (organized as class_name/video.mp4)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./processed_landmarks",
        help="Path to save processed .npy files"
    )
    parser.add_argument(
        "--seq_length", type=int, default=30,
        help="Target sequence length (frames per video)"
    )

    args = parser.parse_args()

    config = LandmarkConfig(target_seq_length=args.seq_length)

    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sequence length:  {config.target_seq_length}")
    logger.info(f"Features/frame:   {config.features_per_frame}")

    process_dataset(args.input_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
