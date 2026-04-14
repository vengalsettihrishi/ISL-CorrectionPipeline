"""
landmark_loader.py -- Load iSign pre-extracted pose files or extract from video.

The iSign dataset (ACL 2024) provides pre-extracted MediaPipe landmarks as .npy
files alongside the original videos. This module:

1. Loads pre-extracted .npy pose files (preferred, fast).
2. Falls back to MediaPipe Hands + Pose extraction from video if a pose
   file is unavailable.
3. Handles missing landmarks by zero-filling (when a hand or pose is
   not detected in a frame).

Input:
    - video_id (str): Unique identifier for a sample in the iSign dataset.
    - config (Config): Pipeline configuration.

Output:
    - np.ndarray of shape (T, 225) where T = number of frames,
      225 = left_hand(63) + right_hand(63) + pose(99).

Usage:
    python -m data.landmark_loader
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_landmarks(
    video_id: str,
    config: Config,
) -> Optional[np.ndarray]:
    """
    Load landmark sequence for a given video ID.

    Tries pre-extracted .npy first; falls back to MediaPipe extraction
    from the raw video file.

    Args:
        video_id: Unique identifier (e.g. "abc123") -- used to locate
                  both the .npy and video files.
        config:   Pipeline configuration with paths and MP settings.

    Returns:
        np.ndarray of shape (T, 225), dtype float32, or None on failure.
        T is the natural frame count of the video (variable length).
    """
    # --- Attempt 1: pre-extracted pose file ---
    pose_path = Path(config.pose_dir) / f"{video_id}.npy"
    if pose_path.exists():
        landmarks = _load_pose_file(pose_path, config)
        if landmarks is not None:
            return landmarks
        logger.warning(
            f"Pose file for {video_id} exists but could not be loaded; "
            "falling back to video extraction."
        )

    # --- Attempt 2: extract from video ---
    video_path = _find_video_file(video_id, config)
    if video_path is not None:
        return extract_landmarks_from_video(str(video_path), config)

    logger.error(f"No pose file or video found for video_id={video_id}")
    return None


def load_landmarks_from_npy(
    npy_path: str,
    config: Config,
) -> Optional[np.ndarray]:
    """
    Load landmarks directly from a .npy file path.

    Args:
        npy_path: Absolute or relative path to a .npy file.
        config:   Pipeline configuration.

    Returns:
        np.ndarray of shape (T, 225), dtype float32, or None on failure.
    """
    return _load_pose_file(Path(npy_path), config)


# ---------------------------------------------------------------------------
# Internal: load from pre-extracted .npy
# ---------------------------------------------------------------------------

def _load_pose_file(
    path: Path,
    config: Config,
) -> Optional[np.ndarray]:
    """
    Load and validate a pre-extracted .npy pose file.

    The iSign dataset stores landmarks in various layouts. This function
    handles the common case where the file is already (T, 225) or can
    be reshaped to it.

    Args:
        path:   Path to the .npy file.
        config: Pipeline configuration.

    Returns:
        np.ndarray (T, 225) float32, or None if the file is invalid.
    """
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

    data = np.array(data, dtype=np.float32)

    # Handle various shapes from the iSign dataset
    if data.ndim == 1:
        # Flat array -- try to reshape
        if data.size % config.landmark_dim == 0:
            n_frames = data.size // config.landmark_dim
            data = data.reshape(n_frames, config.landmark_dim)
        else:
            logger.warning(
                f"Cannot reshape flat array of size {data.size} "
                f"into (T, {config.landmark_dim})"
            )
            return None

    elif data.ndim == 2:
        if data.shape[1] == config.landmark_dim:
            pass  # Already correct shape
        elif data.shape[0] == config.landmark_dim:
            data = data.T  # Transpose if landmarks are on axis 0
        else:
            # Try to extract the subset we need
            if data.shape[1] > config.landmark_dim:
                logger.info(
                    f"Truncating features from {data.shape[1]} "
                    f"to {config.landmark_dim}"
                )
                data = data[:, :config.landmark_dim]
            else:
                # Pad with zeros if fewer features
                logger.info(
                    f"Padding features from {data.shape[1]} "
                    f"to {config.landmark_dim}"
                )
                padded = np.zeros(
                    (data.shape[0], config.landmark_dim), dtype=np.float32
                )
                padded[:, :data.shape[1]] = data
                data = padded

    elif data.ndim == 3:
        # Possibly (T, num_landmarks, 3) -- flatten last two dims
        if data.shape[1] * data.shape[2] == config.landmark_dim:
            data = data.reshape(data.shape[0], config.landmark_dim)
        else:
            logger.warning(f"Unexpected 3D shape: {data.shape}")
            return None
    else:
        logger.warning(f"Unexpected ndim={data.ndim} for {path}")
        return None

    # Replace NaNs with zeros (missing landmarks)
    data = np.nan_to_num(data, nan=0.0)

    if data.shape[0] == 0:
        logger.warning(f"Empty sequence in {path}")
        return None

    return data


# ---------------------------------------------------------------------------
# Internal: find video file
# ---------------------------------------------------------------------------

def _find_video_file(
    video_id: str,
    config: Config,
) -> Optional[Path]:
    """
    Locate the video file for a given video_id in the video directory.

    Searches for common video extensions.

    Args:
        video_id: Unique video identifier.
        config:   Pipeline configuration.

    Returns:
        Path to the video file, or None if not found.
    """
    video_dir = Path(config.video_dir)
    if not video_dir.exists():
        return None

    extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    for ext in extensions:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate

    # Also search subdirectories (iSign may organize by splits)
    for ext in extensions:
        matches = list(video_dir.rglob(f"{video_id}{ext}"))
        if matches:
            return matches[0]

    return None


# ---------------------------------------------------------------------------
# Fallback: MediaPipe extraction from video
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(
    video_path: str,
    config: Config,
) -> Optional[np.ndarray]:
    """
    Extract hand + pose landmarks from every frame of a video using MediaPipe.

    Layout per frame: [left_hand(63) | right_hand(63) | pose(99)] = 225.
    Missing landmarks (hand not detected, etc.) are zero-filled.

    Args:
        video_path: Path to the video file.
        config:     Pipeline configuration with MediaPipe thresholds.

    Returns:
        np.ndarray of shape (T, 225), dtype float32, where T is the
        number of frames in the video. Returns None if the video cannot
        be opened or no frames are extracted.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.error(
            "mediapipe is not installed. Install with: pip install mediapipe"
        )
        return None

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None

    all_frames: List[np.ndarray] = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.max_num_hands,
        min_detection_confidence=config.min_hand_detection_confidence,
        min_tracking_confidence=config.min_hand_tracking_confidence,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=config.min_pose_detection_confidence,
        min_tracking_confidence=config.min_pose_tracking_confidence,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)

            frame_vector = _build_frame_vector(
                hand_results, pose_results, config
            )
            all_frames.append(frame_vector)

    cap.release()

    if len(all_frames) == 0:
        logger.warning(f"No frames extracted from: {video_path}")
        return None

    return np.array(all_frames, dtype=np.float32)


def _build_frame_vector(
    hand_results,
    pose_results,
    config: Config,
) -> np.ndarray:
    """
    Build a single 225-dim feature vector for one frame.

    Layout: [left_hand(63) | right_hand(63) | pose(99)].

    Missing landmarks are zero-filled. This is important -- the model
    learns that zeros mean "not detected", which is a valid signal.

    Args:
        hand_results:  MediaPipe Hands detection result.
        pose_results:  MediaPipe Pose detection result.
        config:        Pipeline configuration.

    Returns:
        np.ndarray of shape (225,), dtype float32.
    """
    n_hand = config.hand_landmarks * config.coords_per_landmark  # 63
    n_pose = config.pose_landmarks * config.coords_per_landmark  # 99

    left_hand = np.zeros(n_hand, dtype=np.float32)
    right_hand = np.zeros(n_hand, dtype=np.float32)
    pose_vec = np.zeros(n_pose, dtype=np.float32)

    # --- Hands ---
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
        ):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords_arr = np.array(coords, dtype=np.float32)

            label = handedness.classification[0].label
            if label == "Left":
                left_hand = coords_arr
            else:
                right_hand = coords_arr

    # --- Pose ---
    if pose_results.pose_landmarks:
        coords = []
        for lm in pose_results.pose_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        pose_vec = np.array(coords, dtype=np.float32)

    return np.concatenate([left_hand, right_hand, pose_vec])


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print("landmark_loader.py -- verification")
    print("=" * 60)

    # Create a synthetic landmark sequence for testing
    T = 50
    synthetic = np.random.randn(T, cfg.landmark_dim).astype(np.float32)
    print(f"  Synthetic sequence shape: {synthetic.shape}")
    assert synthetic.shape == (T, 225), f"Expected ({T}, 225), got {synthetic.shape}"

    # Test _load_pose_file with a temporary file
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp_path = tmp.name
    tmp.close()
    np.save(tmp_path, synthetic)
    loaded = _load_pose_file(Path(tmp_path), cfg)
    os.unlink(tmp_path)

    assert loaded is not None, "Failed to load synthetic .npy"
    assert loaded.shape == (T, 225), f"Expected ({T}, 225), got {loaded.shape}"
    print(f"  Loaded from .npy: {loaded.shape}")
    print(f"  NaN count: {np.isnan(loaded).sum()}")
    print("=" * 60)
    print("[PASS] landmark_loader.py OK")
