"""
convert_parquet_to_npy.py — Convert landmark_files_all/ parquet dataset to pipeline-ready .npy files.

WHAT THIS DOES:
  1. Reads each .parquet file (one per video, long/tidy format)
  2. Drops face landmarks (468 pts) — pipeline does not use them
  3. Fills NaN coordinates with 0.0  (undetected = zero convention)
  4. Zero-pads any frame whose hand/pose was fully absent
  5. Builds per-frame flat vector: [left_hand(63) | right_hand(63) | pose(99)] = 225
  6. Resamples sequence to target_seq_length frames (default: 30)
  7. Applies per-frame z-score normalization (preserving zeros)
  8. Saves as .npy with shape (target_seq_length, 225)
  9. Writes label_map.json

USAGE:
  python convert_parquet_to_npy.py
  python convert_parquet_to_npy.py --input_dir ./landmark_files_all --output_dir ./processed_landmarks --seq_length 30

OUTPUT:
  processed_landmarks/
    label_map.json
    Hello/
      0029.npy   ← shape (30, 225), dtype float32
      0030.npy
      ...
    Baby/
      3769.npy
      ...
    ...
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the existing resample + normalize functions from this project
from landmark_extractor import resample_sequence, normalize_landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core conversion: one parquet file → one (T, 225) numpy array
# ---------------------------------------------------------------------------

EXPECTED_HAND_LM   = 21   # MediaPipe Hands: 21 landmarks per hand
EXPECTED_POSE_LM   = 33   # MediaPipe Pose: 33 landmarks
COORDS             = 3    # x, y, z
LEFT_DIM  = EXPECTED_HAND_LM * COORDS   # 63
RIGHT_DIM = EXPECTED_HAND_LM * COORDS   # 63
POSE_DIM  = EXPECTED_POSE_LM * COORDS   # 99
TOTAL_DIM = LEFT_DIM + RIGHT_DIM + POSE_DIM  # 225


def parquet_to_array(parquet_path: str, target_seq_length: int = 30) -> np.ndarray:
    """
    Convert one .parquet landmark file to a (target_seq_length, 225) float32 array.

    Raises:
        ValueError  if the file has no usable frames after filtering.
    """
    df = pd.read_parquet(parquet_path)

    # ── Step 1: Drop face ──────────────────────────────────────────────────
    # The dataset contains 'face' rows (468 FaceMesh landmarks per frame).
    # Pipeline expects only left_hand + right_hand + pose → total 225 features.
    df = df[df["type"].isin(["left_hand", "right_hand", "pose"])].copy()

    # ── Step 2: Replace NaN coordinates with 0 ────────────────────────────
    # NaN occurs when MediaPipe detects the landmark structure but cannot
    # confidently localize it (e.g. occluded). Convention: undetected = 0.
    df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)

    frames = sorted(df["frame"].unique())
    if len(frames) == 0:
        raise ValueError(f"No usable frames in {parquet_path}")

    frame_vectors = []

    for frame_idx in frames:
        fdf = df[df["frame"] == frame_idx]

        # ── Step 3: Build left-hand vector (63 floats) ────────────────────
        lh = fdf[fdf["type"] == "left_hand"].sort_values("landmark_index")
        if len(lh) == EXPECTED_HAND_LM:
            left_vec = lh[["x", "y", "z"]].values.flatten().astype(np.float32)
        else:
            # Hand not detected in this frame → zero-fill
            left_vec = np.zeros(LEFT_DIM, dtype=np.float32)

        # ── Step 4: Build right-hand vector (63 floats) ───────────────────
        rh = fdf[fdf["type"] == "right_hand"].sort_values("landmark_index")
        if len(rh) == EXPECTED_HAND_LM:
            right_vec = rh[["x", "y", "z"]].values.flatten().astype(np.float32)
        else:
            right_vec = np.zeros(RIGHT_DIM, dtype=np.float32)

        # ── Step 5: Build pose vector (99 floats) ─────────────────────────
        ps = fdf[fdf["type"] == "pose"].sort_values("landmark_index")
        if len(ps) == EXPECTED_POSE_LM:
            pose_vec = ps[["x", "y", "z"]].values.flatten().astype(np.float32)
        else:
            pose_vec = np.zeros(POSE_DIM, dtype=np.float32)

        # ── Step 6: Concatenate in required order: [Left | Right | Pose] ──
        frame_vec = np.concatenate([left_vec, right_vec, pose_vec])  # shape (225,)
        frame_vectors.append(frame_vec)

    sequence = np.array(frame_vectors, dtype=np.float32)  # (T, 225)

    # ── Step 7: Resample to fixed sequence length ──────────────────────────
    sequence = resample_sequence(sequence, target_seq_length)   # → (target_seq_length, 225)

    # ── Step 8: Per-frame normalization (zeros stay zero) ─────────────────
    sequence = normalize_landmarks(sequence)

    assert sequence.shape == (target_seq_length, TOTAL_DIM), (
        f"Unexpected shape {sequence.shape} from {parquet_path}"
    )
    return sequence


# ---------------------------------------------------------------------------
# Batch converter: entire dataset
# ---------------------------------------------------------------------------

def convert_dataset(
    input_dir: str,
    output_dir: str,
    target_seq_length: int = 30,
) -> dict:
    """
    Walk input_dir (class_name/video.parquet), convert every file, write to output_dir.

    Returns label_map {class_name: int}.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if not class_dirs:
        logger.error(f"No class subdirectories found in {input_dir}")
        sys.exit(1)

    label_map = {d.name: idx for idx, d in enumerate(class_dirs)}
    logger.info(f"Found {len(label_map)} classes.")
    logger.info(f"Target sequence length: {target_seq_length} frames | Features per frame: {TOTAL_DIM}")

    total   = 0
    success = 0
    failed  = 0

    for class_dir in class_dirs:
        class_name  = class_dir.name
        class_out   = output_path / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(class_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"  [{class_name}] No .parquet files found — skipping")
            continue

        class_ok = 0
        for pq_file in parquet_files:
            total += 1
            out_file = class_out / (pq_file.stem + ".npy")

            try:
                array = parquet_to_array(str(pq_file), target_seq_length)
                np.save(str(out_file), array)
                class_ok += 1
                success  += 1
            except Exception as exc:
                logger.warning(f"  [FAIL] {pq_file.name}: {exc}")
                failed += 1

        logger.info(f"  [{class_name}] {class_ok}/{len(parquet_files)} converted")

    # ── Save label map ─────────────────────────────────────────────────────
    label_map_path = output_path / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    logger.info("─" * 60)
    logger.info(f"DONE.  {success}/{total} files converted successfully.")
    logger.info(f"FAILED: {failed}")
    logger.info(f"Output: {output_path.resolve()}")
    logger.info(f"label_map.json saved: {label_map_path}")

    return label_map


# ---------------------------------------------------------------------------
# Quick sanity check on a single file
# ---------------------------------------------------------------------------

def verify_one(npy_path: str, target_seq_length: int = 30) -> None:
    """Load a converted .npy and assert it matches pipeline expectations."""
    arr = np.load(npy_path)
    assert arr.shape == (target_seq_length, TOTAL_DIM), \
        f"Shape mismatch: {arr.shape} — expected ({target_seq_length}, {TOTAL_DIM})"
    assert arr.dtype == np.float32, f"Dtype mismatch: {arr.dtype}"
    assert not np.isnan(arr).any(), "NaN values found!"

    left  = arr[:, :LEFT_DIM]
    right = arr[:, LEFT_DIM:LEFT_DIM + RIGHT_DIM]
    pose  = arr[:, LEFT_DIM + RIGHT_DIM:]

    print(f"\n✅  {Path(npy_path).name} — PASSED")
    print(f"   Shape : {arr.shape}  dtype={arr.dtype}")
    print(f"   Left  hand slice [{0}:{LEFT_DIM}]  — non-zero frames: {(left.any(axis=1)).sum()}/{target_seq_length}")
    print(f"   Right hand slice [{LEFT_DIM}:{LEFT_DIM+RIGHT_DIM}] — non-zero frames: {(right.any(axis=1)).sum()}/{target_seq_length}")
    print(f"   Pose       slice [{LEFT_DIM+RIGHT_DIM}:225]        — non-zero frames: {(pose.any(axis=1)).sum()}/{target_seq_length}")
    print(f"   Value range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   Motion (mean |v_t|): {np.abs(np.diff(arr, axis=0)).mean():.6f}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ISL .parquet landmark dataset to pipeline-ready .npy files"
    )
    parser.add_argument(
        "--input_dir",  default="./landmark_files_all",
        help="Root dir containing class_name/video.parquet (default: ./landmark_files_all)"
    )
    parser.add_argument(
        "--output_dir", default="./processed_landmarks",
        help="Root dir for output .npy files (default: ./processed_landmarks)"
    )
    parser.add_argument(
        "--seq_length", type=int, default=30,
        help="Target sequence length in frames (default: 30)"
    )
    parser.add_argument(
        "--test_only", action="store_true",
        help="Dry-run: convert a single file and verify shape, then exit"
    )
    args = parser.parse_args()

    if args.test_only:
        # Pick the very first parquet file found and test conversion
        input_path = Path(args.input_dir)
        sample = next(input_path.rglob("*.parquet"), None)
        if sample is None:
            logger.error("No .parquet files found for test.")
            sys.exit(1)

        logger.info(f"TEST MODE — converting single file: {sample}")
        arr = parquet_to_array(str(sample), args.seq_length)
        test_out = Path(args.output_dir).parent / "_test_sample.npy"
        test_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(test_out), arr)
        verify_one(str(test_out), args.seq_length)
        test_out.unlink(missing_ok=True)  # clean up
        logger.info("Test passed. Run without --test_only to convert the full dataset.")
        return

    label_map = convert_dataset(args.input_dir, args.output_dir, args.seq_length)

    # Spot-check: verify the first converted .npy
    output_path = Path(args.output_dir)
    first_npy = next(output_path.rglob("*.npy"), None)
    if first_npy and first_npy.name != "label_map.json":
        verify_one(str(first_npy), args.seq_length)

    print(f"\nNext step: update config.py → num_classes = {len(label_map)}")
    print("Then run:  python train.py")


if __name__ == "__main__":
    main()
