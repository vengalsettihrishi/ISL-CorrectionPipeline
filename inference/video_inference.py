"""
inference/video_inference.py -- Offline video/batch inference with JSON output.

Supports:
    - Single video file
    - Directory of videos
    - Dataset split evaluation with reference comparison
    - JSON results output with metrics

Metrics (English-token CTC):
    - Token-level WER (Word Error Rate)
    - Sequence accuracy (exact match)
    - Latency summary
    - Decoder/refinement statistics

Usage:
    python -m inference.video_inference --video path/to/video.mp4
    python -m inference.video_inference --video-dir path/to/videos/
    python -m inference.video_inference --video path/to/video.mp4 --output results.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.utils import ProfileTimer, InferenceConfig
from inference.ctc_decoder import GreedyDecoder, PrefixBeamDecoder, DecoderOutput
from inference.refinement import TokenRefiner, RefinementOutput
from inference.model_loader import load_model_bundle, ModelBundle
from inference.word_fallback import WordFallbackRecognizer
from inference.fingerspell_fallback import FingerspellFallbackRecognizer
from inference.fallback_controller import FallbackController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video Feature Extraction
# ---------------------------------------------------------------------------

def extract_video_landmarks(
    video_path: str,
) -> Optional[np.ndarray]:
    """
    Extract landmarks from every frame of a video file.

    Args:
        video_path: Path to video file.

    Returns:
        (T, 225) numpy array or None if extraction fails.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.error("mediapipe required. Install: pip install mediapipe")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    all_frames = []

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)

            left_hand = np.zeros(63, dtype=np.float32)
            right_hand = np.zeros(63, dtype=np.float32)
            pose_vec = np.zeros(99, dtype=np.float32)

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hlm, hnd in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    coords = []
                    for lm in hlm.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    arr = np.array(coords, dtype=np.float32)
                    if hnd.classification[0].label == "Left":
                        left_hand = arr
                    else:
                        right_hand = arr

            if pose_results.pose_landmarks:
                coords = []
                for lm in pose_results.pose_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                pose_vec = np.array(coords, dtype=np.float32)

            all_frames.append(np.concatenate([left_hand, right_hand, pose_vec]))

    cap.release()

    if not all_frames:
        logger.warning(f"No frames extracted from {video_path}")
        return None

    return np.array(all_frames, dtype=np.float32)


def load_pose_file(pose_path: str) -> Optional[np.ndarray]:
    """Load pre-extracted landmarks from .npy file."""
    try:
        data = np.load(pose_path, allow_pickle=True)
        data = np.array(data, dtype=np.float32)

        if data.ndim == 2 and data.shape[1] == 225:
            return np.nan_to_num(data, nan=0.0)
        elif data.ndim == 3 and data.shape[1] * data.shape[2] == 225:
            return np.nan_to_num(data.reshape(data.shape[0], 225), nan=0.0)
        else:
            logger.warning(f"Unexpected shape {data.shape} in {pose_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load {pose_path}: {e}")
        return None


def build_features_from_landmarks(
    landmarks: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Build 450-dim features and velocity magnitudes from landmarks.

    Args:
        landmarks: (T, 225) raw landmarks.
        mean, std: Normalization stats (450,) each.

    Returns:
        features:  (T, 450) normalized feature array.
        vel_mags:  (T,) velocity magnitudes.
    """
    T = landmarks.shape[0]

    # Velocity
    velocity = np.zeros_like(landmarks)
    if T > 1:
        velocity[1:] = landmarks[1:] - landmarks[:-1]

    # Concatenate
    features = np.concatenate([landmarks, velocity], axis=1)  # (T, 450)

    # Normalize
    if mean is not None and std is not None:
        features = (features - mean) / np.maximum(std, 1e-8)

    # Velocity magnitudes
    vel_mags = np.linalg.norm(velocity, axis=1).tolist()

    return features, vel_mags


# ---------------------------------------------------------------------------
# Edit distance for WER
# ---------------------------------------------------------------------------

def _edit_distance(ref: List[int], hyp: List[int]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


# ---------------------------------------------------------------------------
# Video Inference Engine
# ---------------------------------------------------------------------------

class VideoInference:
    """
    Offline video inference engine.

    Processes video files through the full pipeline and produces
    structured JSON results.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        config: InferenceConfig,
    ):
        self.bundle = bundle
        self.config = config
        self.timer = ProfileTimer()

        # Decoder
        if config.use_beam_search:
            self.decoder = PrefixBeamDecoder(
                blank_id=config.blank_id,
                beam_width=config.beam_width,
                id2word=bundle.id2word,
            )
        else:
            self.decoder = GreedyDecoder(
                blank_id=config.blank_id,
                id2word=bundle.id2word,
            )

        # Refiner
        self.refiner = TokenRefiner(
            confidence_threshold=config.confidence_threshold,
            uncertainty_threshold=config.uncertainty_threshold,
            min_token_duration=config.min_token_duration,
            transition_max_frames=config.transition_max_frames,
            motion_suppression_enabled=config.motion_suppression_enabled,
            motion_velocity_threshold=config.motion_velocity_threshold,
            vocabulary=set(bundle.id2word.values()) if bundle.id2word else None,
        )
        self.word_fallback = WordFallbackRecognizer(
            word2id=bundle.word2id,
            model_path=config.word_fallback_model_path,
        )
        self.fingerspell_fallback = FingerspellFallbackRecognizer(
            word2id=bundle.word2id,
            model_path=config.fingerspell_model_path,
        )
        self.fallback_controller = FallbackController(
            word2id=bundle.word2id,
            sentence_accept_threshold=config.sentence_accept_threshold,
            span_uncertainty_threshold=config.span_uncertainty_threshold,
            span_confidence_threshold=config.confidence_threshold,
            word_accept_threshold=config.word_accept_threshold,
            spell_accept_threshold=config.spell_accept_threshold,
            motion_threshold=config.motion_velocity_threshold,
            min_fallback_frames=config.min_fallback_frames,
            enable_word_fallback=config.enable_word_fallback,
            enable_fingerspell_fallback=config.enable_fingerspell_fallback,
            word_fallback=self.word_fallback,
            fingerspell_fallback=self.fingerspell_fallback,
        )

    def process_video(
        self,
        video_path: str,
        reference: Optional[str] = None,
    ) -> Dict:
        """
        Process a single video file.

        Args:
            video_path: Path to video or .npy pose file.
            reference:  Optional reference text for comparison.

        Returns:
            Dict with prediction, metrics, and timing.
        """
        path = Path(video_path)

        # Extract landmarks
        with self.timer.measure("extraction"):
            if path.suffix == ".npy":
                landmarks = load_pose_file(str(path))
            else:
                landmarks = extract_video_landmarks(str(path))

        if landmarks is None:
            return {"error": f"Failed to extract landmarks from {path.name}",
                    "file": str(path)}

        # Build features
        with self.timer.measure("features"):
            features, vel_mags = build_features_from_landmarks(
                landmarks, self.bundle.mean, self.bundle.std
            )

        # Model forward
        with self.timer.measure("model"):
            log_probs, aux = self.bundle.predict_with_aux(features)

        # Decode
        with self.timer.measure("decode"):
            frame_uncertainty = aux["tup"]["frame_uncertainty"].squeeze(0)
            decoder_output = self.decoder.decode(
                log_probs,
                frame_uncertainties=frame_uncertainty,
            )

        # Fallback routing
        with self.timer.measure("fallback"):
            fallback_result = self.fallback_controller.route(
                decoder_output,
                features=features,
                velocity_magnitudes=vel_mags,
            )

        # Refine
        with self.timer.measure("refine"):
            refined = self.refiner.refine(
                fallback_result.decoder_output,
                vel_mags,
            )

        # Build result
        result = {
            "file": str(path),
            "frames": landmarks.shape[0],
            "prediction": {
                "raw_tokens": decoder_output.tokens,
                "raw_text": decoder_output.text,
                "raw_confidence": round(decoder_output.sequence_confidence, 4),
                "raw_mean_uncertainty": round(fallback_result.mean_uncertainty, 4),
                "accepted_sentence": fallback_result.accepted_sentence,
                "fallback_events": [e.to_dict() for e in fallback_result.routing_events],
                "routed_tokens": fallback_result.decoder_output.tokens,
                "routed_text": fallback_result.decoder_output.text,
                "refined_tokens": refined.tokens,
                "refined_text": refined.display_text,
                "rules_fired": refined.rules_fired,
            },
            "timing": self.timer.summary(),
        }

        # Compare to reference if provided
        if reference:
            ref_tokens = reference.lower().split()
            pred_tokens = refined.tokens

            # Build token id sequences for WER
            ref_ids = [self.bundle.word2id.get(w, 1) for w in ref_tokens]
            pred_ids = refined.token_ids

            wer_dist = _edit_distance(ref_ids, pred_ids)
            wer = wer_dist / max(len(ref_ids), 1)
            exact_match = ref_tokens == pred_tokens

            result["reference"] = {
                "text": reference,
                "tokens": ref_tokens,
            }
            result["metrics"] = {
                "wer": round(wer, 4),
                "edit_distance": wer_dist,
                "exact_match": exact_match,
            }

        self.timer.reset()
        return result

    def process_directory(
        self,
        dir_path: str,
        references: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Process all video/pose files in a directory.

        Args:
            dir_path:   Directory containing .mp4, .avi, .npy files.
            references: Optional dict mapping filename -> reference text.

        Returns:
            Dict with per-file results and aggregate metrics.
        """
        directory = Path(dir_path)
        if not directory.exists():
            return {"error": f"Directory not found: {dir_path}"}

        # Find all processable files
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".npy"}
        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in extensions
        )

        if not files:
            return {"error": f"No video/pose files found in {dir_path}"}

        logger.info(f"Processing {len(files)} files from {dir_path}")

        results = []
        total_wer_sum = 0.0
        total_wer_count = 0
        exact_matches = 0

        for i, f in enumerate(files):
            ref = references.get(f.name) if references else None
            logger.info(f"  [{i+1}/{len(files)}] {f.name}")

            result = self.process_video(str(f), reference=ref)
            results.append(result)

            if "metrics" in result:
                total_wer_sum += result["metrics"]["wer"]
                total_wer_count += 1
                if result["metrics"]["exact_match"]:
                    exact_matches += 1

        # Aggregate
        output = {
            "directory": str(directory),
            "file_count": len(files),
            "label_type": self.bundle.label_type,
            "results": results,
        }

        if total_wer_count > 0:
            output["aggregate_metrics"] = {
                "mean_wer": round(total_wer_sum / total_wer_count, 4),
                "sequence_accuracy": round(exact_matches / total_wer_count, 4),
                "evaluated_count": total_wer_count,
            }

        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ISL Video Inference")
    parser.add_argument("--video", type=str, help="Path to a single video or .npy file")
    parser.add_argument("--video-dir", type=str, help="Directory of videos")
    parser.add_argument("--reference", type=str, help="Reference text for comparison")
    parser.add_argument("--references-json", type=str,
                       help="JSON file mapping filename -> reference text")
    parser.add_argument("--output", type=str, default="./inference_results.json",
                       help="Output JSON path")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--vocab", type=str, default="./checkpoints/vocab.json")
    parser.add_argument("--norm-stats", type=str, default="./data_iSign/norm_stats.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--beam-width", type=int, default=5)
    args = parser.parse_args()

    # Load model
    bundle = load_model_bundle(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )

    config = InferenceConfig(
        device=args.device,
        use_beam_search=args.beam,
        beam_width=args.beam_width,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_stats_path=args.norm_stats,
    )

    engine = VideoInference(bundle, config)

    # Process
    if args.video:
        result = engine.process_video(args.video, reference=args.reference)
        results = result
    elif args.video_dir:
        refs = None
        if args.references_json:
            with open(args.references_json) as f:
                refs = json.load(f)
        results = engine.process_directory(args.video_dir, references=refs)
    else:
        parser.error("Provide --video or --video-dir")
        return

    # Print summary
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)

    if isinstance(results, dict) and "results" in results:
        # Directory mode
        print(f"  Files processed: {results['file_count']}")
        if "aggregate_metrics" in results:
            agg = results["aggregate_metrics"]
            print(f"  Mean WER:        {agg['mean_wer']*100:.2f}%")
            print(f"  Seq Accuracy:    {agg['sequence_accuracy']*100:.2f}%")
    elif isinstance(results, dict) and "prediction" in results:
        # Single file mode
        pred = results["prediction"]
        print(f"  File:      {results['file']}")
        print(f"  Frames:    {results['frames']}")
        print(f"  Raw text:  {pred['raw_text']}")
        print(f"  Refined:   {pred['refined_text']}")
        if "metrics" in results:
            m = results["metrics"]
            print(f"  WER:       {m['wer']*100:.2f}%")
            print(f"  Exact:     {m['exact_match']}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
