"""
inference/realtime.py -- Live webcam inference for ISL recognition.

Main loop:
    1. Capture frame from webcam
    2. Run MediaPipe Hands + Pose in tracking mode
    3. Produce 225-dim landmark vector
    4. Compute velocity from previous frame
    5. Concatenate to 450-dim feature
    6. Normalize using saved training statistics
    7. Append to pre-allocated frame buffer
    8. Detect utterance boundary (pause or manual trigger)
    9. Run model -> decoder -> refinement
    10. Display cleaned output on screen

Controls:
    SPACE  = force decode current buffer
    R      = reset buffer
    Q/ESC  = quit

Usage:
    python -m inference.realtime
    python -m inference.realtime --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

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
# Landmark Extraction (MediaPipe)
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """
    Extract MediaPipe Hands + Pose landmarks from video frames.

    Produces a 225-dim vector per frame:
        [left_hand(63) | right_hand(63) | pose(99)]

    Uses tracking mode (static_image_mode=False) for temporal coherence.
    """

    def __init__(self, config: InferenceConfig):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is required for realtime inference. "
                "Install: pip install mediapipe"
            )

        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 225-dim landmark vector from a BGR frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3).

        Returns:
            np.ndarray of shape (225,), dtype float32.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        pose_results = self.pose.process(rgb)

        # Initialize zeros
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)
        pose_vec = np.zeros(99, dtype=np.float32)

        # Hands
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_lm, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                coords = []
                for lm in hand_lm.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                coords_arr = np.array(coords, dtype=np.float32)

                label = handedness.classification[0].label
                if label == "Left":
                    left_hand = coords_arr
                else:
                    right_hand = coords_arr

        # Pose
        if pose_results.pose_landmarks:
            coords = []
            for lm in pose_results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            pose_vec = np.array(coords, dtype=np.float32)

        return np.concatenate([left_hand, right_hand, pose_vec])

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
        self.pose.close()


# ---------------------------------------------------------------------------
# Frame Buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """
    Pre-allocated circular buffer for frame features.

    Attributes:
        landmarks: (max_frames, 225) buffer for raw landmarks.
        features:  computed on demand as [landmarks, velocity] = (T, 450).
    """

    def __init__(self, max_frames: int = 300, landmark_dim: int = 225):
        self.max_frames = max_frames
        self.landmark_dim = landmark_dim
        self.landmarks = np.zeros(
            (max_frames, landmark_dim), dtype=np.float32
        )
        self.count = 0
        self.prev_landmarks = None

    def append(self, landmarks: np.ndarray) -> None:
        """Add a frame's landmarks to the buffer."""
        if self.count >= self.max_frames:
            # Shift buffer left by 1 (drop oldest frame)
            self.landmarks[:-1] = self.landmarks[1:]
            self.count = self.max_frames - 1

        self.landmarks[self.count] = landmarks
        self.count += 1
        self.prev_landmarks = landmarks.copy()

    def get_features(self) -> np.ndarray:
        """
        Build (T, 450) feature array from buffered landmarks.

        Computes velocity and concatenates: [landmarks, velocity].
        """
        if self.count == 0:
            return np.zeros((0, self.landmark_dim * 2), dtype=np.float32)

        lm = self.landmarks[:self.count]  # (T, 225)

        # Velocity: v_t = x_t - x_{t-1}, v_0 = zeros
        velocity = np.zeros_like(lm)
        if self.count > 1:
            velocity[1:] = lm[1:] - lm[:-1]

        return np.concatenate([lm, velocity], axis=1)  # (T, 450)

    def get_velocity_magnitudes(self) -> list:
        """Get per-frame velocity magnitude for refinement."""
        if self.count <= 1:
            return [0.0] * self.count

        lm = self.landmarks[:self.count]
        velocity = np.zeros_like(lm)
        velocity[1:] = lm[1:] - lm[:-1]

        # L2 norm of velocity per frame
        mags = np.linalg.norm(velocity, axis=1).tolist()
        return mags

    def reset(self) -> None:
        """Clear the buffer."""
        self.count = 0
        self.prev_landmarks = None

    @property
    def is_full(self) -> bool:
        return self.count >= self.max_frames


# ---------------------------------------------------------------------------
# Pause Detector
# ---------------------------------------------------------------------------

class PauseDetector:
    """Detect utterance boundaries via velocity-based pause detection."""

    def __init__(
        self,
        velocity_threshold: float = 0.01,
        pause_frames: int = 15,
    ):
        self.velocity_threshold = velocity_threshold
        self.pause_frames = pause_frames
        self.low_velocity_count = 0

    def update(self, landmarks: np.ndarray, prev_landmarks: Optional[np.ndarray]) -> bool:
        """
        Update with new frame. Returns True if pause detected.

        Args:
            landmarks:      Current 225-dim landmarks.
            prev_landmarks: Previous 225-dim landmarks (or None).

        Returns:
            True if a pause has been detected (enough low-velocity frames).
        """
        if prev_landmarks is None:
            self.low_velocity_count = 0
            return False

        velocity = np.linalg.norm(landmarks - prev_landmarks)

        if velocity < self.velocity_threshold:
            self.low_velocity_count += 1
        else:
            self.low_velocity_count = 0

        return self.low_velocity_count >= self.pause_frames

    def reset(self):
        self.low_velocity_count = 0


# ---------------------------------------------------------------------------
# Realtime Inference Engine
# ---------------------------------------------------------------------------

class RealtimeInference:
    """
    Live webcam ISL recognition engine.

    Combines:
        - MediaPipe landmark extraction
        - Velocity computation
        - Feature normalization
        - Model inference
        - CTC decoding
        - Token refinement
        - Pause-based utterance boundary detection
        - OpenCV display with overlay
    """

    def __init__(
        self,
        bundle: ModelBundle,
        config: InferenceConfig,
    ):
        self.bundle = bundle
        self.config = config
        self.timer = ProfileTimer()

        # Components
        self.extractor = LandmarkExtractor(config)
        self.buffer = FrameBuffer(max_frames=config.max_buffer_frames)
        self.pause_detector = PauseDetector(
            velocity_threshold=config.pause_velocity_threshold,
            pause_frames=config.pause_frame_count,
        )

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

        # State
        self.last_result_text = ""
        self.last_status_text = ""
        self.fps_history = []
        self.frame_count = 0

    def run(self) -> None:
        """Main webcam inference loop."""
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)

        print("\n" + "=" * 60)
        print("ISL Realtime Inference")
        print("=" * 60)
        print(f"  Model:     {self.bundle.model.count_parameters():,} params")
        print(f"  Vocab:     {self.bundle.vocab_size} tokens ({self.bundle.label_type})")
        print(f"  Device:    {self.bundle.device}")
        print(f"  Decoder:   {'Beam Search' if self.config.use_beam_search else 'Greedy'}")
        print(f"  Buffer:    max {self.config.max_buffer_frames} frames")
        print()
        print("  Controls:")
        print("    SPACE = decode current buffer")
        print("    R     = reset buffer")
        print("    Q/ESC = quit")
        print("=" * 60 + "\n")

        try:
            while True:
                loop_start = time.perf_counter()

                # 1. Capture frame
                with self.timer.measure("capture"):
                    ret, frame = cap.read()
                    if not ret:
                        break

                # 2. Extract landmarks
                with self.timer.measure("mediapipe"):
                    landmarks = self.extractor.extract(frame)

                # 3. Check for pause
                pause_detected = self.pause_detector.update(
                    landmarks, self.buffer.prev_landmarks
                )

                # 4. Buffer frame
                self.buffer.append(landmarks)
                self.frame_count += 1

                # 5. Check decode triggers
                should_decode = False
                trigger = ""

                if pause_detected and self.buffer.count > 10:
                    should_decode = True
                    trigger = "pause"
                elif self.buffer.is_full:
                    should_decode = True
                    trigger = "buffer_full"

                # Check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and self.buffer.count > 5:
                    should_decode = True
                    trigger = "manual"
                elif key == ord('r'):
                    self.buffer.reset()
                    self.pause_detector.reset()
                    self.last_result_text = "[Buffer Reset]"
                    trigger = "reset"
                elif key == ord('q') or key == 27:
                    break

                # 6. Decode if triggered
                if should_decode:
                    self._decode_buffer()
                    self.buffer.reset()
                    self.pause_detector.reset()

                # 7. Display
                fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                self._draw_overlay(frame, avg_fps, trigger)
                cv2.imshow("ISL Recognition", frame)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            self.timer.print_summary()

    def _decode_buffer(self) -> None:
        """Decode the current buffer contents."""
        import torch

        # Build features
        with self.timer.measure("features"):
            features = self.buffer.get_features()  # (T, 450)
            features = self.bundle.normalize(features)
            vel_mags = self.buffer.get_velocity_magnitudes()

        # Model forward
        with self.timer.measure("model"):
            log_probs, aux = self.bundle.predict_with_aux(features)

        # Decode
        with self.timer.measure("decode"):
            decoder_output = self.decoder.decode(
                log_probs,
                frame_uncertainties=aux["tup"]["frame_uncertainty"].squeeze(0),
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

        self.last_result_text = refined.display_text or "[empty]"
        self.last_status_text = (
            f"conf={fallback_result.sentence_confidence:.2f} "
            f"unc={fallback_result.mean_uncertainty:.2f} "
            f"events={len(fallback_result.routing_events)}"
        )

        # Print to console
        print(f"  [{self.buffer.count} frames] -> {self.last_result_text}")
        print(f"    {self.last_status_text}")
        if refined.rules_fired:
            for rule in refined.rules_fired[:3]:
                print(f"    - {rule}")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        trigger: str,
    ) -> None:
        """Draw status overlay on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
        )

        # Buffer status
        buf_pct = int(100 * self.buffer.count / self.config.max_buffer_frames)
        cv2.putText(
            frame, f"Buffer: {self.buffer.count}/{self.config.max_buffer_frames} ({buf_pct}%)",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        # Trigger indicator
        if trigger:
            cv2.putText(
                frame, f"[{trigger}]", (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
            )

        # Result text at bottom
        if self.last_result_text:
            cv2.rectangle(
                frame, (0, h - 50), (w, h), (0, 0, 0), -1
            )
            cv2.putText(
                frame, self.last_result_text[:60],
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2,
            )
        if self.last_status_text:
            cv2.putText(
                frame, self.last_status_text[:50],
                (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1,
            )

        # Latency breakdown (right side)
        y_pos = 100
        for stage in ["mediapipe", "model", "decode", "fallback", "refine"]:
            ms = self.timer.get_last(stage)
            if ms > 0:
                cv2.putText(
                    frame, f"{stage}: {ms:.1f}ms",
                    (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1,
                )
                y_pos += 20


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ISL Realtime Inference")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--vocab", type=str, default="./checkpoints/vocab.json")
    parser.add_argument("--norm-stats", type=str, default="./data_iSign/norm_stats.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--beam", action="store_true", help="Use beam search")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-buffer", type=int, default=300)
    args = parser.parse_args()

    # Load model
    bundle = load_model_bundle(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )

    # Build config
    config = InferenceConfig(
        device=args.device,
        camera_index=args.camera,
        use_beam_search=args.beam,
        beam_width=args.beam_width,
        max_buffer_frames=args.max_buffer,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_stats_path=args.norm_stats,
    )

    # Run
    engine = RealtimeInference(bundle, config)
    engine.run()


if __name__ == "__main__":
    main()
