"""
output/full_pipeline.py — End-to-End ISL Recognition Runner.

Single entry point that:
    - Initializes model, decoder, refinement, correction, translation, and TTS
    - Runs in webcam, video, or benchmark mode
    - Displays text outputs with rich formatting
    - Optionally plays English and Hindi speech
    - Logs results to JSON

Usage:
    python output/full_pipeline.py --mode webcam
    python output/full_pipeline.py --mode video --input test_video.mp4
    python output/full_pipeline.py --mode benchmark --data_dir ./data_iSign/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Full Pipeline
# ============================================================================

class FullPipeline:
    """
    End-to-end ISL recognition pipeline.

    Combines:
        - Model inference (Sprint 2/3)
        - CTC decoding + refinement (Sprint 3)
        - English post-correction (Sprint 4)
        - Hindi translation (Sprint 4)
        - TTS output (Sprint 4)
        - Display (Sprint 4)
    """

    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/best_model.pth",
        vocab_path: str = "./checkpoints/vocab.json",
        norm_stats_path: str = "./data_iSign/norm_stats.npz",
        kenlm_model_path: Optional[str] = None,
        device: str = "cpu",
        enable_tts: bool = False,
        tts_engine: str = "pyttsx3",
        enable_hindi: bool = True,
        use_beam_search: bool = False,
        beam_width: int = 5,
    ):
        self.device = device
        self.enable_tts = enable_tts
        self.enable_hindi = enable_hindi
        self._model_bundle = None
        self._decoder = None
        self._refiner = None
        self._correction_pipeline = None
        self._tts = None
        self._display = None

        # Store config for lazy init
        self._config = {
            "checkpoint_path": checkpoint_path,
            "vocab_path": vocab_path,
            "norm_stats_path": norm_stats_path,
            "kenlm_model_path": kenlm_model_path,
            "device": device,
            "use_beam_search": use_beam_search,
            "beam_width": beam_width,
            "tts_engine": tts_engine,
        }

        # Initialize correction pipeline (always available)
        self._init_correction(kenlm_model_path)

        # Initialize TTS (optional)
        if enable_tts:
            self._init_tts(tts_engine)

    def _init_correction(self, kenlm_model_path: Optional[str]):
        """Initialize the correction pipeline."""
        from correction.config import CorrectionConfig
        from correction.pipeline import CorrectionPipeline

        config = CorrectionConfig(
            kenlm_model_path=kenlm_model_path,
            enable_hindi_translation=self.enable_hindi,
        )
        self._correction_pipeline = CorrectionPipeline(config)

    def _init_tts(self, engine: str):
        """Initialize TTS engine."""
        try:
            from tts_output import TTSEngine
            self._tts = TTSEngine(engine=engine, async_mode=True)
            if self._tts.available:
                logger.info(f"TTS initialized ({engine})")
            else:
                logger.warning("TTS engine not available")
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")

    def _init_model(self):
        """Lazily initialize model bundle, decoder, and refiner."""
        if self._model_bundle is not None:
            return True

        try:
            from inference.model_loader import load_model_bundle
            from inference.ctc_decoder import GreedyDecoder, PrefixBeamDecoder
            from inference.refinement import TokenRefiner

            logger.info("Loading model bundle...")
            self._model_bundle = load_model_bundle(
                checkpoint_path=self._config["checkpoint_path"],
                vocab_path=self._config["vocab_path"],
                norm_stats_path=self._config["norm_stats_path"],
                device=self._config["device"],
            )

            # Decoder
            if self._config["use_beam_search"]:
                self._decoder = PrefixBeamDecoder(
                    blank_id=0,
                    beam_width=self._config["beam_width"],
                    id2word=self._model_bundle.id2word,
                )
            else:
                self._decoder = GreedyDecoder(
                    blank_id=0,
                    id2word=self._model_bundle.id2word,
                )

            # Refiner
            self._refiner = TokenRefiner(
                confidence_threshold=0.3,
                min_token_duration=3,
                vocabulary=set(self._model_bundle.id2word.values()),
            )

            logger.info("Model bundle ready")
            return True

        except FileNotFoundError as e:
            logger.error(f"Model loading failed: {e}")
            logger.error("Ensure checkpoint and vocab files exist.")
            return False
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Mode: Video
    # ------------------------------------------------------------------

    def run_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Process a single video file through the full pipeline.

        Args:
            video_path:  Path to video (.mp4) or pose file (.npy).
            output_path: Optional JSON output path.

        Returns:
            Dict with prediction, correction, and translation results.
        """
        from output.display import DisplayEngine

        display = DisplayEngine(mode="video")

        if not self._init_model():
            return {"error": "Model initialization failed"}

        from inference.video_inference import (
            load_pose_file, extract_video_landmarks,
            build_features_from_landmarks,
        )

        path = Path(video_path)
        logger.info(f"Processing: {path.name}")

        # Extract features
        t_start = time.perf_counter()

        if path.suffix == ".npy":
            landmarks = load_pose_file(str(path))
        else:
            landmarks = extract_video_landmarks(str(path))

        if landmarks is None:
            return {"error": f"Failed to extract landmarks from {path.name}"}

        features, vel_mags = build_features_from_landmarks(
            landmarks, self._model_bundle.mean, self._model_bundle.std,
        )

        # Model forward
        log_probs = self._model_bundle.predict(features)

        # Decode + refine
        decoder_output = self._decoder.decode(log_probs)
        refined = self._refiner.refine(decoder_output, vel_mags)

        t_inference = (time.perf_counter() - t_start) * 1000

        # Correct
        correction_result = self._correction_pipeline.correct(
            refined.tokens,
            token_confidences=refined.spans and [s.confidence for s in refined.spans],
        )

        # Display
        display.show(
            tokens=refined.tokens,
            corrected_english=correction_result.corrected_english,
            hindi_translation=correction_result.hindi_translation,
            token_confidences=[s.confidence for s in refined.spans] if refined.spans else None,
            timing_ms={
                "inference_ms": t_inference,
                **correction_result.timing_ms,
            },
        )

        # TTS
        if self._tts and self._tts.available:
            self._tts.speak_bilingual(
                correction_result.corrected_english,
                correction_result.hindi_translation,
            )

        result = {
            "file": str(path),
            "frames": landmarks.shape[0],
            "raw_tokens": refined.tokens,
            "corrected_english": correction_result.corrected_english,
            "hindi_translation": correction_result.hindi_translation,
            "rules_applied": correction_result.rules_applied,
            "kenlm_used": correction_result.kenlm_used,
            "timing_ms": {
                "inference_ms": t_inference,
                **correction_result.timing_ms,
            },
        }

        # Save if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")

        return result

    # ------------------------------------------------------------------
    # Mode: Webcam
    # ------------------------------------------------------------------

    def run_webcam(self) -> None:
        """
        Run live webcam inference with real-time display.

        Captures frames, extracts landmarks, runs inference,
        applies correction, and displays results continuously.
        Press 'q' to quit.
        """
        from output.display import DisplayEngine

        if not self._init_model():
            logger.error("Model initialization failed. Cannot start webcam.")
            return

        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("OpenCV required for webcam mode. Install: pip install opencv-python")
            return

        display = DisplayEngine(mode="webcam", compact=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return

        logger.info("Webcam started. Press 'q' to quit.")

        frame_buffer = []
        max_buffer = 150  # frames before forced decode

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract landmarks from frame
                from inference.video_inference import extract_video_landmarks
                import tempfile
                import os

                # Save frame as temp image for mediapipe
                frame_buffer.append(frame)

                fps = display.update_fps()

                if len(frame_buffer) >= max_buffer:
                    # Process accumulated frames
                    logger.info(f"Processing {len(frame_buffer)} buffered frames...")

                    # Save as temporary video and process
                    tmp_path = os.path.join(".", "_webcam_tmp.avi")
                    h, w = frame_buffer[0].shape[:2]
                    writer = cv2.VideoWriter(
                        tmp_path, cv2.VideoWriter_fourcc(*"XVID"),
                        30, (w, h),
                    )
                    for f in frame_buffer:
                        writer.write(f)
                    writer.release()

                    from inference.video_inference import (
                        extract_video_landmarks, build_features_from_landmarks,
                    )

                    landmarks = extract_video_landmarks(tmp_path)
                    if landmarks is not None:
                        features, vel_mags = build_features_from_landmarks(
                            landmarks, self._model_bundle.mean, self._model_bundle.std,
                        )
                        log_probs = self._model_bundle.predict(features)
                        decoder_output = self._decoder.decode(log_probs)
                        refined = self._refiner.refine(decoder_output, vel_mags)

                        correction = self._correction_pipeline.correct(refined.tokens)

                        display.show(
                            tokens=refined.tokens,
                            corrected_english=correction.corrected_english,
                            hindi_translation=correction.hindi_translation,
                            fps=fps,
                        )

                        if self._tts and self._tts.available:
                            self._tts.speak_bilingual(
                                correction.corrected_english,
                                correction.hindi_translation,
                            )

                    # Cleanup
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    frame_buffer = []

                # Show frame
                cv2.putText(
                    frame, f"FPS: {fps:.1f}  Frames: {len(frame_buffer)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("ISL Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam stopped.")

    # ------------------------------------------------------------------
    # Mode: Benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        data_dir: str = "./data_iSign",
        csv_path: str = "./data_iSign/iSign_v1.1.csv",
        max_samples: Optional[int] = None,
        results_dir: str = "./results",
        correction_only: bool = False,
    ) -> Dict:
        """
        Run full benchmark on the iSign test set.

        Args:
            data_dir:        Root data directory.
            csv_path:        Path to iSign CSV with references.
            max_samples:     Limit number of samples.
            results_dir:     Directory to save results.
            correction_only: Run correction-only benchmark (no model needed).

        Returns:
            Benchmark results dict.
        """
        from output.display import DisplayEngine
        from correction.benchmark import BenchmarkEngine
        from correction.config import CorrectionConfig

        display = DisplayEngine(mode="benchmark")

        config = CorrectionConfig(
            kenlm_model_path=self._config.get("kenlm_model_path"),
            results_dir=results_dir,
            data_dir=data_dir,
            csv_path=csv_path,
            checkpoint_path=self._config["checkpoint_path"],
            vocab_path=self._config["vocab_path"],
            norm_stats_path=self._config["norm_stats_path"],
            device=self._config["device"],
        )

        engine = BenchmarkEngine(config)

        if correction_only:
            csv_p = csv_path if Path(csv_path).exists() else None
            results = engine.run_correction_benchmark(
                csv_path=csv_p,
                max_samples=max_samples,
            )
        else:
            results = engine.run_full_benchmark(
                data_dir=data_dir,
                checkpoint_path=self._config["checkpoint_path"],
                vocab_path=self._config["vocab_path"],
                norm_stats_path=self._config["norm_stats_path"],
                csv_path=csv_path,
                device=self._config["device"],
                max_samples=max_samples,
            )

        # Display results
        from dataclasses import asdict
        results_dict = asdict(results) if hasattr(results, '__dataclass_fields__') else results

        display.show_summary(
            metrics=results_dict.get("overall_metrics", {}),
            num_samples=results_dict.get("num_samples", 0),
            timing=results_dict.get("timing_summary"),
        )

        # Display ablation table
        ablations = results_dict.get("ablation_results", [])
        if ablations:
            display.show_ablation_table(ablations)

        return results_dict

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get pipeline component status."""
        status = {
            "model_loaded": self._model_bundle is not None,
            "correction_ready": self._correction_pipeline is not None,
            "tts_available": self._tts.available if self._tts else False,
            "device": self.device,
            "config": self._config,
        }
        if self._model_bundle:
            status["model_vocab_size"] = self._model_bundle.vocab_size
            status["model_label_type"] = self._model_bundle.label_type
        return status


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ISL Full Pipeline Runner (Sprint 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python output/full_pipeline.py --mode webcam
  python output/full_pipeline.py --mode video --input test_video.mp4
  python output/full_pipeline.py --mode benchmark --data_dir ./data_iSign/
  python output/full_pipeline.py --mode benchmark --correction-only
        """,
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["webcam", "video", "benchmark"],
        help="Execution mode",
    )
    parser.add_argument("--input", type=str, help="Input video/pose file (video mode)")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--data_dir", type=str, default="./data_iSign")
    parser.add_argument("--csv", type=str, default="./data_iSign/iSign_v1.1.csv")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--vocab", type=str, default="./checkpoints/vocab.json")
    parser.add_argument("--norm_stats", type=str, default="./data_iSign/norm_stats.npz")
    parser.add_argument("--kenlm_model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tts", action="store_true", help="Enable TTS output")
    parser.add_argument("--tts_engine", type=str, default="pyttsx3",
                        choices=["pyttsx3", "gtts"])
    parser.add_argument("--no-hindi", action="store_true",
                        help="Disable Hindi translation")
    parser.add_argument("--beam", action="store_true", help="Use beam search")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--correction-only", action="store_true",
                        help="Run correction-only benchmark (no model needed)")

    args = parser.parse_args()

    # Build pipeline
    pipeline = FullPipeline(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_stats_path=args.norm_stats,
        kenlm_model_path=args.kenlm_model,
        device=args.device,
        enable_tts=args.tts,
        tts_engine=args.tts_engine,
        enable_hindi=not args.no_hindi,
        use_beam_search=args.beam,
        beam_width=args.beam_width,
    )

    # Run mode
    if args.mode == "webcam":
        pipeline.run_webcam()

    elif args.mode == "video":
        if not args.input:
            parser.error("--input required for video mode")
        pipeline.run_video(args.input, output_path=args.output)

    elif args.mode == "benchmark":
        pipeline.run_benchmark(
            data_dir=args.data_dir,
            csv_path=args.csv,
            max_samples=args.max_samples,
            results_dir=args.results_dir,
            correction_only=args.correction_only,
        )


if __name__ == "__main__":
    main()
