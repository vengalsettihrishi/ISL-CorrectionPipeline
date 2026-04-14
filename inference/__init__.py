"""
inference -- Sprint 3: Real-time ISL recognition inference engine.

This is a GLOSS-FREE pipeline. The model predicts English word tokens
via CTC. All outputs, metrics, and UI use English-token terminology.

Modules:
    ctc_decoder     : Greedy and prefix beam search CTC decoding
    refinement      : Motion-aware token refinement (6 operations)
    model_loader    : Checkpoint, vocabulary, and normalization loader
    realtime        : Live webcam inference loop
    sliding_window  : Overlapping window continuous inference
    video_inference : Offline video/batch inference with JSON output
    utils           : Profiling timer and inference configuration

Quick Start:
    # Offline inference
    python -m inference.video_inference --video path/to/video.mp4

    # Live webcam
    python -m inference.realtime --checkpoint ./checkpoints/best_model.pth
"""

from inference.utils import ProfileTimer, InferenceConfig
from inference.ctc_decoder import (
    GreedyDecoder,
    PrefixBeamDecoder,
    DecoderOutput,
    TokenSpan,
)
from inference.refinement import TokenRefiner, RefinementOutput
from inference.model_loader import load_model_bundle, ModelBundle

__all__ = [
    "ProfileTimer",
    "InferenceConfig",
    "GreedyDecoder",
    "PrefixBeamDecoder",
    "DecoderOutput",
    "TokenSpan",
    "TokenRefiner",
    "RefinementOutput",
    "load_model_bundle",
    "ModelBundle",
]
