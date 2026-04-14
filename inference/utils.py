"""
inference/utils.py -- Profiling timer and inference configuration.

Provides:
    - ProfileTimer: context-manager based timing for inference stages
    - InferenceConfig: dataclass with all runtime-tunable knobs
    - FrameBuffer: pre-allocated circular buffer for frame sequences

Usage:
    from inference.utils import ProfileTimer, InferenceConfig
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

class ProfileTimer:
    """
    Lightweight profiling timer for inference pipeline stages.

    Usage:
        timer = ProfileTimer()
        with timer.measure("mediapipe"):
            ...
        with timer.measure("model"):
            ...
        timer.print_summary()
    """

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._active: Dict[str, float] = {}

    class _TimerContext:
        def __init__(self, timer: "ProfileTimer", name: str):
            self._timer = timer
            self._name = name

        def __enter__(self):
            self._timer._active[self._name] = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self._timer._active.pop(self._name)
            if self._name not in self._timer._timings:
                self._timer._timings[self._name] = []
            self._timer._timings[self._name].append(elapsed * 1000)  # ms

    def measure(self, name: str) -> "_TimerContext":
        """Create a context manager that times the enclosed block."""
        return self._TimerContext(self, name)

    def record(self, name: str, elapsed_ms: float) -> None:
        """Manually record a timing."""
        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(elapsed_ms)

    def get_last(self, name: str) -> float:
        """Get the most recent timing for a stage (ms)."""
        if name in self._timings and self._timings[name]:
            return self._timings[name][-1]
        return 0.0

    def get_mean(self, name: str) -> float:
        """Get mean timing for a stage (ms)."""
        if name in self._timings and self._timings[name]:
            vals = self._timings[name]
            return sum(vals) / len(vals)
        return 0.0

    def get_total(self) -> float:
        """Sum of all mean timings (ms) -- approximate total pipeline time."""
        return sum(self.get_mean(k) for k in self._timings)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary dict: {stage: {mean_ms, min_ms, max_ms, count}}."""
        result = {}
        for name, vals in self._timings.items():
            result[name] = {
                "mean_ms": sum(vals) / len(vals),
                "min_ms": min(vals),
                "max_ms": max(vals),
                "count": len(vals),
            }
        return result

    def print_summary(self) -> None:
        """Print a formatted timing summary."""
        print("\n  Timing Breakdown:")
        print(f"  {'Stage':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
        print(f"  {'-'*57}")
        for name, stats in self.summary().items():
            print(
                f"  {name:<25} {stats['mean_ms']:>7.1f}ms "
                f"{stats['min_ms']:>7.1f}ms {stats['max_ms']:>7.1f}ms "
                f"{stats['count']:>5}"
            )
        print(f"  {'Total (mean)':<25} {self.get_total():>7.1f}ms")

    def reset(self) -> None:
        """Clear all timings."""
        self._timings.clear()
        self._active.clear()


# ---------------------------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Runtime configuration for the inference engine."""

    # --- Decoder ---
    beam_width: int = 5
    """Beam width for prefix beam search decoder."""

    blank_id: int = 0
    """CTC blank token ID."""

    # --- Refinement ---
    confidence_threshold: float = 0.3
    """Remove token spans with confidence below this threshold."""

    min_token_duration: int = 3
    """Minimum number of frames for a token span to survive filtering."""

    transition_max_frames: int = 2
    """Max frames for a noisy insertion between identical tokens."""

    motion_suppression_enabled: bool = True
    """Enable velocity-aware suppression of low-confidence short tokens."""

    motion_velocity_threshold: float = 0.05
    """Velocity magnitude above which a frame is considered high-motion."""

    # --- Realtime ---
    pause_velocity_threshold: float = 0.01
    """Velocity magnitude below which frames count as 'paused'."""

    pause_frame_count: int = 15
    """Number of consecutive low-velocity frames to trigger utterance end."""

    max_buffer_frames: int = 300
    """Maximum frames to accumulate before forced decode."""

    camera_index: int = 0
    """Webcam index for OpenCV capture."""

    display_width: int = 640
    """Display window width."""

    display_height: int = 480
    """Display window height."""

    # --- Sliding window ---
    window_size: int = 150
    """Number of frames per sliding window."""

    window_stride: int = 100
    """Stride between overlapping windows."""

    # --- Model ---
    checkpoint_path: str = "./checkpoints/best_model.pth"
    """Path to trained model checkpoint."""

    vocab_path: str = "./checkpoints/vocab.json"
    """Path to vocabulary JSON file."""

    norm_stats_path: str = "./data_iSign/norm_stats.npz"
    """Path to normalization statistics (mean, std)."""

    device: str = "cpu"
    """Inference device ('cpu' or 'cuda')."""

    # --- Output ---
    use_beam_search: bool = False
    """Use beam search instead of greedy decoding."""

    show_confidences: bool = True
    """Display per-token confidences in output."""


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("inference/utils.py -- verification")
    print("=" * 60)

    # Test ProfileTimer
    timer = ProfileTimer()
    for _ in range(5):
        with timer.measure("test_stage"):
            time.sleep(0.01)

    assert timer.get_mean("test_stage") > 5.0  # > 5ms
    assert len(timer.summary()) == 1
    timer.print_summary()

    # Test InferenceConfig
    cfg = InferenceConfig()
    print(f"\n  beam_width:       {cfg.beam_width}")
    print(f"  blank_id:         {cfg.blank_id}")
    print(f"  max_buffer:       {cfg.max_buffer_frames}")
    print(f"  window_size:      {cfg.window_size}")

    print("=" * 60)
    print("[PASS] inference/utils.py OK")
