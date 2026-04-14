"""
inference/sliding_window.py -- Continuous inference for long signing sessions.

Processes long feature sequences using overlapping sliding windows:
    1. Slice features into overlapping windows
    2. Decode each window independently
    3. Merge overlapping predictions using confidence-aware voting
    4. Return a single unified token sequence

Deterministic and CPU-friendly.

Usage:
    python -m inference.sliding_window
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from inference.ctc_decoder import DecoderOutput, GreedyDecoder, PrefixBeamDecoder, TokenSpan
from inference.refinement import TokenRefiner, RefinementOutput
from inference.utils import InferenceConfig


# ---------------------------------------------------------------------------
# Sliding Window Output
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowOutput:
    """Result of sliding-window continuous inference."""
    tokens: List[str]
    token_ids: List[int]
    spans: List[TokenSpan]
    display_text: str
    window_count: int
    total_frames: int
    per_window_results: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "display_text": self.display_text,
            "spans": [s.to_dict() for s in self.spans],
            "window_count": self.window_count,
            "total_frames": self.total_frames,
        }


# ---------------------------------------------------------------------------
# Sliding Window Engine
# ---------------------------------------------------------------------------

class SlidingWindowInference:
    """
    Continuous inference using overlapping sliding windows.

    Args:
        model_forward:  Callable (features_tensor) -> log_probs tensor.
        decoder:        CTC decoder instance (Greedy or Beam).
        refiner:        Token refinement instance.
        window_size:    Number of frames per window (default 150).
        stride:         Stride between windows (default 100).
        id2word:        Token ID -> word mapping.
    """

    def __init__(
        self,
        model_forward,
        decoder,
        refiner: Optional[TokenRefiner] = None,
        window_size: int = 150,
        stride: int = 100,
        id2word: Optional[Dict[int, str]] = None,
    ):
        self.model_forward = model_forward
        self.decoder = decoder
        self.refiner = refiner
        self.window_size = window_size
        self.stride = stride
        self.id2word = id2word or {}

    def process(
        self,
        features: np.ndarray,
        velocity_magnitudes: Optional[List[float]] = None,
    ) -> SlidingWindowOutput:
        """
        Process a long feature sequence using overlapping windows.

        Args:
            features: (T, 450) numpy array (already normalized).
            velocity_magnitudes: (T,) per-frame velocity for refinement.

        Returns:
            SlidingWindowOutput with merged predictions.
        """
        T = features.shape[0]

        if T <= self.window_size:
            # Short sequence: process in one shot
            return self._process_single(features, T, velocity_magnitudes)

        # Generate overlapping windows
        windows = []
        starts = list(range(0, T - self.window_size + 1, self.stride))

        # Ensure the last window covers the end
        if starts[-1] + self.window_size < T:
            starts.append(T - self.window_size)

        per_window_results = []

        for start in starts:
            end = min(start + self.window_size, T)
            window_features = features[start:end]

            # Get velocity for this window
            window_vel = None
            if velocity_magnitudes:
                window_vel = velocity_magnitudes[start:end]

            # Run model
            x = torch.from_numpy(window_features).float().unsqueeze(0)
            with torch.no_grad():
                log_probs = self.model_forward(x)
            log_probs = log_probs.squeeze(0)  # (W, V)

            # Decode
            result = self.decoder.decode(log_probs)

            # Refine if refiner available
            if self.refiner:
                refined = self.refiner.refine(result, window_vel)
                window_tokens = refined.tokens
                window_ids = refined.token_ids
                window_spans = refined.spans
            else:
                window_tokens = result.tokens
                window_ids = result.token_ids
                window_spans = result.spans

            # Adjust span frames to global coordinates
            adjusted_spans = []
            for s in window_spans:
                adjusted_spans.append(TokenSpan(
                    token=s.token,
                    token_id=s.token_id,
                    start_frame=s.start_frame + start,
                    end_frame=s.end_frame + start,
                    confidence=s.confidence,
                ))

            per_window_results.append({
                "start_frame": start,
                "end_frame": end,
                "tokens": window_tokens,
                "token_ids": window_ids,
                "spans": adjusted_spans,
            })

        # Merge overlapping predictions
        merged_spans = self._merge_windows(per_window_results, T)

        tokens = [s.token for s in merged_spans]
        token_ids = [s.token_id for s in merged_spans]
        display_text = " ".join(tokens) if tokens else ""

        return SlidingWindowOutput(
            tokens=tokens,
            token_ids=token_ids,
            spans=merged_spans,
            display_text=display_text,
            window_count=len(starts),
            total_frames=T,
            per_window_results=[
                {"start": w["start_frame"], "end": w["end_frame"],
                 "tokens": w["tokens"]}
                for w in per_window_results
            ],
        )

    def _process_single(
        self,
        features: np.ndarray,
        T: int,
        velocity_magnitudes: Optional[List[float]],
    ) -> SlidingWindowOutput:
        """Process a short sequence in one shot."""
        x = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model_forward(x)
        log_probs = log_probs.squeeze(0)

        result = self.decoder.decode(log_probs)

        if self.refiner:
            refined = self.refiner.refine(result, velocity_magnitudes)
            tokens = refined.tokens
            token_ids = refined.token_ids
            spans = refined.spans
        else:
            tokens = result.tokens
            token_ids = result.token_ids
            spans = result.spans

        return SlidingWindowOutput(
            tokens=tokens,
            token_ids=token_ids,
            spans=spans,
            display_text=" ".join(tokens) if tokens else "",
            window_count=1,
            total_frames=T,
            per_window_results=[{
                "start": 0, "end": T, "tokens": tokens,
            }],
        )

    def _merge_windows(
        self,
        window_results: List[Dict],
        total_frames: int,
    ) -> List[TokenSpan]:
        """
        Merge overlapping window predictions using confidence-aware voting.

        For each frame, collect all predictions from windows covering that frame.
        Use the highest-confidence prediction for each frame.
        Then collapse the frame-level sequence into spans.
        """
        if not window_results:
            return []

        # Frame-level voting: for each frame, track (token_id, confidence, token_name)
        frame_votes: List[List[Tuple[int, float, str]]] = [[] for _ in range(total_frames)]

        for w_result in window_results:
            for span in w_result["spans"]:
                for f in range(span.start_frame, min(span.end_frame, total_frames)):
                    frame_votes[f].append((span.token_id, span.confidence, span.token))

        # Best prediction per frame (highest confidence)
        frame_best = []
        for f in range(total_frames):
            if frame_votes[f]:
                # Pick highest confidence
                best = max(frame_votes[f], key=lambda x: x[1])
                frame_best.append(best)
            else:
                frame_best.append((0, 0.0, "<blank>"))  # blank

        # Collapse into spans (like greedy decode collapse)
        spans = []
        if not frame_best:
            return spans

        current_id, current_conf, current_token = frame_best[0]
        current_start = 0
        conf_accum = [current_conf]

        for f in range(1, total_frames):
            fid, fconf, ftok = frame_best[f]
            if fid == current_id:
                conf_accum.append(fconf)
            else:
                # Emit span (skip blanks)
                if current_id != 0:
                    mean_conf = sum(conf_accum) / len(conf_accum)
                    spans.append(TokenSpan(
                        token=current_token,
                        token_id=current_id,
                        start_frame=current_start,
                        end_frame=f,
                        confidence=mean_conf,
                    ))
                current_id = fid
                current_conf = fconf
                current_token = ftok
                current_start = f
                conf_accum = [fconf]

        # Final span
        if current_id != 0:
            mean_conf = sum(conf_accum) / len(conf_accum)
            spans.append(TokenSpan(
                token=current_token,
                token_id=current_id,
                start_frame=current_start,
                end_frame=total_frames,
                confidence=mean_conf,
            ))

        return spans


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 60)
    print("inference/sliding_window.py -- verification")
    print("=" * 60)

    # Create a simple mock model
    vocab_size = 10

    def mock_model_forward(x):
        """Mock: returns uniform log-probs with pattern injection."""
        B, T, D = x.shape
        logits = torch.randn(B, T, vocab_size)
        # Inject pattern: token 3 near start, token 7 near end
        quarter = T // 4
        logits[:, quarter:quarter*2, :] = -5.0
        logits[:, quarter:quarter*2, 3] = 2.0
        logits[:, quarter*3:, :] = -5.0
        logits[:, quarter*3:, 7] = 2.0
        return F.log_softmax(logits, dim=-1)

    id2word = {0: "<blank>", 3: "hello", 7: "world"}
    decoder = GreedyDecoder(blank_id=0, id2word=id2word)

    sw = SlidingWindowInference(
        model_forward=mock_model_forward,
        decoder=decoder,
        window_size=50,
        stride=30,
        id2word=id2word,
    )

    # Create long feature sequence (200 frames)
    features = np.random.randn(200, 450).astype(np.float32)
    result = sw.process(features)

    print(f"  Total frames:  {result.total_frames}")
    print(f"  Windows used:  {result.window_count}")
    print(f"  Output tokens: {result.tokens}")
    print(f"  Display text:  '{result.display_text}'")
    print(f"  Spans:")
    for s in result.spans:
        print(f"    {s.token:<10} frames [{s.start_frame}-{s.end_frame}) "
              f"conf={s.confidence:.3f}")

    assert result.window_count > 1, "Should use multiple windows for T=200"
    assert result.total_frames == 200

    # Test short sequence (should use single window)
    short_features = np.random.randn(30, 450).astype(np.float32)
    short_result = sw.process(short_features)
    assert short_result.window_count == 1

    print(f"\n  Short sequence: {short_result.window_count} window, "
          f"tokens={short_result.tokens}")

    print("=" * 60)
    print("[PASS] inference/sliding_window.py OK")
