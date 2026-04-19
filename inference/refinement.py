"""
inference/refinement.py -- Motion-aware token refinement for CTC output.

Cleans noisy CTC decoder output using five deterministic operations:

    1. Confidence filtering    -- remove low-confidence token spans
    2. Repetition merging      -- merge consecutive duplicate tokens
    3. Minimum duration filter -- remove spans shorter than N frames
    4. Transition smoothing    -- remove short noisy insertions between
                                  identical surrounding tokens
    5. Motion-aware suppress.  -- suppress short low-confidence tokens
                                  during high-motion transitions
    6. Vocabulary consistency  -- verify tokens exist in vocabulary

All operations use real span/confidence information from the decoder.
This is English-token cleanup, NOT gloss-to-English conversion.

Usage:
    python -m inference.refinement
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from inference.ctc_decoder import DecoderOutput, TokenSpan


# ---------------------------------------------------------------------------
# Refinement output
# ---------------------------------------------------------------------------

@dataclass
class RefinementOutput:
    """Result of token refinement."""
    tokens: List[str]
    token_ids: List[int]
    spans: List[TokenSpan]
    display_text: str
    rules_fired: List[str]
    removed_tokens: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "display_text": self.display_text,
            "spans": [s.to_dict() for s in self.spans],
            "rules_fired": self.rules_fired,
            "removed_count": len(self.removed_tokens),
        }


# ---------------------------------------------------------------------------
# Refinement engine
# ---------------------------------------------------------------------------

class TokenRefiner:
    """
    Lightweight deterministic token refinement for CTC output.

    Operates on English token spans. Does NOT perform grammar correction
    or gloss-to-English conversion.

    Args:
        confidence_threshold:       Min confidence for a span to survive.
        min_token_duration:         Min frame count for a span.
        transition_max_frames:      Max frames for a noisy insertion.
        motion_suppression_enabled: Enable velocity-based suppression.
        motion_velocity_threshold:  Velocity above which = high motion.
        vocabulary:                 Set of valid token strings (optional).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        uncertainty_threshold: float = 0.55,
        min_token_duration: int = 3,
        transition_max_frames: int = 2,
        motion_suppression_enabled: bool = True,
        motion_velocity_threshold: float = 0.05,
        vocabulary: Optional[Set[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.min_token_duration = min_token_duration
        self.transition_max_frames = transition_max_frames
        self.motion_suppression_enabled = motion_suppression_enabled
        self.motion_velocity_threshold = motion_velocity_threshold
        self.vocabulary = vocabulary

    def refine(
        self,
        decoder_output: DecoderOutput,
        velocity_magnitudes: Optional[List[float]] = None,
    ) -> RefinementOutput:
        """
        Apply all refinement operations to decoder output.

        Args:
            decoder_output:      Structured output from CTC decoder.
            velocity_magnitudes: Per-frame velocity magnitude for motion
                                 suppression. Shape (T,). Optional.

        Returns:
            RefinementOutput with cleaned tokens and rule trace.
        """
        spans = list(decoder_output.spans)
        rules_fired = []
        removed = []

        # 1. Confidence filtering
        spans, r, fired = self._filter_confidence(spans)
        removed.extend(r)
        rules_fired.extend(fired)

        # 2. Explicit uncertainty filtering
        spans, r, fired = self._filter_uncertainty(spans)
        removed.extend(r)
        rules_fired.extend(fired)

        # 3. Minimum duration filtering
        spans, r, fired = self._filter_min_duration(spans)
        removed.extend(r)
        rules_fired.extend(fired)

        # 4. Repetition merging (after duration filter may have removed gaps)
        spans, fired = self._merge_repetitions(spans)
        rules_fired.extend(fired)

        # 5. Transition smoothing
        spans, r, fired = self._smooth_transitions(spans)
        removed.extend(r)
        rules_fired.extend(fired)

        # 6. Motion-aware suppression (if velocity data available)
        if self.motion_suppression_enabled and velocity_magnitudes:
            spans, r, fired = self._motion_suppress(spans, velocity_magnitudes)
            removed.extend(r)
            rules_fired.extend(fired)

        # 7. Vocabulary consistency
        if self.vocabulary:
            spans, r, fired = self._check_vocabulary(spans)
            removed.extend(r)
            rules_fired.extend(fired)

        # Final merge pass after all removals
        spans, fired = self._merge_repetitions(spans)
        rules_fired.extend(fired)

        # Build output
        tokens = [s.token for s in spans]
        token_ids = [s.token_id for s in spans]
        display_text = " ".join(tokens) if tokens else ""

        return RefinementOutput(
            tokens=tokens,
            token_ids=token_ids,
            spans=spans,
            display_text=display_text,
            rules_fired=rules_fired,
            removed_tokens=removed,
        )

    # -----------------------------------------------------------------------
    # Operation 1: Confidence filtering
    # -----------------------------------------------------------------------

    def _filter_confidence(
        self, spans: List[TokenSpan]
    ) -> tuple:
        kept = []
        removed = []
        fired = []

        for s in spans:
            if s.confidence < self.confidence_threshold:
                removed.append({
                    "rule": "confidence_filter",
                    "token": s.token,
                    "confidence": round(s.confidence, 4),
                    "threshold": self.confidence_threshold,
                })
                fired.append(f"confidence_filter: removed '{s.token}' "
                           f"(conf={s.confidence:.3f} < {self.confidence_threshold})")
            else:
                kept.append(s)

        return kept, removed, fired

    # -----------------------------------------------------------------------
    # Operation 2: Repetition merging
    # -----------------------------------------------------------------------

    def _merge_repetitions(
        self, spans: List[TokenSpan]
    ) -> tuple:
        if len(spans) <= 1:
            return spans, []

        merged = [spans[0]]
        fired = []

        for s in spans[1:]:
            if s.token_id == merged[-1].token_id:
                # Merge: extend end frame, average confidence
                prev = merged[-1]
                merged[-1] = TokenSpan(
                    token=prev.token,
                    token_id=prev.token_id,
                    start_frame=prev.start_frame,
                    end_frame=s.end_frame,
                    confidence=(prev.confidence + s.confidence) / 2.0,
                    uncertainty=(prev.uncertainty + s.uncertainty) / 2.0,
                    origin=prev.origin if prev.origin == s.origin else "merged",
                )
                fired.append(
                    f"repetition_merge: merged consecutive '{s.token}' "
                    f"spans [{prev.start_frame}-{s.end_frame})"
                )
            else:
                merged.append(s)

        return merged, fired

    def _filter_uncertainty(
        self, spans: List[TokenSpan]
    ) -> tuple:
        kept = []
        removed = []
        fired = []

        for s in spans:
            should_remove = (
                s.uncertainty >= self.uncertainty_threshold
                and s.confidence < max(self.confidence_threshold, 0.5)
            )
            if should_remove:
                removed.append({
                    "rule": "uncertainty_filter",
                    "token": s.token,
                    "uncertainty": round(s.uncertainty, 4),
                    "threshold": self.uncertainty_threshold,
                })
                fired.append(
                    f"uncertainty_filter: removed '{s.token}' "
                    f"(unc={s.uncertainty:.3f} >= {self.uncertainty_threshold})"
                )
            else:
                kept.append(s)

        return kept, removed, fired

    # -----------------------------------------------------------------------
    # Operation 3: Minimum duration filtering
    # -----------------------------------------------------------------------

    def _filter_min_duration(
        self, spans: List[TokenSpan]
    ) -> tuple:
        kept = []
        removed = []
        fired = []

        for s in spans:
            if s.frame_count < self.min_token_duration:
                removed.append({
                    "rule": "min_duration_filter",
                    "token": s.token,
                    "frames": s.frame_count,
                    "min_required": self.min_token_duration,
                })
                fired.append(
                    f"min_duration_filter: removed '{s.token}' "
                    f"({s.frame_count} frames < {self.min_token_duration})"
                )
            else:
                kept.append(s)

        return kept, removed, fired

    # -----------------------------------------------------------------------
    # Operation 4: Transition smoothing
    # -----------------------------------------------------------------------

    def _smooth_transitions(
        self, spans: List[TokenSpan]
    ) -> tuple:
        """
        Remove short noisy insertions between identical surrounding tokens.

        Example: [hello(long), food(1 frame), hello(long)] -> [hello]
        """
        if len(spans) < 3:
            return spans, [], []

        kept = []
        removed = []
        fired = []
        skip_indices = set()

        for i in range(1, len(spans) - 1):
            if i in skip_indices:
                continue
            prev_span = spans[i - 1] if i - 1 not in skip_indices else None
            curr_span = spans[i]
            next_span = spans[i + 1]

            if prev_span is None:
                continue

            # Check if current is a short insertion between identical tokens
            if (prev_span.token_id == next_span.token_id
                    and curr_span.frame_count <= self.transition_max_frames
                    and curr_span.uncertainty >= self.uncertainty_threshold * 0.75
                    and curr_span.token_id != prev_span.token_id):
                removed.append({
                    "rule": "transition_smoothing",
                    "token": curr_span.token,
                    "frames": curr_span.frame_count,
                    "surrounding": prev_span.token,
                })
                fired.append(
                    f"transition_smoothing: removed '{curr_span.token}' "
                    f"({curr_span.frame_count} frames) between "
                    f"'{prev_span.token}' spans"
                )
                skip_indices.add(i)

        for i, s in enumerate(spans):
            if i not in skip_indices:
                kept.append(s)

        return kept, removed, fired

    # -----------------------------------------------------------------------
    # Operation 5: Motion-aware suppression
    # -----------------------------------------------------------------------

    def _motion_suppress(
        self,
        spans: List[TokenSpan],
        velocity_magnitudes: List[float],
    ) -> tuple:
        """
        Suppress short, low-confidence tokens that occur during high-motion
        transition regions. High motion + low confidence + short duration
        suggests a spurious detection during hand movement.
        """
        kept = []
        removed = []
        fired = []

        for s in spans:
            # Only consider short spans
            if s.frame_count > self.min_token_duration * 2:
                kept.append(s)
                continue

            # Check if the span overlaps with high-motion region
            start = max(0, s.start_frame)
            end = min(len(velocity_magnitudes), s.end_frame)
            if start >= end:
                kept.append(s)
                continue

            span_vels = velocity_magnitudes[start:end]
            mean_vel = sum(span_vels) / max(len(span_vels), 1)

            if (
                mean_vel > self.motion_velocity_threshold
                and s.confidence < self.confidence_threshold * 1.5
                and s.uncertainty >= self.uncertainty_threshold * 0.75
            ):
                removed.append({
                    "rule": "motion_suppression",
                    "token": s.token,
                    "mean_velocity": round(mean_vel, 4),
                    "confidence": round(s.confidence, 4),
                    "uncertainty": round(s.uncertainty, 4),
                })
                fired.append(
                    f"motion_suppression: removed '{s.token}' "
                    f"(vel={mean_vel:.3f}, conf={s.confidence:.3f}, "
                    f"unc={s.uncertainty:.3f})"
                )
            else:
                kept.append(s)

        return kept, removed, fired

    # -----------------------------------------------------------------------
    # Operation 6: Vocabulary consistency
    # -----------------------------------------------------------------------

    def _check_vocabulary(
        self, spans: List[TokenSpan]
    ) -> tuple:
        """Verify all tokens exist in the vocabulary."""
        kept = []
        removed = []
        fired = []

        for s in spans:
            if s.token not in self.vocabulary and not s.token.startswith("<"):
                removed.append({
                    "rule": "vocabulary_check",
                    "token": s.token,
                })
                fired.append(f"vocabulary_check: removed '{s.token}' (not in vocab)")
            else:
                kept.append(s)

        return kept, removed, fired


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    print("=" * 60)
    print("inference/refinement.py -- verification")
    print("=" * 60)

    # Build test spans simulating noisy CTC output
    test_spans = [
        TokenSpan("hello", 3, 0, 15, 0.85),      # strong
        TokenSpan("the", 5, 15, 17, 0.15),         # low confidence, short
        TokenSpan("hello", 3, 17, 30, 0.80),       # repeat of hello
        TokenSpan("world", 7, 30, 45, 0.90),       # strong
        TokenSpan("x", 9, 45, 46, 0.10),           # very short, low conf
    ]

    # Create a fake DecoderOutput
    decoder_out = DecoderOutput(
        tokens=[s.token for s in test_spans],
        token_ids=[s.token_id for s in test_spans],
        sequence_confidence=0.5,
        token_confidences=[s.confidence for s in test_spans],
        spans=test_spans,
        raw_frame_predictions=list(range(50)),
        raw_frame_confidences=[0.5] * 50,
    )

    # Create refiner
    refiner = TokenRefiner(
        confidence_threshold=0.3,
        min_token_duration=3,
        transition_max_frames=2,
        motion_suppression_enabled=False,
    )

    result = refiner.refine(decoder_out)

    print(f"  Input tokens:  {[s.token for s in test_spans]}")
    print(f"  Output tokens: {result.tokens}")
    print(f"  Display text:  '{result.display_text}'")
    print(f"  Rules fired:   {len(result.rules_fired)}")
    for rule in result.rules_fired:
        print(f"    - {rule}")

    # Verify: 'the' removed (low conf), 'x' removed (short + low conf),
    # two 'hello' merged
    assert "hello" in result.tokens, "hello should survive"
    assert "world" in result.tokens, "world should survive"
    assert "the" not in result.tokens, "'the' should be filtered (low conf)"
    assert "x" not in result.tokens, "'x' should be filtered (short + low conf)"
    assert result.tokens.count("hello") == 1, "two hellos should merge"

    print(f"\n  Expected: 'hello world' -> got '{result.display_text}'")
    assert result.display_text == "hello world"

    print("=" * 60)
    print("[PASS] inference/refinement.py OK")
