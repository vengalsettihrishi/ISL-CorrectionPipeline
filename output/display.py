"""
output/display.py — Visual/terminal display for ISL pipeline output.

Displays:
    - Recognized English tokens with confidence bars
    - Corrected English sentence
    - Hindi translation
    - FPS and per-step latency breakdown
    - Current mode (webcam / video / benchmark)

Works in terminal (ANSI colors) and degrades gracefully without color support.

Usage:
    from output.display import DisplayEngine
    display = DisplayEngine(mode="webcam")
    display.show(tokens, corrected, hindi, confidences, timing)
"""

import os
import sys
import time
from typing import Dict, List, Optional


# ============================================================================
# ANSI color codes (graceful fallback on Windows without color support)
# ============================================================================

def _supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if os.name == "nt":
        # Windows 10+ supports ANSI if ENABLE_VIRTUAL_TERMINAL_PROCESSING
        try:
            os.system("")  # enable ANSI on Windows
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


class _Colors:
    """ANSI color codes."""
    if _COLOR:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        BG_DARK = "\033[48;5;236m"
        UNDERLINE = "\033[4m"
    else:
        RESET = BOLD = DIM = RED = GREEN = YELLOW = ""
        BLUE = MAGENTA = CYAN = WHITE = BG_DARK = UNDERLINE = ""


C = _Colors()


# ============================================================================
# Safe print (handles non-ASCII on Windows cp1252)
# ============================================================================

def _safe_print(*args, **kwargs):
    """Print with fallback for non-ASCII characters on Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode("ascii"), **kwargs)


# ============================================================================
# Confidence bar renderer
# ============================================================================

def _confidence_bar(confidence: float, width: int = 20) -> str:
    """Render an ASCII confidence bar."""
    filled = int(confidence * width)
    empty = width - filled

    if confidence >= 0.8:
        color = C.GREEN
    elif confidence >= 0.5:
        color = C.YELLOW
    else:
        color = C.RED

    bar = color + "#" * filled + C.DIM + "." * empty + C.RESET
    pct = f"{confidence * 100:5.1f}%"
    return f"{bar} {pct}"


# ============================================================================
# Display Engine
# ============================================================================

class DisplayEngine:
    """
    Terminal display engine for ISL pipeline output.

    Provides rich formatted output for live inference,
    video processing, and benchmark results.
    """

    def __init__(
            self,
            mode: str = "webcam",
            show_confidences: bool = True,
            show_timing: bool = True,
            show_hindi: bool = True,
            compact: bool = False,
    ):
        """
        Args:
            mode:             Display mode: "webcam", "video", or "benchmark".
            show_confidences: Show per-token confidence bars.
            show_timing:      Show latency breakdown.
            show_hindi:       Show Hindi translation.
            compact:          Compact single-line mode (for streaming).
        """
        self.mode = mode
        self.show_confidences = show_confidences
        self.show_timing = show_timing
        self.show_hindi = show_hindi
        self.compact = compact
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0

    def show(
        self,
        tokens: List[str],
        corrected_english: str,
        hindi_translation: Optional[str] = None,
        token_confidences: Optional[List[float]] = None,
        timing_ms: Optional[Dict[str, float]] = None,
        fps: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Display a complete recognition result.

        Args:
            tokens:             Recognized English tokens.
            corrected_english:  Post-corrected English sentence.
            hindi_translation:  Hindi translation (if available).
            token_confidences:  Per-token confidence scores.
            timing_ms:          Per-step timing breakdown (ms).
            fps:                Current FPS (for webcam mode).
            metadata:           Additional metadata to display.
        """
        if self.compact:
            self._show_compact(tokens, corrected_english, hindi_translation)
            return

        # Header
        mode_label = {
            "webcam": f"{C.GREEN}* LIVE{C.RESET}",
            "video": f"{C.BLUE}> VIDEO{C.RESET}",
            "benchmark": f"{C.CYAN}+ BENCHMARK{C.RESET}",
        }.get(self.mode, self.mode.upper())

        print(f"\n{C.BOLD}{'-' * 60}{C.RESET}")
        print(f"  {mode_label}  {C.DIM}ISL Recognition Pipeline{C.RESET}")
        print(f"{C.BOLD}{'-' * 60}{C.RESET}")

        # Raw tokens
        if tokens:
            token_str = " ".join(tokens)
            print(f"\n  {C.DIM}Tokens:{C.RESET}     {token_str}")
        else:
            print(f"\n  {C.DIM}Tokens:{C.RESET}     {C.DIM}(no tokens){C.RESET}")

        # Token confidences
        if self.show_confidences and token_confidences and tokens:
            print(f"  {C.DIM}Confidence:{C.RESET}")
            for token, conf in zip(tokens, token_confidences):
                bar = _confidence_bar(conf)
                print(f"    {token:<15} {bar}")

        # Corrected English
        print(f"\n  {C.BOLD}{C.GREEN}English:{C.RESET}    {C.BOLD}{corrected_english}{C.RESET}")

        # Hindi translation
        if self.show_hindi and hindi_translation:
            _safe_print(f"  {C.BOLD}{C.MAGENTA}Hindi:{C.RESET}      {hindi_translation}")
        elif self.show_hindi:
            print(f"  {C.DIM}Hindi:{C.RESET}      {C.DIM}(translation unavailable){C.RESET}")

        # Timing breakdown
        if self.show_timing and timing_ms:
            print(f"\n  {C.DIM}Latency:{C.RESET}")
            for step, ms in timing_ms.items():
                if isinstance(ms, (int, float)):
                    bar_len = min(int(ms / 5), 30)
                    bar = C.CYAN + "|" * bar_len + C.RESET
                    print(f"    {step:<25} {ms:>7.1f} ms  {bar}")

        # FPS
        if fps is not None:
            color = C.GREEN if fps >= 15 else C.YELLOW if fps >= 8 else C.RED
            print(f"\n  {C.DIM}FPS:{C.RESET}        {color}{fps:.1f}{C.RESET}")

        # Metadata
        if metadata:
            print(f"\n  {C.DIM}Metadata:{C.RESET}")
            for key, value in metadata.items():
                print(f"    {key}: {value}")

        print(f"{C.BOLD}{'-' * 60}{C.RESET}")

    def _show_compact(
        self,
        tokens: List[str],
        corrected: str,
        hindi: Optional[str],
    ) -> None:
        """Compact single-line display for streaming mode."""
        parts = [f"{C.GREEN}EN:{C.RESET} {corrected}"]
        if hindi:
            parts.append(f"{C.MAGENTA}HI:{C.RESET} {hindi}")
        _safe_print("  ".join(parts))

    def show_benchmark_progress(
        self,
        current: int,
        total: int,
        sample_id: str,
        corrected: str,
        bleu: float,
        wer: float,
    ) -> None:
        """Show benchmark progress for a single sample."""
        progress = current / max(total, 1)
        bar_width = 30
        filled = int(progress * bar_width)
        bar = "#" * filled + "." * (bar_width - filled)

        bleu_color = C.GREEN if bleu >= 0.5 else C.YELLOW if bleu >= 0.2 else C.RED
        wer_color = C.GREEN if wer <= 0.3 else C.YELLOW if wer <= 0.6 else C.RED

        print(
            f"\r  [{bar}] {current}/{total}  "
            f"BLEU={bleu_color}{bleu:.3f}{C.RESET}  "
            f"WER={wer_color}{wer:.3f}{C.RESET}  "
            f"{C.DIM}{sample_id}{C.RESET}",
            end="", flush=True,
        )

        if current == total:
            print()  # newline at end

    def show_ablation_table(
        self,
        ablation_results: List[dict],
    ) -> None:
        """Display ablation results as a formatted table."""
        print(f"\n{C.BOLD}{'=' * 70}{C.RESET}")
        print(f"  {C.BOLD}ABLATION STUDY RESULTS{C.RESET}")
        print(f"{'=' * 70}")

        header = (
            f"  {C.BOLD}{'Configuration':<30} "
            f"{'BLEU-4':<10} {'chrF':<10} {'WER':<10} {'EM%':<8}{C.RESET}"
        )
        print(header)
        print(f"  {'-' * 68}")

        for result in ablation_results:
            name = result.get("name", "unknown")
            m = result.get("metrics", {})

            if not m:
                print(f"  {name:<30} {'SKIPPED'}")
                continue

            bleu = m.get("bleu4_sentence_avg", 0.0)
            chrf = m.get("chrf_avg", 0.0)
            wer = m.get("wer_avg", 0.0)
            em = m.get("exact_match_rate", 0.0) * 100

            delta_bleu = m.get("delta_bleu4_from_baseline", None)

            line = f"  {name:<30} {bleu:<10.4f} {chrf:<10.4f} {wer:<10.4f} {em:<8.1f}"

            if delta_bleu is not None:
                sign = "+" if delta_bleu >= 0 else ""
                delta_color = C.RED if delta_bleu < -0.01 else C.GREEN if delta_bleu > 0.01 else C.DIM
                line += f"  {delta_color}d{sign}{delta_bleu:.4f}{C.RESET}"

            print(line)

        print(f"{'=' * 70}")

    def show_summary(
        self,
        metrics: Dict[str, float],
        num_samples: int,
        timing: Optional[Dict[str, float]] = None,
    ) -> None:
        """Display final benchmark summary."""
        print(f"\n{C.BOLD}{'=' * 60}{C.RESET}")
        print(f"  {C.BOLD}FINAL BENCHMARK SUMMARY{C.RESET}")
        print(f"{'=' * 60}")
        print(f"  Samples evaluated: {num_samples}")
        print()

        for key, value in metrics.items():
            if isinstance(value, float):
                # Color code metrics
                if "bleu" in key.lower():
                    color = C.GREEN if value >= 0.5 else C.YELLOW
                elif "wer" in key.lower():
                    color = C.GREEN if value <= 0.3 else C.YELLOW
                elif "chrf" in key.lower():
                    color = C.GREEN if value >= 0.5 else C.YELLOW
                else:
                    color = C.RESET
                print(f"  {key:<25} {color}{value:.4f}{C.RESET}")
            else:
                print(f"  {key:<25} {value}")

        if timing:
            print(f"\n  {C.DIM}Timing:{C.RESET}")
            for key, value in timing.items():
                if isinstance(value, float):
                    print(f"    {key:<25} {value:.2f}")

        print(f"{'=' * 60}")

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def update_fps(self) -> float:
        """Calculate and return current FPS."""
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now
        return self._fps


# ============================================================================
# Standalone verification
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("output/display.py -- verification")
    print("=" * 60)

    display = DisplayEngine(mode="video", show_hindi=True)

    # Test normal display
    display.show(
        tokens=["i", "food", "eat", "yesterday"],
        corrected_english="I ate food yesterday.",
        hindi_translation="मैंने कल खाना खाया।",
        token_confidences=[0.95, 0.82, 0.91, 0.88],
        timing_ms={
            "detokenize_ms": 0.1,
            "rule_correction_ms": 2.3,
            "kenlm_rerank_ms": 0.0,
            "hindi_translation_ms": 45.2,
            "total_correction_ms": 47.6,
        },
        fps=24.5,
    )

    # Test compact mode
    compact = DisplayEngine(mode="webcam", compact=True)
    compact.show(
        tokens=["hello", "world"],
        corrected_english="Hello world.",
        hindi_translation="नमस्ते दुनिया।",
    )

    # Test benchmark progress
    bench = DisplayEngine(mode="benchmark")
    for i in range(1, 6):
        bench.show_benchmark_progress(
            current=i, total=5,
            sample_id=f"sample_{i}",
            corrected=f"Test sentence {i}.",
            bleu=0.3 + i * 0.1,
            wer=0.5 - i * 0.08,
        )
        time.sleep(0.1)

    # Test ablation table
    bench.show_ablation_table([
        {"name": "no_correction", "metrics": {"bleu4_sentence_avg": 0.31, "chrf_avg": 0.45, "wer_avg": 0.52, "exact_match_rate": 0.05}},
        {"name": "rules_only", "metrics": {"bleu4_sentence_avg": 0.58, "chrf_avg": 0.72, "wer_avg": 0.28, "exact_match_rate": 0.25}},
        {"name": "rules_plus_kenlm", "metrics": {}},
        {"name": "disable_R6", "metrics": {"bleu4_sentence_avg": 0.42, "chrf_avg": 0.61, "wer_avg": 0.38, "exact_match_rate": 0.10, "delta_bleu4_from_baseline": -0.16}},
    ])

    print("\n" + "=" * 60)
    print("[PASS] output/display.py OK")
