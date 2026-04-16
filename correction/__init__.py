"""
correction -- Sprint 4: Post-correction, bilingual output, and benchmark.

This is a GLOSS-FREE post-correction layer. It operates on English token
sequences produced by the Sprint 3 CTC pipeline.

Modules:
    config              : Correction pipeline configuration
    english_corrector   : English-token cleanup rules (adapted from corrector.py)
    pipeline            : Unified post-correction pipeline
    benchmark           : Full iSign benchmark with BLEU-4, chrF, WER
    utils               : Metric computation helpers

Quick Start:
    from correction.pipeline import CorrectionPipeline
    pipeline = CorrectionPipeline()
    result = pipeline.correct(["i", "food", "eat", "yesterday"])
    print(result.corrected_english)  # "I ate food yesterday."
    print(result.hindi_translation)  # Hindi translation
"""

from correction.config import CorrectionConfig
from correction.english_corrector import EnglishCorrector
from correction.pipeline import CorrectionPipeline, CorrectionResult
from correction.utils import compute_bleu, compute_chrf, compute_wer

__all__ = [
    "CorrectionConfig",
    "EnglishCorrector",
    "CorrectionPipeline",
    "CorrectionResult",
    "compute_bleu",
    "compute_chrf",
    "compute_wer",
]
