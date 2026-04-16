"""
correction/config.py -- Configuration for the Sprint 4 post-correction pipeline.

Centralizes all tunable knobs for:
    - English cleanup rules
    - KenLM rescoring
    - Hindi translation (argostranslate)
    - TTS output
    - Benchmark settings

Usage:
    from correction.config import CorrectionConfig
    cfg = CorrectionConfig()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set


@dataclass
class CorrectionConfig:
    """Configuration for the gloss-free post-correction pipeline."""

    # ------------------------------------------------------------------
    # English Correction Rules
    # ------------------------------------------------------------------
    enable_pronoun_normalization: bool = True
    """R1: Normalize pronoun forms (me -> I, him -> he)."""

    enable_capitalization: bool = True
    """R8: Capitalize first word, add punctuation."""

    enable_verb_conjugation: bool = True
    """R6: Conjugate verbs based on tense markers."""

    enable_article_insertion: bool = True
    """NEW: Insert articles (a, an, the) where appropriate."""

    enable_contraction_expansion: bool = True
    """NEW: Handle common contractions (don't, can't, etc.)."""

    enable_repeated_word_removal: bool = True
    """NEW: Remove consecutive duplicate words."""

    enable_filler_removal: bool = True
    """NEW: Remove filler/noise tokens."""

    # ISL-specific rules — DISABLED by default for gloss-free pipeline
    enable_wh_fronting: bool = False
    """R2: WH-word fronting. DISABLED: assumes ISL gloss order."""

    enable_negation_reorder: bool = False
    """R3: Negation reordering. DISABLED: assumes ISL gloss order."""

    enable_sov_to_svo: bool = False
    """R4: SOV→SVO. DISABLED: assumes ISL gloss order."""

    enable_adjective_reorder: bool = False
    """R5: Adj-Noun swap. DISABLED: assumes ISL gloss order."""

    disabled_rules: Set[str] = field(default_factory=set)
    """Set of rule IDs to explicitly disable (for ablation study)."""

    # ------------------------------------------------------------------
    # KenLM
    # ------------------------------------------------------------------
    kenlm_model_path: Optional[str] = None
    """Path to KenLM .arpa or .bin model file. None = skip LM rescoring."""

    kenlm_weight: float = 1.0
    """Weight for KenLM score in candidate selection."""

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    enable_hindi_translation: bool = True
    """Enable English-to-Hindi translation."""

    translation_backend: str = "argostranslate"
    """Translation backend: 'argostranslate' or 'none'."""

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    enable_tts: bool = True
    """Enable text-to-speech output."""

    tts_engine: str = "pyttsx3"
    """TTS engine: 'pyttsx3' (offline) or 'gtts' (online)."""

    tts_english_rate: int = 150
    """Words per minute for English TTS."""

    tts_hindi_rate: int = 130
    """Words per minute for Hindi TTS."""

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    data_dir: str = "./data_iSign"
    """Root directory for iSign data."""

    csv_path: str = "./data_iSign/iSign_v1.1.csv"
    """Path to iSign CSV with references."""

    pose_dir: str = "./data_iSign/poses"
    """Directory containing pre-extracted .npy pose files."""

    results_dir: str = "./results"
    """Directory to save benchmark results."""

    checkpoint_path: str = "./checkpoints/best_model.pth"
    """Path to trained model checkpoint."""

    vocab_path: str = "./checkpoints/vocab.json"
    """Path to vocabulary JSON."""

    norm_stats_path: str = "./data_iSign/norm_stats.npz"
    """Path to normalization statistics."""

    device: str = "cpu"
    """Inference device."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create required directories."""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config to dict for reproducibility."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, set):
                d[k] = sorted(list(v))
            elif isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = CorrectionConfig()
    print("=" * 60)
    print("correction/config.py -- verification")
    print("=" * 60)
    for k, v in cfg.to_dict().items():
        print(f"  {k:<35} = {v}")
    print("=" * 60)
    print("[PASS] correction/config.py OK")
