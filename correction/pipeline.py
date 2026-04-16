"""
correction/pipeline.py -- Unified Post-Correction Pipeline for gloss-free ISL.

End-to-end pipeline:
    English tokens → detokenize → rule-based cleanup → optional KenLM reranking
    → final English → Hindi translation → bilingual output

All timing uses time.perf_counter().

Usage:
    from correction.pipeline import CorrectionPipeline
    pipeline = CorrectionPipeline()
    result = pipeline.correct(["i", "food", "eat", "yesterday"])
    print(result.corrected_english)
    print(result.hindi_translation)
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from correction.config import CorrectionConfig
from correction.english_corrector import EnglishCorrector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Translation backend
# ============================================================================

class TranslationEngine:
    """English-to-Hindi translation using argostranslate."""

    def __init__(self):
        self.available = False
        self._translate_fn = None
        self._setup()

    def _setup(self):
        """Try to load argostranslate."""
        try:
            import argostranslate.translate
            installed = argostranslate.translate.get_installed_languages()
            en_lang = next((l for l in installed if l.code == "en"), None)
            hi_lang = next((l for l in installed if l.code == "hi"), None)
            if en_lang and hi_lang:
                self._translate_fn = en_lang.get_translation(hi_lang)
                if self._translate_fn:
                    self.available = True
                    logger.info("Hindi translation engine loaded (argostranslate)")
                else:
                    logger.warning(
                        "English→Hindi translation not installed. "
                        "Run: python scripts/setup_translation_models.py"
                    )
            else:
                logger.warning(
                    "argostranslate languages not found. "
                    "Run: python scripts/setup_translation_models.py"
                )
        except ImportError:
            logger.warning(
                "argostranslate not installed. "
                "Install with: pip install argostranslate\n"
                "Then run: python scripts/setup_translation_models.py"
            )

    def translate(self, text: str) -> Optional[str]:
        """Translate English text to Hindi."""
        if not self.available or not self._translate_fn:
            return None
        try:
            return self._translate_fn.translate(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return None


# ============================================================================
# KenLM integration
# ============================================================================

class KenLMReranker:
    """Optional KenLM-based reranking of correction candidates."""

    def __init__(self, model_path: Optional[str] = None):
        self.available = False
        self.model = None

        if model_path is None:
            return

        try:
            import kenlm
            model_file = Path(model_path)
            if model_file.exists():
                self.model = kenlm.Model(str(model_file))
                self.available = True
                logger.info(f"KenLM loaded: {model_path} (order={self.model.order})")
            else:
                logger.warning(
                    f"KenLM model not found at {model_path}. "
                    "Run: python scripts/setup_kenlm.py"
                )
        except ImportError:
            logger.warning(
                "kenlm not installed. Install with: pip install kenlm\n"
                "Then run: python scripts/setup_kenlm.py"
            )

    def score(self, sentence: str) -> float:
        """Score a sentence (log probability). Higher = more likely."""
        if not self.available:
            return 0.0
        clean = sentence.strip().rstrip(".?!").strip()
        return self.model.score(clean, bos=True, eos=True)

    def rerank(self, candidates: List[str]) -> str:
        """Return the highest-scoring candidate."""
        if not self.available or not candidates:
            return candidates[0] if candidates else ""
        scored = [(c, self.score(c)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]


# ============================================================================
# Correction Result
# ============================================================================

@dataclass
class CorrectionResult:
    """Structured result from the correction pipeline."""
    raw_tokens: List[str]
    raw_english: str
    corrected_english: str
    hindi_translation: Optional[str]
    rules_applied: List[str]
    kenlm_used: bool
    timing_ms: Dict[str, float]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "raw_tokens": self.raw_tokens,
            "raw_english": self.raw_english,
            "corrected_english": self.corrected_english,
            "hindi_translation": self.hindi_translation,
            "rules_applied": self.rules_applied,
            "kenlm_used": self.kenlm_used,
            "timing_ms": {k: round(v, 3) for k, v in self.timing_ms.items()},
            "metadata": self.metadata,
        }


# ============================================================================
# Main Pipeline
# ============================================================================

class CorrectionPipeline:
    """
    Unified post-correction pipeline for the gloss-free ISL system.

    Steps:
        1. Detokenize / join raw English tokens
        2. Apply rule-based English cleanup
        3. Optionally rerank with KenLM
        4. Translate to Hindi
    """

    def __init__(self, config: Optional[CorrectionConfig] = None):
        """
        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or CorrectionConfig()

        # English corrector
        self.corrector = EnglishCorrector(
            disabled_rules=self.config.disabled_rules,
        )

        # KenLM reranker
        self.reranker = KenLMReranker(self.config.kenlm_model_path)

        # Hindi translation
        self._translator = None
        if self.config.enable_hindi_translation:
            self._translator = TranslationEngine()

        logger.info(
            f"CorrectionPipeline initialized "
            f"(KenLM={'yes' if self.reranker.available else 'no'}, "
            f"Hindi={'yes' if self._translator and self._translator.available else 'no'})"
        )

    def correct(
        self,
        tokens: List[str],
        token_confidences: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> CorrectionResult:
        """
        Run the full correction pipeline on English tokens.

        Args:
            tokens:             Cleaned English tokens from Sprint 3.
            token_confidences:  Optional per-token confidence scores.
            metadata:           Optional metadata for logging.

        Returns:
            CorrectionResult with corrected English, Hindi translation,
            applied rules, and per-step timing.
        """
        timing = {}
        total_start = time.perf_counter()

        # Step 1: Detokenize / join
        t0 = time.perf_counter()
        raw_english = " ".join(tokens) if tokens else ""
        timing["detokenize_ms"] = (time.perf_counter() - t0) * 1000

        # Step 2: Rule-based English correction
        t0 = time.perf_counter()
        corrected = self.corrector.correct(tokens)
        rules_applied = [r.rule_id for r in self.corrector.get_applied_rules()]
        timing["rule_correction_ms"] = (time.perf_counter() - t0) * 1000

        # Step 3: Optional KenLM reranking
        kenlm_used = False
        if self.reranker.available:
            t0 = time.perf_counter()
            # Generate candidates from the corrector's candidate system
            candidates = self._generate_candidates(corrected)
            if candidates:
                corrected = self.reranker.rerank([corrected] + candidates)
                kenlm_used = True
            timing["kenlm_rerank_ms"] = (time.perf_counter() - t0) * 1000
        else:
            timing["kenlm_rerank_ms"] = 0.0

        # Step 4: Hindi translation
        hindi = None
        if self._translator and self._translator.available:
            t0 = time.perf_counter()
            hindi = self._translator.translate(corrected)
            timing["hindi_translation_ms"] = (time.perf_counter() - t0) * 1000
        else:
            timing["hindi_translation_ms"] = 0.0

        timing["total_correction_ms"] = (time.perf_counter() - total_start) * 1000

        return CorrectionResult(
            raw_tokens=list(tokens),
            raw_english=raw_english,
            corrected_english=corrected,
            hindi_translation=hindi,
            rules_applied=rules_applied,
            kenlm_used=kenlm_used,
            timing_ms=timing,
            metadata=metadata or {},
        )

    def correct_rules_only(self, tokens: List[str]) -> str:
        """Run only rule-based correction (no KenLM, no translation)."""
        return self.corrector.correct(tokens)

    def correct_no_rules(self, tokens: List[str]) -> str:
        """Detokenize only (no rules, no KenLM) — for ablation baseline."""
        if not tokens:
            return ""
        # Basic join, capitalize first letter, add period
        text = " ".join(t.lower() for t in tokens if t.strip())
        if text:
            text = text[0].upper() + text[1:]
            if not text.endswith((".", "?", "!")):
                text += "."
        return text

    def _generate_candidates(self, sentence: str) -> List[str]:
        """Generate alternative phrasings for KenLM reranking."""
        candidates = []
        lower = sentence.lower()

        # Contraction variants
        contractions = {
            "do not": "don't", "does not": "doesn't",
            "did not": "didn't", "will not": "won't",
            "can not": "can't", "is not": "isn't",
            "are not": "aren't", "was not": "wasn't",
        }
        for full, short in contractions.items():
            if full in lower:
                candidates.append(
                    sentence[:lower.find(full)] + short +
                    sentence[lower.find(full) + len(full):]
                )

        return candidates

    def translate_to_hindi(self, english_text: str) -> Optional[str]:
        """Standalone Hindi translation."""
        if self._translator and self._translator.available:
            return self._translator.translate(english_text)
        return None


# ============================================================================
# Standalone verification
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("correction/pipeline.py -- verification")
    print("=" * 70)

    pipeline = CorrectionPipeline()

    test_cases = [
        ["i", "food", "eat", "yesterday"],
        ["she", "happy"],
        ["he", "book", "read"],
        ["i", "go", "tomorrow"],
        ["you", "name", "what"],
    ]

    for tokens in test_cases:
        result = pipeline.correct(tokens)
        print(f"\n  Input:     {' '.join(tokens)}")
        print(f"  Raw:       {result.raw_english}")
        print(f"  Corrected: {result.corrected_english}")
        if result.hindi_translation:
            print(f"  Hindi:     {result.hindi_translation}")
        print(f"  Rules:     {result.rules_applied}")
        print(f"  KenLM:     {result.kenlm_used}")
        print(f"  Timing:    {result.timing_ms}")

    # Test no-rules baseline
    baseline = pipeline.correct_no_rules(["i", "food", "eat", "yesterday"])
    print(f"\n  No-rules baseline: {baseline}")

    print("\n" + "=" * 70)
    print("[PASS] correction/pipeline.py OK")
