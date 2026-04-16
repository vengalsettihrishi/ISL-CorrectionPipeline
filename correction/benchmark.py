"""
correction/benchmark.py — Full gloss-free iSign benchmark with ablation study.

Runs the complete pipeline on the iSign test set:
    1. Load pre-extracted poses / features
    2. Run model → decode → refinement → English tokens
    3. Run English post-correction pipeline
    4. Compare corrected English against iSign English references
    5. Compute BLEU-4, chrF, WER, exact match
    6. Run ablation study (no-correction, rules-only, rules+KenLM, per-rule)

Outputs:
    results/benchmark_results.json   — overall metrics + ablation results
    results/per_sentence_results.csv — per-sentence breakdown

Usage:
    # Full pipeline benchmark (requires model checkpoint + data)
    python -m correction.benchmark --data_dir ./data_iSign --checkpoint ./checkpoints/best_model.pth

    # Correction-only benchmark (test correction on pre-decoded tokens)
    python -m correction.benchmark --correction-only

    # Correction-only with custom CSV
    python -m correction.benchmark --correction-only --csv ./data_iSign/iSign_v1.1.csv
"""

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from correction.config import CorrectionConfig
from correction.english_corrector import EnglishCorrector
from correction.pipeline import CorrectionPipeline, CorrectionResult
from correction.utils import (
    compute_bleu, compute_chrf, compute_wer,
    compute_exact_match, compute_aggregate_metrics,
    compute_bleu_corpus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark result types
# ============================================================================

@dataclass
class SentenceResult:
    """Result for a single test sentence."""
    sample_id: str
    reference: str
    raw_tokens: List[str]
    raw_english: str
    corrected_english: str
    hindi_translation: Optional[str]
    rules_applied: List[str]
    kenlm_used: bool
    bleu4: float
    chrf: float
    wer: float
    exact_match: bool
    timing_ms: Dict[str, float]


@dataclass
class AblationResult:
    """Result of a single ablation configuration."""
    name: str
    description: str
    disabled_rules: List[str]
    kenlm_enabled: bool
    metrics: Dict[str, float]
    num_samples: int


@dataclass
class BenchmarkResults:
    """Complete benchmark output."""
    timestamp: str
    config: dict
    pipeline_type: str  # "gloss_free"
    label_space: str    # "english"
    num_samples: int
    overall_metrics: Dict[str, float]
    ablation_results: List[dict]
    per_sentence_summary: List[dict]
    timing_summary: Dict[str, float]
    notes: List[str]


# ============================================================================
# Benchmark Engine
# ============================================================================

class BenchmarkEngine:
    """
    Full gloss-free iSign benchmark engine.

    Supports two modes:
        1. Full pipeline: model → decode → refine → correct → evaluate
        2. Correction-only: pre-decoded tokens → correct → evaluate
    """

    def __init__(self, config: Optional[CorrectionConfig] = None):
        self.config = config or CorrectionConfig()
        self.config.ensure_dirs()

    # ------------------------------------------------------------------
    # Full pipeline benchmark (requires model + data)
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        data_dir: str,
        checkpoint_path: str,
        vocab_path: str,
        norm_stats_path: str,
        csv_path: str,
        device: str = "cpu",
        max_samples: Optional[int] = None,
    ) -> BenchmarkResults:
        """
        Run the full end-to-end benchmark: model → decode → correct → evaluate.

        Args:
            data_dir:        Root data directory with pose files.
            checkpoint_path: Model checkpoint path.
            vocab_path:      Vocabulary JSON path.
            norm_stats_path: Normalization stats path.
            csv_path:        iSign CSV with references.
            device:          Inference device.
            max_samples:     Limit number of samples (None = all).

        Returns:
            BenchmarkResults with complete metrics.
        """
        import numpy as np
        import torch

        total_start = time.perf_counter()

        # Load model bundle
        from inference.model_loader import load_model_bundle
        from inference.ctc_decoder import GreedyDecoder
        from inference.refinement import TokenRefiner
        from inference.video_inference import (
            load_pose_file, build_features_from_landmarks,
        )

        logger.info("Loading model bundle...")
        bundle = load_model_bundle(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            norm_stats_path=norm_stats_path,
            device=device,
        )
        logger.info(f"Model loaded: {bundle.summary()}")

        # Setup decoder and refiner
        decoder = GreedyDecoder(blank_id=0, id2word=bundle.id2word)
        refiner = TokenRefiner(
            confidence_threshold=0.3,
            min_token_duration=3,
            vocabulary=set(bundle.id2word.values()),
        )

        # Setup correction pipeline
        pipeline = CorrectionPipeline(self.config)

        # Load test data references
        records = self._load_csv_references(csv_path)
        if max_samples:
            records = records[:max_samples]

        logger.info(f"Benchmarking {len(records)} samples...")

        # Process each sample
        sentence_results = []
        hypotheses = []
        references = []
        all_timings = []

        pose_dir = Path(data_dir) / "poses"

        for i, record in enumerate(records):
            sample_id = record["video_id"]
            reference = record["text"]

            # Find pose file
            pose_path = pose_dir / f"{sample_id}.npy"
            if not pose_path.exists():
                # Try alternative naming
                pose_path = pose_dir / f"{sample_id}_poses.npy"
            if not pose_path.exists():
                logger.warning(f"Pose file not found for {sample_id}, skipping")
                continue

            # Load and process
            t_start = time.perf_counter()

            landmarks = load_pose_file(str(pose_path))
            if landmarks is None:
                continue

            features, vel_mags = build_features_from_landmarks(
                landmarks, bundle.mean, bundle.std
            )

            # Model forward
            log_probs = bundle.predict(features)

            # Decode
            decoder_output = decoder.decode(log_probs)

            # Refine
            refined = refiner.refine(decoder_output, vel_mags)

            # Correct
            correction_result = pipeline.correct(
                refined.tokens,
                metadata={"sample_id": sample_id},
            )

            t_total = (time.perf_counter() - t_start) * 1000

            # Evaluate
            corrected = correction_result.corrected_english
            bleu = compute_bleu(corrected, reference)
            chrf_score = compute_chrf(corrected, reference)
            wer_score = compute_wer(corrected, reference)
            exact = compute_exact_match(corrected, reference)

            sentence_results.append(SentenceResult(
                sample_id=sample_id,
                reference=reference,
                raw_tokens=refined.tokens,
                raw_english=correction_result.raw_english,
                corrected_english=corrected,
                hindi_translation=correction_result.hindi_translation,
                rules_applied=correction_result.rules_applied,
                kenlm_used=correction_result.kenlm_used,
                bleu4=bleu,
                chrf=chrf_score,
                wer=wer_score,
                exact_match=exact,
                timing_ms={**correction_result.timing_ms, "total_pipeline_ms": t_total},
            ))

            hypotheses.append(corrected)
            references.append(reference)
            all_timings.append(t_total)

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(records)} samples")

        # Compute aggregate metrics
        overall = compute_aggregate_metrics(hypotheses, references)

        # Timing summary
        timing_summary = {}
        if all_timings:
            timing_summary = {
                "mean_total_ms": sum(all_timings) / len(all_timings),
                "min_total_ms": min(all_timings),
                "max_total_ms": max(all_timings),
                "total_benchmark_s": (time.perf_counter() - total_start),
            }

        # Run ablations
        logger.info("Running ablation study...")
        ablation_results = self._run_ablations(
            [(r.raw_tokens, r.reference) for r in sentence_results]
        )

        # Build final results
        import datetime
        results = BenchmarkResults(
            timestamp=datetime.datetime.now().isoformat(),
            config=self.config.to_dict(),
            pipeline_type="gloss_free",
            label_space="english",
            num_samples=len(sentence_results),
            overall_metrics=overall,
            ablation_results=[asdict(a) for a in ablation_results],
            per_sentence_summary=[asdict(s) for s in sentence_results],
            timing_summary=timing_summary,
            notes=[
                "Pipeline: model → CTC decode → token refinement → English correction → evaluation",
                "Metrics computed against English reference sentences from iSign dataset",
                "This is a gloss-free pipeline; metrics are in English space",
            ],
        )

        # Save results
        self._save_results(results, sentence_results)

        return results

    # ------------------------------------------------------------------
    # Correction-only benchmark (no model needed)
    # ------------------------------------------------------------------

    def run_correction_benchmark(
        self,
        test_pairs: Optional[List[Tuple[List[str], str]]] = None,
        csv_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> BenchmarkResults:
        """
        Run correction-only benchmark on pre-decoded English tokens.

        This tests the correction pipeline independently from the recognizer.

        Args:
            test_pairs:  List of (tokens, reference) tuples.
            csv_path:    CSV file to generate synthetic test pairs from.
            max_samples: Limit number of samples.

        Returns:
            BenchmarkResults with correction-only metrics.
        """
        total_start = time.perf_counter()

        # Generate test data if not provided
        if test_pairs is None:
            if csv_path and Path(csv_path).exists():
                test_pairs = self._generate_test_pairs_from_csv(csv_path)
            else:
                test_pairs = self._generate_synthetic_test_pairs()

        if max_samples:
            test_pairs = test_pairs[:max_samples]

        logger.info(f"Correction-only benchmark: {len(test_pairs)} samples")

        # Setup pipeline
        pipeline = CorrectionPipeline(self.config)

        # Process each sample
        sentence_results = []
        hypotheses = []
        references = []
        all_timings = []

        for i, (tokens, reference) in enumerate(test_pairs):
            t_start = time.perf_counter()

            result = pipeline.correct(
                tokens,
                metadata={"sample_id": f"sample_{i}"},
            )

            t_total = (time.perf_counter() - t_start) * 1000

            corrected = result.corrected_english
            bleu = compute_bleu(corrected, reference)
            chrf_score = compute_chrf(corrected, reference)
            wer_score = compute_wer(corrected, reference)
            exact = compute_exact_match(corrected, reference)

            sentence_results.append(SentenceResult(
                sample_id=f"sample_{i}",
                reference=reference,
                raw_tokens=tokens,
                raw_english=result.raw_english,
                corrected_english=corrected,
                hindi_translation=result.hindi_translation,
                rules_applied=result.rules_applied,
                kenlm_used=result.kenlm_used,
                bleu4=bleu,
                chrf=chrf_score,
                wer=wer_score,
                exact_match=exact,
                timing_ms={**result.timing_ms, "total_pipeline_ms": t_total},
            ))

            hypotheses.append(corrected)
            references.append(reference)
            all_timings.append(t_total)

        # Aggregate metrics
        overall = compute_aggregate_metrics(hypotheses, references)

        # Timing summary
        timing_summary = {}
        if all_timings:
            timing_summary = {
                "mean_correction_ms": sum(all_timings) / len(all_timings),
                "min_correction_ms": min(all_timings),
                "max_correction_ms": max(all_timings),
                "total_benchmark_s": time.perf_counter() - total_start,
            }

        # Run ablations
        logger.info("Running ablation study...")
        ablation_results = self._run_ablations(test_pairs)

        # Build results
        import datetime
        results = BenchmarkResults(
            timestamp=datetime.datetime.now().isoformat(),
            config=self.config.to_dict(),
            pipeline_type="gloss_free",
            label_space="english",
            num_samples=len(sentence_results),
            overall_metrics=overall,
            ablation_results=[asdict(a) for a in ablation_results],
            per_sentence_summary=[asdict(s) for s in sentence_results],
            timing_summary=timing_summary,
            notes=[
                "Correction-only benchmark (no model inference)",
                "Tests English post-correction pipeline independently",
                "Metrics computed against English reference sentences",
            ],
        )

        # Save results
        self._save_results(results, sentence_results)

        return results

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def _run_ablations(
        self,
        test_pairs: List[Tuple[List[str], str]],
    ) -> List[AblationResult]:
        """
        Run ablation configurations:
            A. No post-correction (raw detokenization)
            B. Rules only (no KenLM)
            C. Rules + KenLM (full pipeline)
            D. Per-rule ablation (disable one rule at a time)
        """
        results = []

        # --- A. No post-correction ---
        logger.info("  Ablation A: No post-correction")
        no_correction_cfg = CorrectionConfig(
            disabled_rules={"R1", "R6", "R8", "RE1", "RE2", "RE3", "RE4"},
            kenlm_model_path=None,
            enable_hindi_translation=False,
        )
        no_correction_pipeline = CorrectionPipeline(no_correction_cfg)

        hyps_a = []
        refs_a = []
        for tokens, ref in test_pairs:
            corrected = no_correction_pipeline.correct_no_rules(tokens)
            hyps_a.append(corrected)
            refs_a.append(ref)

        metrics_a = compute_aggregate_metrics(hyps_a, refs_a)
        results.append(AblationResult(
            name="no_correction",
            description="Raw detokenization only — no rules, no KenLM",
            disabled_rules=["R1", "R6", "R8", "RE1", "RE2", "RE3", "RE4"],
            kenlm_enabled=False,
            metrics=metrics_a,
            num_samples=len(test_pairs),
        ))

        # --- B. Rules only ---
        logger.info("  Ablation B: Rules only")
        rules_only_cfg = CorrectionConfig(
            kenlm_model_path=None,
            enable_hindi_translation=False,
        )
        rules_only_pipeline = CorrectionPipeline(rules_only_cfg)

        hyps_b = []
        refs_b = []
        for tokens, ref in test_pairs:
            result = rules_only_pipeline.correct(tokens)
            hyps_b.append(result.corrected_english)
            refs_b.append(ref)

        metrics_b = compute_aggregate_metrics(hyps_b, refs_b)
        results.append(AblationResult(
            name="rules_only",
            description="English cleanup rules only — no KenLM",
            disabled_rules=[],
            kenlm_enabled=False,
            metrics=metrics_b,
            num_samples=len(test_pairs),
        ))

        # --- C. Rules + KenLM ---
        kenlm_path = self.config.kenlm_model_path
        if kenlm_path and Path(kenlm_path).exists():
            logger.info("  Ablation C: Rules + KenLM")
            full_cfg = CorrectionConfig(
                kenlm_model_path=kenlm_path,
                enable_hindi_translation=False,
            )
            full_pipeline = CorrectionPipeline(full_cfg)

            hyps_c = []
            refs_c = []
            for tokens, ref in test_pairs:
                result = full_pipeline.correct(tokens)
                hyps_c.append(result.corrected_english)
                refs_c.append(ref)

            metrics_c = compute_aggregate_metrics(hyps_c, refs_c)
            results.append(AblationResult(
                name="rules_plus_kenlm",
                description="English cleanup rules + KenLM reranking",
                disabled_rules=[],
                kenlm_enabled=True,
                metrics=metrics_c,
                num_samples=len(test_pairs),
            ))
        else:
            logger.info("  Ablation C: Skipped (KenLM model not available)")
            results.append(AblationResult(
                name="rules_plus_kenlm",
                description="SKIPPED — KenLM model not available. Run scripts/setup_kenlm.py",
                disabled_rules=[],
                kenlm_enabled=False,
                metrics={},
                num_samples=0,
            ))

        # --- D. Per-rule ablation ---
        logger.info("  Ablation D: Per-rule ablation")
        english_rules = ["R1", "R6", "R8", "RE1", "RE2", "RE4"]
        rule_names = {
            "R1": "Pronoun Normalization",
            "R6": "Verb Conjugation",
            "R8": "Capitalization & Punctuation",
            "RE1": "Repeated Word Removal",
            "RE2": "Filler/Noise Removal",
            "RE4": "Grammar Smoothing",
        }

        for rule_id in english_rules:
            disable_cfg = CorrectionConfig(
                disabled_rules={rule_id},
                kenlm_model_path=None,
                enable_hindi_translation=False,
            )
            disable_pipeline = CorrectionPipeline(disable_cfg)

            hyps_d = []
            refs_d = []
            for tokens, ref in test_pairs:
                result = disable_pipeline.correct(tokens)
                hyps_d.append(result.corrected_english)
                refs_d.append(ref)

            metrics_d = compute_aggregate_metrics(hyps_d, refs_d)

            # Compute delta from rules-only baseline
            delta_bleu = metrics_d.get("bleu4_sentence_avg", 0) - metrics_b.get("bleu4_sentence_avg", 0)
            delta_wer = metrics_d.get("wer_avg", 0) - metrics_b.get("wer_avg", 0)
            metrics_d["delta_bleu4_from_baseline"] = round(delta_bleu, 6)
            metrics_d["delta_wer_from_baseline"] = round(delta_wer, 6)

            results.append(AblationResult(
                name=f"disable_{rule_id}",
                description=f"Disable {rule_id} ({rule_names.get(rule_id, '')})",
                disabled_rules=[rule_id],
                kenlm_enabled=False,
                metrics=metrics_d,
                num_samples=len(test_pairs),
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_csv_references(self, csv_path: str) -> List[Dict[str, str]]:
        """Load video_id + reference text from iSign CSV."""
        records = []
        path = Path(csv_path)
        if not path.exists():
            logger.error(f"CSV not found: {csv_path}")
            return records

        import csv as csv_module
        with open(path, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            fields = reader.fieldnames or []
            fields_lower = {fn.lower().strip(): fn for fn in fields}

            # Detect columns
            vid_col = None
            for c in ["video_id", "id", "vid", "name"]:
                if c in fields_lower:
                    vid_col = fields_lower[c]
                    break

            eng_col = None
            for c in ["english", "text", "translation", "sentence"]:
                if c in fields_lower:
                    eng_col = fields_lower[c]
                    break

            if not vid_col or not eng_col:
                logger.error(f"CSV missing required columns. Found: {fields}")
                return records

            for row in reader:
                video_id = row[vid_col].strip()
                text = row[eng_col].strip()
                if video_id and text:
                    records.append({"video_id": video_id, "text": text})

        logger.info(f"Loaded {len(records)} references from {csv_path}")
        return records

    def _generate_test_pairs_from_csv(
        self,
        csv_path: str,
    ) -> List[Tuple[List[str], str]]:
        """
        Generate test pairs from CSV by simulating CTC decoder output.

        Simulates by tokenizing the reference and optionally degrading it
        (removing articles, reverting verbs to root forms, etc.) to mimic
        what a CTC decoder might actually output.
        """
        records = self._load_csv_references(csv_path)
        pairs = []

        for record in records:
            reference = record["text"]
            # Simulate CTC output: lowercase, remove punctuation, split
            tokens = reference.lower().strip()
            tokens = tokens.replace(".", "").replace("?", "").replace("!", "")
            tokens = tokens.replace(",", "").replace(";", "")
            token_list = tokens.split()

            if token_list:
                pairs.append((token_list, reference))

        return pairs

    def _generate_synthetic_test_pairs(
        self,
    ) -> List[Tuple[List[str], str]]:
        """Generate synthetic test pairs for correction-only testing."""
        pairs = [
            # (simulated CTC output tokens, reference English)
            (["i", "food", "eat", "yesterday"],
             "I ate food yesterday."),

            (["she", "happy"],
             "She is happy."),

            (["he", "book", "read"],
             "He reads book."),

            (["i", "go", "tomorrow"],
             "I will go tomorrow."),

            (["you", "name", "what"],
             "What is your name?"),

            (["me", "food", "like"],
             "I like food."),

            (["she", "market", "go", "yesterday"],
             "She went to market yesterday."),

            (["i", "understand", "not"],
             "I do not understand."),

            (["he", "book", "big", "buy"],
             "He bought a big book."),

            (["we", "food", "cook", "now"],
             "We are cooking food now."),

            (["they", "school", "go"],
             "They go to school."),

            (["i", "sleep", "yesterday"],
             "I slept yesterday."),

            (["she", "food", "cook"],
             "She cooks food."),

            (["he", "doctor", "not"],
             "He is not a doctor."),

            (["i", "help", "want"],
             "I want help."),

            (["you", "go", "where"],
             "Where are you going?"),

            (["she", "cry", "yesterday"],
             "She cried yesterday."),

            (["we", "play", "tomorrow"],
             "We will play tomorrow."),

            (["he", "food", "eat"],
             "He eats food."),

            (["i", "think", "you", "good"],
             "I think you are good."),
        ]
        return pairs

    def _save_results(
        self,
        results: BenchmarkResults,
        sentence_results: List[SentenceResult],
    ) -> None:
        """Save benchmark results to JSON and CSV."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = results_dir / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {json_path}")

        # CSV
        csv_path = results_dir / "per_sentence_results.csv"
        if sentence_results:
            fieldnames = [
                "sample_id", "reference", "raw_english", "corrected_english",
                "hindi_translation", "bleu4", "chrf", "wer", "exact_match",
                "rules_applied", "kenlm_used",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for sr in sentence_results:
                    writer.writerow({
                        "sample_id": sr.sample_id,
                        "reference": sr.reference,
                        "raw_english": sr.raw_english,
                        "corrected_english": sr.corrected_english,
                        "hindi_translation": sr.hindi_translation or "",
                        "bleu4": round(sr.bleu4, 4),
                        "chrf": round(sr.chrf, 4),
                        "wer": round(sr.wer, 4),
                        "exact_match": sr.exact_match,
                        "rules_applied": ";".join(sr.rules_applied),
                        "kenlm_used": sr.kenlm_used,
                    })
            logger.info(f"Per-sentence results saved to {csv_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ISL Gloss-Free Benchmark (Sprint 4)",
    )
    parser.add_argument(
        "--correction-only", action="store_true",
        help="Run correction-only benchmark (no model inference needed)",
    )
    parser.add_argument("--data_dir", type=str, default="./data_iSign")
    parser.add_argument("--csv", type=str, default="./data_iSign/iSign_v1.1.csv")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--vocab", type=str, default="./checkpoints/vocab.json")
    parser.add_argument("--norm_stats", type=str, default="./data_iSign/norm_stats.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--kenlm_model", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Config
    config = CorrectionConfig(
        kenlm_model_path=args.kenlm_model,
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        norm_stats_path=args.norm_stats,
        device=args.device,
        seed=args.seed,
    )

    engine = BenchmarkEngine(config)

    if args.correction_only:
        # Correction-only mode
        csv_p = args.csv if Path(args.csv).exists() else None
        results = engine.run_correction_benchmark(
            csv_path=csv_p,
            max_samples=args.max_samples,
        )
    else:
        # Full pipeline mode
        results = engine.run_full_benchmark(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            norm_stats_path=args.norm_stats,
            csv_path=args.csv,
            device=args.device,
            max_samples=args.max_samples,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — GLOSS-FREE ISL PIPELINE")
    print("=" * 70)
    print(f"  Samples:          {results.num_samples}")
    print(f"  Pipeline type:    {results.pipeline_type}")
    print(f"  Label space:      {results.label_space}")
    print()

    print("  Overall Metrics:")
    for key, value in results.overall_metrics.items():
        if isinstance(value, float):
            print(f"    {key:<25} {value:.4f}")
        else:
            print(f"    {key:<25} {value}")
    print()

    print("  Ablation Results:")
    print(f"  {'Config':<30} {'BLEU-4':<10} {'chrF':<10} {'WER':<10}")
    print(f"  {'-'*60}")
    for ablation in results.ablation_results:
        m = ablation.get("metrics", {})
        bleu = m.get("bleu4_sentence_avg", "N/A")
        chrf_val = m.get("chrf_avg", "N/A")
        wer_val = m.get("wer_avg", "N/A")
        name = ablation.get("name", "")
        if isinstance(bleu, float):
            print(f"  {name:<30} {bleu:<10.4f} {chrf_val:<10.4f} {wer_val:<10.4f}")
        else:
            print(f"  {name:<30} {'SKIPPED':<10}")
    print()

    if results.timing_summary:
        print("  Timing:")
        for key, value in results.timing_summary.items():
            if isinstance(value, float):
                print(f"    {key:<25} {value:.2f}")
    print()

    print(f"  Results saved to: {config.results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
