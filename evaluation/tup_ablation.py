"""
evaluation/tup_ablation.py -- Sprint 5 ablation harness.

Provides a lightweight experiment manifest for the new methodological
components:
    - velocity-aware gate temperature
    - Temporal Uncertainty Propagation (TUP)
    - fallback routing
    - uncertainty-aware refinement

This harness is intentionally lightweight: it can either export a dry-run
experiment plan or run small evaluation sweeps once trained checkpoints for
each variant are available.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AblationVariant:
    name: str
    enable_velocity_temperature: bool
    enable_tup: bool
    enable_fallback: bool
    uncertainty_aware_refinement: bool
    checkpoint_path: Optional[str] = None


def default_variants() -> List[AblationVariant]:
    return [
        AblationVariant(
            name="baseline",
            enable_velocity_temperature=False,
            enable_tup=False,
            enable_fallback=False,
            uncertainty_aware_refinement=False,
        ),
        AblationVariant(
            name="velocity_only",
            enable_velocity_temperature=True,
            enable_tup=False,
            enable_fallback=False,
            uncertainty_aware_refinement=False,
        ),
        AblationVariant(
            name="tup_only",
            enable_velocity_temperature=False,
            enable_tup=True,
            enable_fallback=False,
            uncertainty_aware_refinement=True,
        ),
        AblationVariant(
            name="velocity_plus_tup",
            enable_velocity_temperature=True,
            enable_tup=True,
            enable_fallback=False,
            uncertainty_aware_refinement=True,
        ),
        AblationVariant(
            name="velocity_tup_fallback",
            enable_velocity_temperature=True,
            enable_tup=True,
            enable_fallback=True,
            uncertainty_aware_refinement=True,
        ),
        AblationVariant(
            name="velocity_tup_no_unc_refine",
            enable_velocity_temperature=True,
            enable_tup=True,
            enable_fallback=True,
            uncertainty_aware_refinement=False,
        ),
    ]


def build_manifest(results_dir: str) -> Dict:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = default_variants()
    return {
        "results_dir": str(output_dir),
        "variants": [asdict(variant) for variant in variants],
        "notes": [
            "Train a dedicated checkpoint per variant for fair comparison.",
            "Evaluate sequence WER / BLEU and insertion errors near high-motion spans.",
            "Use the same data split and seed across all variants.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 5 TUP ablation harness")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/tup_ablation",
        help="Directory to store the ablation manifest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit JSON output path",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.results_dir)
    output_path = Path(args.output) if args.output else Path(args.results_dir) / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("=" * 60)
    print("Sprint 5 TUP Ablation Manifest")
    print("=" * 60)
    for variant in manifest["variants"]:
        print(
            f"  {variant['name']:<24} "
            f"vel={variant['enable_velocity_temperature']} "
            f"tup={variant['enable_tup']} "
            f"fallback={variant['enable_fallback']} "
            f"unc_refine={variant['uncertainty_aware_refinement']}"
        )
    print(f"\nManifest saved to {output_path}")


if __name__ == "__main__":
    main()
