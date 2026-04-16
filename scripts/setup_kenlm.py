"""
scripts/setup_kenlm.py — Download and set up KenLM for the ISL pipeline.

KenLM is used for n-gram language model rescoring of English correction
candidates. This script:
    1. Installs the kenlm Python package
    2. Downloads a pre-trained English n-gram model
    3. Verifies the installation

Usage:
    python scripts/setup_kenlm.py
    python scripts/setup_kenlm.py --model-dir ./models/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def install_kenlm():
    """Install the kenlm Python package."""
    print("=" * 60)
    print("Step 1: Installing kenlm Python package")
    print("=" * 60)

    try:
        import kenlm
        print(f"  kenlm already installed (version available)")
        return True
    except ImportError:
        pass

    print("  Installing kenlm via pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "kenlm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("  kenlm installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  pip install kenlm failed: {e}")
        print()
        print("  KenLM requires C++ compilation tools.")
        print("  On Windows, you may need:")
        print("    - Visual Studio Build Tools")
        print("    - Or: conda install -c conda-forge kenlm")
        print()
        print("  On Linux/macOS:")
        print("    sudo apt-get install build-essential libboost-all-dev")
        print("    pip install kenlm")
        return False


def download_model(model_dir: str = "./models"):
    """
    Download a pre-trained English language model.

    Note: We provide instructions for building your own model,
    since large pre-trained ARPA files may not be freely downloadable.
    """
    print()
    print("=" * 60)
    print("Step 2: Language Model Setup")
    print("=" * 60)

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    arpa_path = model_path / "english.arpa"
    bin_path = model_path / "english.bin"

    if bin_path.exists():
        print(f"  Model already exists: {bin_path}")
        return str(bin_path)

    if arpa_path.exists():
        print(f"  ARPA file found: {arpa_path}")
        print("  Convert to binary for faster loading:")
        print(f"    build_binary {arpa_path} {bin_path}")
        return str(arpa_path)

    # Create a small sample model for testing
    print("  No pre-trained model found.")
    print()
    print("  OPTION A: Build from corpus")
    print("  ---------------------------")
    print("  1. Download an English text corpus (e.g., from OpenSubtitles)")
    print("  2. Install KenLM tools:")
    print("       git clone https://github.com/kpu/kenlm.git")
    print("       cd kenlm && mkdir build && cd build")
    print("       cmake .. && make -j4")
    print("  3. Build the model:")
    print(f"       bin/lmplz -o 3 < corpus.txt > {arpa_path}")
    print(f"       bin/build_binary {arpa_path} {bin_path}")
    print()
    print("  OPTION B: Use a pre-trained model")
    print("  ----------------------------------")
    print("  Download from https://kheafield.com/code/kenlm/")
    print(f"  Place the .arpa or .bin file in {model_dir}/")
    print()

    # Create a minimal test model
    _create_minimal_model(model_dir)

    return None


def _create_minimal_model(model_dir: str):
    """Create a minimal ARPA file for testing (not useful for real scoring)."""
    arpa_path = Path(model_dir) / "test_minimal.arpa"

    print(f"  Creating minimal test model at {arpa_path}")

    arpa_content = r"""\data\
ngram 1=10
ngram 2=5

\1-grams:
-1.0000	</s>
-99	<s>
-0.6021	i
-0.9031	ate
-0.7782	food
-0.9031	yesterday
-0.7782	she
-0.8451	is
-0.7782	happy
-0.7782	he

\2-grams:
-0.3010	<s> i
-0.3010	i ate
-0.3010	ate food
-0.3010	food yesterday
-0.3010	she is

\end\
"""

    with open(arpa_path, "w") as f:
        f.write(arpa_content)

    print(f"  Minimal test model created: {arpa_path}")
    print("  NOTE: This is for testing only. Use a real model for benchmarking.")


def verify_installation(model_path: str = None):
    """Verify kenlm installation and model loading."""
    print()
    print("=" * 60)
    print("Step 3: Verification")
    print("=" * 60)

    try:
        import kenlm
        print("  ✓ kenlm imported successfully")
    except ImportError:
        print("  ✗ kenlm NOT installed")
        return False

    if model_path and Path(model_path).exists():
        try:
            model = kenlm.Model(model_path)
            score = model.score("i ate food yesterday", bos=True, eos=True)
            print(f"  ✓ Model loaded: {model_path}")
            print(f"    Order: {model.order}")
            print(f"    Test score: {score:.4f}")
            return True
        except Exception as e:
            print(f"  ✗ Model loading failed: {e}")
            return False
    else:
        print("  ⚠ No model file to verify")
        return True


def main():
    parser = argparse.ArgumentParser(description="Setup KenLM for ISL pipeline")
    parser.add_argument(
        "--model-dir", type=str, default="./models",
        help="Directory to store language model files",
    )
    args = parser.parse_args()

    print("KenLM Setup for ISL Pipeline")
    print("=" * 60)

    # Step 1: Install package
    installed = install_kenlm()

    # Step 2: Download/setup model
    model_path = download_model(args.model_dir)

    # Step 3: Verify
    if installed:
        verify_installation(model_path)

    print()
    print("=" * 60)
    print("Setup complete.")
    print()
    print("To use KenLM in the pipeline, set:")
    print(f"  kenlm_model_path = '{args.model_dir}/english.bin'")
    print()
    print("Or run the full pipeline with:")
    print(f"  python output/full_pipeline.py --mode benchmark --kenlm_model {args.model_dir}/english.bin")
    print("=" * 60)


if __name__ == "__main__":
    main()
