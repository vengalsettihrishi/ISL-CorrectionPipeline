"""
scripts/setup_translation_models.py — Set up Hindi translation for ISL pipeline.

Installs and configures argostranslate for English→Hindi translation.

Steps:
    1. Install argostranslate pip package
    2. Download and install the English→Hindi translation package
    3. Verify translation works

Usage:
    python scripts/setup_translation_models.py
"""

import subprocess
import sys


def install_argostranslate():
    """Install the argostranslate package."""
    print("=" * 60)
    print("Step 1: Installing argostranslate")
    print("=" * 60)

    try:
        import argostranslate
        print("  argostranslate already installed")
        return True
    except ImportError:
        pass

    print("  Installing argostranslate via pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "argostranslate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("  argostranslate installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Installation failed: {e}")
        print()
        print("  Try installing manually:")
        print("    pip install argostranslate")
        return False


def download_hindi_package():
    """Download and install the English→Hindi translation package."""
    print()
    print("=" * 60)
    print("Step 2: Downloading English→Hindi translation package")
    print("=" * 60)

    try:
        import argostranslate.package
        import argostranslate.translate

        # Check if already installed
        installed = argostranslate.translate.get_installed_languages()
        installed_codes = [lang.code for lang in installed]

        if "en" in installed_codes and "hi" in installed_codes:
            # Check if en→hi translation exists
            en_lang = next(l for l in installed if l.code == "en")
            hi_lang = next(l for l in installed if l.code == "hi")
            translation = en_lang.get_translation(hi_lang)
            if translation:
                print("  English→Hindi package already installed")
                return True

        # Update package index
        print("  Updating package index...")
        argostranslate.package.update_package_index()

        # Find English→Hindi package
        available = argostranslate.package.get_available_packages()
        en_hi_pkg = None
        for pkg in available:
            if pkg.from_code == "en" and pkg.to_code == "hi":
                en_hi_pkg = pkg
                break

        if en_hi_pkg is None:
            print("  ✗ English→Hindi package not found in index")
            print()
            print("  Try updating argostranslate:")
            print("    pip install --upgrade argostranslate")
            print("    Then re-run this script.")
            return False

        # Download and install
        print(f"  Downloading: {en_hi_pkg}")
        download_path = en_hi_pkg.download()
        argostranslate.package.install_from_path(download_path)
        print("  English→Hindi package installed successfully")
        return True

    except ImportError:
        print("  ✗ argostranslate not installed")
        return False
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print()
        print("  This may be due to network issues. Try:")
        print("    1. Check your internet connection")
        print("    2. pip install --upgrade argostranslate")
        print("    3. Re-run this script")
        return False


def verify_translation():
    """Verify that English→Hindi translation works."""
    print()
    print("=" * 60)
    print("Step 3: Verification")
    print("=" * 60)

    try:
        import argostranslate.translate

        installed = argostranslate.translate.get_installed_languages()
        en_lang = next((l for l in installed if l.code == "en"), None)
        hi_lang = next((l for l in installed if l.code == "hi"), None)

        if not en_lang or not hi_lang:
            print("  ✗ English or Hindi language not found")
            return False

        translation = en_lang.get_translation(hi_lang)
        if not translation:
            print("  ✗ English→Hindi translation not available")
            return False

        # Test translation
        test_sentences = [
            "Hello, how are you?",
            "I ate food yesterday.",
            "She is happy.",
            "What is your name?",
        ]

        print("  Translation test:")
        for en in test_sentences:
            hi = translation.translate(en)
            print(f"    EN: {en}")
            print(f"    HI: {hi}")
            print()

        print("  ✓ Translation verified successfully")
        return True

    except ImportError:
        print("  ✗ argostranslate not installed")
        return False
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def setup_tts():
    """Install TTS dependencies."""
    print()
    print("=" * 60)
    print("Step 4: TTS Dependencies (optional)")
    print("=" * 60)

    # pyttsx3 (offline TTS)
    try:
        import pyttsx3
        print("  ✓ pyttsx3 already installed")
    except ImportError:
        print("  Installing pyttsx3...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pyttsx3"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("  ✓ pyttsx3 installed")
        except subprocess.CalledProcessError:
            print("  ⚠ pyttsx3 installation failed (optional)")

    # gTTS (online TTS) - optional
    print()
    print("  For online TTS (higher quality), install gTTS:")
    print("    pip install gTTS")
    print()
    print("  For audio playback, install one of:")
    print("    pip install pygame")
    print("    pip install playsound")


def main():
    print("Hindi Translation & TTS Setup for ISL Pipeline")
    print("=" * 60)
    print()

    # Step 1: Install package
    installed = install_argostranslate()
    if not installed:
        print("\n⚠ Setup incomplete. Install argostranslate first.")
        return

    # Step 2: Download Hindi package
    downloaded = download_hindi_package()

    # Step 3: Verify
    if downloaded:
        verify_translation()

    # Step 4: TTS
    setup_tts()

    print()
    print("=" * 60)
    print("Setup complete.")
    print()
    print("To use in the pipeline:")
    print("  from correction.pipeline import CorrectionPipeline")
    print("  pipeline = CorrectionPipeline()")
    print("  result = pipeline.correct(['i', 'food', 'eat', 'yesterday'])")
    print("  print(result.hindi_translation)")
    print("=" * 60)


if __name__ == "__main__":
    main()
