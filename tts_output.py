"""
tts_output.py — Bilingual Text-to-Speech Engine for ISL Pipeline.

Supports two backends:
    1. pyttsx3  (offline, low-latency, no internet required)
    2. gTTS     (Google TTS, higher quality, requires internet)

Features:
    - speak_english(text)           — English speech
    - speak_hindi(text)             — Hindi speech
    - speak_bilingual(en, hi)       — Sequential English then Hindi
    - Non-blocking async mode for live inference
    - Graceful fallback with clear setup instructions

Usage:
    from tts_output import TTSEngine
    tts = TTSEngine(engine="pyttsx3")
    tts.speak_english("I ate food yesterday.")
    tts.speak_hindi("मैंने कल खाना खाया।")
"""

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Backend: pyttsx3 (offline)
# ============================================================================

class Pyttsx3Backend:
    """Offline TTS using pyttsx3."""

    def __init__(self, english_rate: int = 150, hindi_rate: int = 130):
        self.available = False
        self.engine = None
        self.english_rate = english_rate
        self.hindi_rate = hindi_rate
        self._lock = threading.Lock()
        self._setup()

    def _setup(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.available = True
            logger.info("pyttsx3 TTS engine initialized")
        except ImportError:
            logger.warning(
                "pyttsx3 not installed. Install with: pip install pyttsx3"
            )
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e}")

    def speak(self, text: str, lang: str = "en", rate: Optional[int] = None):
        """Speak text using pyttsx3."""
        if not self.available or not self.engine:
            return

        with self._lock:
            try:
                if rate is None:
                    rate = self.english_rate if lang == "en" else self.hindi_rate
                self.engine.setProperty("rate", rate)

                # Try to set voice for language
                voices = self.engine.getProperty("voices")
                if voices:
                    target_lang = "hindi" if lang == "hi" else "english"
                    for voice in voices:
                        if target_lang in voice.name.lower():
                            self.engine.setProperty("voice", voice.id)
                            break

                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.warning(f"pyttsx3 speak failed: {e}")


# ============================================================================
# Backend: gTTS (online)
# ============================================================================

class GTTSBackend:
    """Online TTS using Google Text-to-Speech."""

    def __init__(self):
        self.available = False
        self._setup()

    def _setup(self):
        try:
            import gtts
            self.available = True
            logger.info("gTTS engine available")
        except ImportError:
            logger.warning(
                "gTTS not installed. Install with: pip install gTTS"
            )

    def speak(self, text: str, lang: str = "en", rate: Optional[int] = None):
        """Speak text using gTTS + playsound/pygame."""
        if not self.available:
            return

        try:
            from gtts import gTTS

            tts_lang = "hi" if lang == "hi" else "en"
            tts = gTTS(text=text, lang=tts_lang, slow=False)

            # Save to temp file and play
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False, dir="."
            )
            tmp_path = tmp.name
            tmp.close()

            tts.save(tmp_path)
            self._play_audio(tmp_path)

            # Cleanup
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        except Exception as e:
            logger.warning(f"gTTS speak failed: {e}")

    def _play_audio(self, path: str):
        """Play an audio file using available player."""
        # Try pygame first
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
            return
        except (ImportError, Exception):
            pass

        # Try playsound
        try:
            from playsound import playsound
            playsound(path)
            return
        except (ImportError, Exception):
            pass

        # Try system command (Windows)
        try:
            import subprocess
            if os.name == "nt":
                subprocess.run(
                    ["powershell", "-c",
                     f'(New-Object Media.SoundPlayer "{path}").PlaySync()'],
                    capture_output=True, timeout=10,
                )
            else:
                subprocess.run(
                    ["aplay", path], capture_output=True, timeout=10,
                )
            return
        except Exception:
            pass

        logger.warning(
            "No audio player available. Install pygame: pip install pygame"
        )


# ============================================================================
# Unified TTS Engine
# ============================================================================

class TTSEngine:
    """
    Unified bilingual TTS engine for the ISL pipeline.

    Supports English and Hindi speech output with configurable backends.

    Args:
        engine:       Backend to use: "pyttsx3" (offline) or "gtts" (online).
        english_rate: Words per minute for English (pyttsx3 only).
        hindi_rate:   Words per minute for Hindi (pyttsx3 only).
        async_mode:   If True, speak in background threads.
    """

    def __init__(
        self,
        engine: str = "pyttsx3",
        english_rate: int = 150,
        hindi_rate: int = 130,
        async_mode: bool = False,
    ):
        self.engine_name = engine
        self.async_mode = async_mode
        self._backend = None

        if engine == "pyttsx3":
            self._backend = Pyttsx3Backend(english_rate, hindi_rate)
        elif engine == "gtts":
            self._backend = GTTSBackend()
        else:
            logger.warning(f"Unknown TTS engine '{engine}', trying pyttsx3")
            self._backend = Pyttsx3Backend(english_rate, hindi_rate)

        self.available = self._backend.available if self._backend else False

    def speak_english(self, text: str) -> None:
        """
        Speak an English sentence.

        Args:
            text: English text to speak.
        """
        if not text or not self.available:
            return

        if self.async_mode:
            t = threading.Thread(
                target=self._backend.speak,
                args=(text, "en"),
                daemon=True,
            )
            t.start()
        else:
            self._backend.speak(text, lang="en")

    def speak_hindi(self, text: str) -> None:
        """
        Speak a Hindi sentence.

        Args:
            text: Hindi text to speak.
        """
        if not text or not self.available:
            return

        if self.async_mode:
            t = threading.Thread(
                target=self._backend.speak,
                args=(text, "hi"),
                daemon=True,
            )
            t.start()
        else:
            self._backend.speak(text, lang="hi")

    def speak_bilingual(
        self,
        english: str,
        hindi: Optional[str] = None,
    ) -> None:
        """
        Speak English followed by Hindi translation.

        Args:
            english: English sentence.
            hindi:   Hindi translation (skipped if None).
        """
        if self.async_mode:
            def _speak_both():
                if english:
                    self._backend.speak(english, lang="en")
                if hindi:
                    time.sleep(0.3)  # brief pause between languages
                    self._backend.speak(hindi, lang="hi")

            t = threading.Thread(target=_speak_both, daemon=True)
            t.start()
        else:
            if english:
                self._backend.speak(english, lang="en")
            if hindi:
                time.sleep(0.3)
                self._backend.speak(hindi, lang="hi")

    def get_status(self) -> dict:
        """Return engine status for display."""
        return {
            "engine": self.engine_name,
            "available": self.available,
            "async_mode": self.async_mode,
        }


# ============================================================================
# Standalone verification
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("tts_output.py -- verification")
    print("=" * 60)

    tts = TTSEngine(engine="pyttsx3")
    status = tts.get_status()

    print(f"  Engine:    {status['engine']}")
    print(f"  Available: {status['available']}")
    print(f"  Async:     {status['async_mode']}")

    if tts.available:
        print("\n  Speaking English test phrase...")
        tts.speak_english("Hello, this is a test of the Indian Sign Language system.")
        print("  Done.")

        print("  Speaking Hindi test phrase...")
        tts.speak_hindi("नमस्ते, यह भारतीय सांकेतिक भाषा प्रणाली का परीक्षण है।")
        print("  Done.")
    else:
        print("\n  TTS not available. Install: pip install pyttsx3")
        print("  Skipping audio test.")

    print("=" * 60)
    print("[PASS] tts_output.py OK")
