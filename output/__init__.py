"""
output -- Sprint 4: Visual/terminal display and end-to-end runner.

Modules:
    display        : Terminal/visual display with rich formatting
    full_pipeline  : End-to-end runner (webcam, video, benchmark modes)

Usage:
    python -m output.full_pipeline --mode benchmark --data_dir ./data_iSign/
    python -m output.full_pipeline --mode video --input test_video.mp4
    python -m output.full_pipeline --mode webcam
"""

from output.display import DisplayEngine
from output.full_pipeline import FullPipeline

__all__ = [
    "DisplayEngine",
    "FullPipeline",
]
