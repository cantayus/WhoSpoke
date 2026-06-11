"""Top‑level package for WhoSpoke.

This module exposes the main user‑facing classes and functions.

The public API is deliberately small. Heavy machine learning backends are
optional and configured via the pipeline and configuration classes. See
``SpeakerVideoAnalyzer`` for the main entry point.
"""

from __future__ import annotations

from .schemas import PersonProfile, TranscriptSegment, FinalSpeakerSegment
from .pipeline import SpeakerVideoAnalyzer

__all__ = [
    "PersonProfile",
    "TranscriptSegment",
    "FinalSpeakerSegment",
    "SpeakerVideoAnalyzer",
]

__version__ = "0.1.0"