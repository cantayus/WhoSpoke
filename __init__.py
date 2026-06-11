"""Top-level package for WhoSpoke.

WhoSpoke provides a multimodal interface for speaker identification in video.
The current release exposes a stable configuration and result schema while
keeping heavy ML backends optional.
"""

from __future__ import annotations

from .backend_registry import (
    ACTIVE_SPEAKER_BACKENDS,
    ASR_BACKENDS,
    CLUSTERING_METHODS,
    DIARIZATION_ALGORITHMS,
    FACE_BACKBONES,
    VOICE_BACKENDS,
    supported_backend_names,
)
from .config import AnalysisConfig
from .pipeline import SpeakerVideoAnalyzer, analyze_video
from .schemas import AnalysisResult, FinalSpeakerSegment, PersonProfile, TranscriptSegment

__all__ = [
    "ACTIVE_SPEAKER_BACKENDS",
    "ASR_BACKENDS",
    "CLUSTERING_METHODS",
    "DIARIZATION_ALGORITHMS",
    "FACE_BACKBONES",
    "VOICE_BACKENDS",
    "AnalysisConfig",
    "AnalysisResult",
    "FinalSpeakerSegment",
    "PersonProfile",
    "SpeakerVideoAnalyzer",
    "TranscriptSegment",
    "analyze_video",
    "supported_backend_names",
]

__version__ = "0.1.0"
