"""Core analysis pipeline for WhoSpoke.

This module defines the ``SpeakerVideoAnalyzer`` class, which coordinates
multimodal speaker identification for video.  The current implementation remains
a safe lightweight skeleton: it validates configuration, records which algorithm
families were selected, and returns a structured mock result.  Future milestones
can replace the private stage methods with real ASR, diarization, face, voice,
active-speaker, and fusion backends without changing the public API.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Optional

from .backend_registry import summarize_config_algorithms
from .config import AnalysisConfig
from .schemas import (
    AnalysisResult,
    FinalSpeakerSegment,
    PersonProfile,
    TranscriptSegment,
    VideoMetadata,
)


class SpeakerVideoAnalyzer:
    """High-level interface for analyzing videos.

    Parameters
    ----------
    config:
        Optional :class:`AnalysisConfig`.  The configuration controls which ASR,
        diarization, face, voice, active-speaker and fusion algorithms should be
        used.  In this skeleton, the settings are recorded and surfaced in the
        result; real backends can be added later behind the same interface.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        video_path: str | Path,
        people: Iterable[PersonProfile] | None = None,
        language: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
    ) -> AnalysisResult:
        """Analyze a video and return a structured result.

        This implementation produces a dummy transcript and labels every segment
        as unknown.  It demonstrates the public API and configuration flow
        without importing heavy optional machine-learning dependencies.
        """
        video_path = Path(video_path)
        video_metadata = VideoMetadata(video_id=video_path.stem, path=video_path)
        people_list = list(people or [])
        algorithm_trace = summarize_config_algorithms(self.config)

        dummy_transcript = TranscriptSegment(
            segment_id=str(uuid.uuid4()),
            start=0.0,
            end=5.0,
            text="[DUMMY] This is a placeholder transcript segment.",
            speaker_id=None,
            language=language or self.config.asr.language,
            asr_confidence=1.0,
        )

        final_segment = FinalSpeakerSegment(
            segment_id=dummy_transcript.segment_id,
            start=dummy_transcript.start,
            end=dummy_transcript.end,
            text=dummy_transcript.text,
            speaker_id=None,
            speaker_name=None,
            confidence=None,
            evidence_scores={
                "asr": dummy_transcript.asr_confidence or 0.0,
                "diarization": 0.0,
                "face": 0.0,
                "voice": 0.0,
                "active_speaker": 0.0,
            },
            algorithm_trace=algorithm_trace,
            notes="This is a mock result. Real models will provide speaker labels and confidence scores.",
        )

        warnings = [
            "Using mock analysis; no ML models were run.",
            f"Configured people: {len(people_list)}",
            *[f"{key}: {value}" for key, value in algorithm_trace.items()],
        ]

        return AnalysisResult(
            video_metadata=video_metadata,
            transcript_segments=[dummy_transcript],
            final_segments=[final_segment],
            warnings=warnings,
            errors=[],
        )


def analyze_video(
    video_path: str | Path,
    people: Iterable[PersonProfile] | None = None,
    config: Optional[AnalysisConfig] = None,
    language: Optional[str] = None,
) -> AnalysisResult:
    """Convenience function for analyzing a video using a simple interface."""
    analyzer = SpeakerVideoAnalyzer(config=config)
    return analyzer.analyze(video_path=video_path, people=people, language=language)
