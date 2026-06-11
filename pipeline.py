"""Core analysis pipeline for WhoSpoke.

This module defines the ``SpeakerVideoAnalyzer`` class, which coordinates
the various stages of processing a video. In this initial milestone the
pipeline performs a minimal mock analysis: it does not run any machine
learning models but instead returns placeholder transcript and speaker
segments. Future milestones will add real ASR, diarization, face and voice
matching, active speaker detection, and evidence fusion.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, List, Optional

from .config import AnalysisConfig
from .schemas import (
    AnalysisResult,
    FinalSpeakerSegment,
    PersonProfile,
    TranscriptSegment,
    VideoMetadata,
)


class SpeakerVideoAnalyzer:
    """High‑level interface for analyzing videos.

    The analyzer can be instantiated with a configuration object, or it will
    use default settings. The main method, :meth:`analyze`, accepts a video
    path and a list of known people and returns an :class:`AnalysisResult`.
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

        This initial implementation produces a dummy transcript and labels
        every segment as unknown. It demonstrates the overall flow and
        interfaces without depending on heavy ML backends.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        people:
            Optional iterable of :class:`PersonProfile` objects representing
            enrolled speakers. These are currently unused in the mock
            implementation but reserved for future use.
        language:
            Optional language code to override automatic language detection.
        output_dir:
            Optional directory where outputs will be written. The mock
            implementation does not write files here; use the ``to_json`` and
            ``to_csv`` methods on the returned :class:`AnalysisResult`.
        """
        # Normalise paths and prepare metadata
        video_path = Path(video_path)
        video_id = video_path.stem
        video_metadata = VideoMetadata(video_id=video_id, path=video_path)

        # Generate a dummy transcript: single segment covering zero to five seconds
        dummy_transcript = TranscriptSegment(
            segment_id=str(uuid.uuid4()),
            start=0.0,
            end=5.0,
            text="[DUMMY] This is a placeholder transcript segment.",
            speaker_id=None,
            language=language or self.config.asr.language,
            asr_confidence=1.0,
        )

        # The final speaker segment mirrors the transcript segment but keeps the speaker unknown
        final_segment = FinalSpeakerSegment(
            segment_id=dummy_transcript.segment_id,
            start=dummy_transcript.start,
            end=dummy_transcript.end,
            text=dummy_transcript.text,
            speaker_id=None,
            speaker_name=None,
            confidence=None,
            notes="This is a mock result. Real models will provide confidence scores."
        )

        result = AnalysisResult(
            video_metadata=video_metadata,
            transcript_segments=[dummy_transcript],
            final_segments=[final_segment],
            warnings=["Using mock analysis; no ML models were run."],
            errors=[],
        )

        return result


def analyze_video(
    video_path: str | Path,
    people: Iterable[PersonProfile] | None = None,
    config: Optional[AnalysisConfig] = None,
    language: Optional[str] = None,
) -> AnalysisResult:
    """Convenience function for analyzing a video using a simple interface.

    This wraps :class:`SpeakerVideoAnalyzer` for cases where users do not
    explicitly need to instantiate the class. It accepts the same
    arguments as :meth:`SpeakerVideoAnalyzer.analyze` except ``output_dir``.
    """
    analyzer = SpeakerVideoAnalyzer(config=config)
    return analyzer.analyze(video_path=video_path, people=people, language=language)