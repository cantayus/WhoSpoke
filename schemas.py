"""Data schemas for the WhoSpoke package.

This module defines Pydantic models that structure the inputs, outputs,
and intermediate data used throughout the library. These schemas aim to
capture the essential pieces of information while remaining agnostic to
specific machine learning backends. They may evolve over time as new
capabilities are added, but backward‑compatible changes will be maintained
through semantic versioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PersonProfile(BaseModel):
    """Represents a known person with optional portrait and voice samples.

    The ``id`` should be a unique string (e.g. slug or identifier) that
    persists across analyses. The ``name`` is a human‑readable display name.
    Portraits and voice samples are optional lists of file paths. Additional
    metadata and notes can be provided to assist with tracking and auditing.
    """

    id: str = Field(..., description="Unique identifier for the person")
    name: str = Field(..., description="Human‑readable display name")
    portraits: List[Path] = Field(default_factory=list, description="Paths to portrait images")
    voice_samples: List[Path] = Field(default_factory=list, description="Paths to sample audio files")
    metadata: Dict[str, str] | None = Field(default=None, description="Optional arbitrary metadata about the person")
    notes: str | None = Field(default=None, description="Optional notes about the person")
    external_ids: Dict[str, str] | None = Field(default=None, description="Optional map of external system IDs for this person")

    @validator("portraits", "voice_samples", pre=True, each_item=True)
    def _ensure_path(cls, v: str | Path) -> Path:
        """Convert string paths into ``Path`` objects for consistency."""
        return Path(v)


class VideoMetadata(BaseModel):
    """Metadata about the input video.

    Not all fields are required. Duration, frame rate, and other values may
    remain unknown until processed by later pipeline stages.
    """

    video_id: str
    path: Path
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None


class TranscriptSegment(BaseModel):
    """Represents a segment of transcribed speech.

    ``start`` and ``end`` are floating‑point seconds relative to the beginning
    of the video. ``text`` contains the ASR transcript. ``speaker_id`` is
    optional and may be filled by diarization or identity resolution. A
    ``language`` code may be provided if known. Confidence scores from
    the ASR backend can be stored in ``asr_confidence``.
    """

    segment_id: str
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    language: Optional[str] = None
    asr_confidence: Optional[float] = None


class FinalSpeakerSegment(BaseModel):
    """Represents a final speaker-labeled segment produced by the pipeline.

    After evidence fusion, each segment includes the resolved ``speaker_id`` and
    optionally a ``speaker_name``, a confidence score, and any notes or warnings.
    ``evidence_scores`` can store per-modality confidence scores (for example
    face, voice, active-speaker and diarization scores). ``algorithm_trace``
    records which configured algorithms contributed to the assignment.
    """

    segment_id: str
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    confidence: Optional[float] = None
    evidence_scores: Optional[Dict[str, float]] = None
    algorithm_trace: Optional[Dict[str, str]] = None
    notes: Optional[str] = None


class AnalysisResult(BaseModel):
    """Holds the results of analyzing a single video.

    This object contains the video metadata, raw transcript segments, final
    speaker segments, and any warnings or errors encountered. It also provides
    convenience methods for exporting the results to JSON or CSV.
    """

    video_metadata: VideoMetadata
    transcript_segments: List[TranscriptSegment]
    final_segments: List[FinalSpeakerSegment]
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable dictionary representation of the result."""
        return self.model_dump()

    def to_json(self, path: str | Path) -> None:
        """Serialize the analysis result to a JSON file.

        Parameters
        ----------
        path:
            Destination file path. The parent directory will be created if it
            does not exist.
        """
        import json
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            # Use default=str to convert non‑serialisable types like Path to strings
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def to_csv(self, path: str | Path) -> None:
        """Serialize the final speaker segments to a CSV file.

        The CSV contains one row per final segment with start, end, speaker and
        text. Additional columns may be added in future versions to capture
        confidence scores and evidence details.

        Parameters
        ----------
        path:
            Destination file path. The parent directory will be created if it
            does not exist.
        """
        import csv
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "segment_id",
                "start",
                "end",
                "speaker_id",
                "speaker_name",
                "confidence",
                "evidence_scores",
                "algorithm_trace",
                "text",
            ])
            for seg in self.final_segments:
                writer.writerow([
                    seg.segment_id,
                    seg.start,
                    seg.end,
                    seg.speaker_id or "",
                    seg.speaker_name or "",
                    f"{seg.confidence:.4f}" if seg.confidence is not None else "",
                    seg.text,
                ])