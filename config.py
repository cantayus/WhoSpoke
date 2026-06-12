"""Configuration models and helpers for WhoSpoke.

The package is intentionally backend‑agnostic.  These Pydantic models record
which ASR, diarization, face matching, voice matching, active‑speaker detection,
and fusion settings the user wants.  Heavy machine‑learning libraries are not
imported here; optional backend implementations can be installed separately and
wired into the pipeline in future milestones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, validator

from .backend_registry import validate_backend_choice


class ASRConfig(BaseModel):
    """Configuration for automatic speech recognition backends."""

    backend: str = Field(
        default="mock",
        description=(
            "ASR backend identifier: mock, whisper, faster_whisper, kaldi, "
            "vosk, deepspeech, wav2letter, open_seq2seq, wav2vec"
        ),
    )
    model: Optional[str] = Field(default=None, description="Deprecated alias for model_name")
    model_name: Optional[str] = Field(default=None, description="ASR model name or size")
    language: Optional[str] = Field(default="auto", description="Target language code, 'auto', or null")
    task: str = Field(default="transcribe", description="ASR task: transcribe or translate")
    word_timestamps: bool = Field(default=True, description="Whether to request word‑level timestamps")

    # Common decoding parameters, especially useful for Whisper-like models.
    beam_size: Optional[int] = Field(default=None, ge=1, description="Beam size for beam search")
    best_of: Optional[int] = Field(default=None, ge=1, description="Number of candidates for sampling")
    temperature: Optional[List[float]] = Field(default=None, description="Temperature schedule")
    compression_ratio_threshold: Optional[float] = Field(default=None, description="Maximum compression ratio")
    log_prob_threshold: Optional[float] = Field(default=None, description="Minimum average log probability")
    no_speech_threshold: Optional[float] = Field(default=None, description="No‑speech probability threshold")

    # Streaming/offline backends such as Vosk may use these.
    sample_rate: Optional[int] = Field(default=None, description="Expected sample rate")
    streaming: bool = Field(default=False, description="Whether to use a streaming ASR mode")
    vocabulary: Optional[List[str]] = Field(default=None, description="Optional custom vocabulary or phrase list")

    @validator("backend")
    def _validate_backend(cls, value: str) -> str:
        return validate_backend_choice("asr", value)


class DiarizationSegmentationConfig(BaseModel):
    """Parameters controlling speech activity and segmentation."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Speech probability threshold")
    min_duration_on: float = Field(default=0.1, ge=0.0, description="Minimum speech duration to keep")
    min_duration_off: float = Field(default=0.1, ge=0.0, description="Maximum silence gap to fill")
    window_size: Optional[float] = Field(default=None, gt=0.0, description="Sliding‑window size in seconds")
    step_size: Optional[float] = Field(default=None, gt=0.0, description="Sliding‑window step in seconds")
    max_speakers_per_window: Optional[int] = Field(default=None, ge=1, description="Maximum local speakers per window")


class DiarizationEmbeddingConfig(BaseModel):
    """Parameters controlling speaker embeddings."""

    model: str = Field(default="xvector", description="Embedding model: ivector, dvector, xvector, ecapa_tdnn, resnet, mock")
    normalize: bool = Field(default=True, description="L2‑normalize embeddings before clustering")
    metric: str = Field(default="cosine", description="Distance metric: cosine, euclidean, plda")


class DiarizationClusteringConfig(BaseModel):
    """Parameters controlling speaker clustering."""

    method: str = Field(default="ahc", description="ahc, kmeans, spectral, affinity, uis_rnn, vb_hmm, mock")
    threshold: float = Field(default=0.7, ge=0.0, description="Distance or affinity threshold")
    num_speakers: Optional[int] = Field(default=None, ge=1, description="Known number of speakers")
    min_speakers: Optional[int] = Field(default=None, ge=1, description="Minimum speaker count")
    max_speakers: Optional[int] = Field(default=None, ge=1, description="Maximum speaker count")
    plda: bool = Field(default=False, description="Whether to use PLDA scoring when available")

    @validator("method")
    def _validate_method(cls, value: str) -> str:
        return validate_backend_choice("clustering", value)


class DiarizationAggregationConfig(BaseModel):
    """Parameters controlling post‑processing of diarization segments."""

    gap: float = Field(default=0.25, ge=0.0, description="Bridge same‑speaker gaps shorter than this")
    min_segment_duration: float = Field(default=0.0, ge=0.0, description="Drop segments shorter than this")
    collar: float = Field(default=0.0, ge=0.0, description="Evaluation collar for DER‑style metrics")


class DiarizationConfig(BaseModel):
    """High‑level diarization configuration."""

    algorithm: str = Field(
        default="mock",
        description="pyannote, uis_rnn, eend, vb_hmm, spectral, kmeans, affinity, mock",
    )
    overlap: bool = Field(default=True, description="Whether to allow overlapping speech labels")
    segmentation: DiarizationSegmentationConfig = Field(default_factory=DiarizationSegmentationConfig)
    embedding: DiarizationEmbeddingConfig = Field(default_factory=DiarizationEmbeddingConfig)
    clustering: DiarizationClusteringConfig = Field(default_factory=DiarizationClusteringConfig)
    aggregation: DiarizationAggregationConfig = Field(default_factory=DiarizationAggregationConfig)

    @validator("algorithm")
    def _validate_algorithm(cls, value: str) -> str:
        return validate_backend_choice("diarization", value)


class FaceRecognitionConfig(BaseModel):
    """Configuration for face recognition and matching."""

    algorithm: str = Field(
        default="arcface",
        description="vgg_face, facenet, facenet512, openface, deepface, deepid, arcface, dlib, sface, ghostfacenet, buffalo_l, mock",
    )
    detector_backend: str = Field(default="retinaface", description="opencv, ssd, mtcnn, retinaface, mediapipe, yolov8, centerface, skip")
    align: bool = Field(default=True, description="Align faces before embedding")
    normalization: str = Field(default="base", description="DeepFace normalization mode")
    distance_metric: str = Field(default="cosine", description="cosine, euclidean, euclidean_l2")
    similarity_threshold: float = Field(default=0.67, ge=0.0, le=1.0, description="Face match threshold")
    enforce_detection: bool = Field(default=False, description="Fail if no face is detected")
    max_faces_per_frame: Optional[int] = Field(default=None, ge=1, description="Limit faces evaluated per frame")

    @validator("algorithm")
    def _validate_algorithm(cls, value: str) -> str:
        return validate_backend_choice("face", value)


class VoiceMatchingConfig(BaseModel):
    """Configuration for voice embedding and matching."""

    algorithm: str = Field(default="resemblyzer", description="resemblyzer, xvector, ecapa_tdnn, mock")
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Voice match threshold")
    min_sample_duration: float = Field(default=2.0, ge=0.0, description="Minimum enrolment sample duration")
    min_segment_duration: float = Field(default=1.0, ge=0.0, description="Minimum segment duration to match")
    metric: str = Field(default="cosine", description="cosine, euclidean, plda")

    @validator("algorithm")
    def _validate_algorithm(cls, value: str) -> str:
        return validate_backend_choice("voice", value)


class ActiveSpeakerConfig(BaseModel):
    """Configuration for active speaker detection."""

    algorithm: str = Field(default="fast_asd", description="fast_asd, talknet, mock")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence threshold")
    frame_rate: Optional[float] = Field(default=None, gt=0.0, description="Frame rate used for ASD")
    face_detection_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Face detector threshold")

    @validator("algorithm")
    def _validate_algorithm(cls, value: str) -> str:
        return validate_backend_choice("active_speaker", value)


class FusionConfig(BaseModel):
    """Configuration for multimodal evidence fusion."""

    mode: str = Field(default="balanced", description="balanced, high_precision, high_recall, visual_first, audio_first")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum final confidence")
    ambiguity_margin: float = Field(default=0.1, ge=0.0, le=1.0, description="Required margin between top identities")
    face_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override face threshold")
    voice_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override voice threshold")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "asr": 1.0,
            "diarization": 1.0,
            "face": 1.0,
            "voice": 1.0,
            "active_speaker": 1.0,
        },
        description="Per‑modality weights",
    )


class AnalysisConfig(BaseModel):
    """Top‑level configuration for running a video analysis."""

    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    face: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    voice: VoiceMatchingConfig = Field(default_factory=VoiceMatchingConfig)
    active_speaker: ActiveSpeakerConfig = Field(default_factory=ActiveSpeakerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    device: str = Field(default="cpu", description="Computation device: cpu, cuda, mps, or auto")
    batch_size: int = Field(default=1, ge=1, description="Batch size for supported backends")
    num_workers: int = Field(default=0, ge=0, description="Number of worker processes/threads")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory for intermediate files")

    class Config:
        extra = "ignore"
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        """Load a configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file {path} is not a valid YAML mapping")
        try:
            return cls.parse_obj(data)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration in {path}: {e}") from e

    def algorithm_summary(self) -> Dict[str, str]:
        """Return a compact dictionary of selected algorithms."""
        return {
            "asr_backend": self.asr.backend,
            "asr_model_name": self.asr.model_name or self.asr.model or "default",
            "diarization_algorithm": self.diarization.algorithm,
            "clustering_method": self.diarization.clustering.method,
            "face_algorithm": self.face.algorithm,
            "voice_algorithm": self.voice.algorithm,
            "active_speaker_algorithm": self.active_speaker.algorithm,
        }