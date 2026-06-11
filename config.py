"""Configuration models and helpers for WhoSpoke.

This module defines Pydantic models for structured configuration. Users may
provide configuration via YAML files; the ``AnalysisConfig.from_yaml`` method
parses such files into validated objects. The configuration models here are
lightweight and intended for expansion in later milestones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class ASRConfig(BaseModel):
    """Configuration for automatic speech recognition backends.

    The minimal skeleton provides a ``backend`` field only. Future versions
    may include model names, language hints, beam sizes and more.
    """

    backend: str = Field(default="mock", description="ASR backend identifier")
    model: Optional[str] = Field(default=None, description="Model name or size")
    language: Optional[str] = Field(default="auto", description="Target language code or 'auto'")
    word_timestamps: bool = Field(default=True, description="Whether to request word‑level timestamps")


class FusionConfig(BaseModel):
    """Configuration for evidence fusion.

    The fusion stage combines evidence from multiple modalities. Default
    threshold values are conservative; advanced weighting will be added in
    future milestones.
    """

    mode: str = Field(default="balanced", description="Fusion weight preset (balanced, high_precision, etc.)")
    confidence_threshold: float = Field(default=0.6, description="Minimum score to assign a known identity")
    ambiguity_margin: float = Field(default=0.1, description="Margin between top two scores to avoid ambiguity")


class AnalysisConfig(BaseModel):
    """Top‑level configuration for running a video analysis.

    Additional sections (e.g. diarization, vision, voice, active_speaker) can
    be added as optional fields in future milestones. Unknown fields in the
    YAML will be ignored by default to allow forward compatibility.
    """

    asr: ASRConfig = Field(default_factory=ASRConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    device: str = Field(default="cpu", description="Computation device: cpu, cuda, or mps")

    class Config:
        extra = "ignore"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        """Load a configuration from a YAML file.

        Parameters
        ----------
        path:
            Path to a YAML file containing configuration fields. Unknown fields
            are ignored to maintain forward compatibility.

        Returns
        -------
        AnalysisConfig
            A validated configuration object.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file {path} is not a valid YAML mapping")
        try:
            return cls.parse_obj(data)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration in {path}: {e}")