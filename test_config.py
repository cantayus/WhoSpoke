from pathlib import Path

import pytest
import yaml

from WhoSpoke.config import AnalysisConfig
from WhoSpoke.backend_registry import supported_backend_names


def test_load_default_config(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("{}", encoding="utf-8")
    cfg = AnalysisConfig.from_yaml(cfg_file)
    assert cfg.asr.backend == "mock"
    assert cfg.diarization.algorithm == "mock"
    assert cfg.diarization.clustering.method == "ahc"
    assert cfg.face.algorithm == "arcface"
    assert cfg.voice.algorithm == "resemblyzer"
    assert cfg.active_speaker.algorithm == "fast_asd"
    assert cfg.fusion.mode == "balanced"
    assert cfg.device == "cpu"


def test_load_expanded_config(tmp_path: Path):
    data = {
        "asr": {
            "backend": "whisper",
            "model_name": "large",
            "language": None,
            "beam_size": 5,
            "temperature": [0.0, 0.2, 0.4],
        },
        "diarization": {
            "algorithm": "pyannote",
            "segmentation": {"threshold": 0.55, "min_duration_on": 0.3},
            "embedding": {"model": "ecapa_tdnn", "metric": "cosine"},
            "clustering": {"method": "spectral", "threshold": 0.75},
            "aggregation": {"gap": 0.3},
        },
        "face": {"algorithm": "facenet", "similarity_threshold": 0.68},
        "voice": {"algorithm": "ecapa_tdnn", "similarity_threshold": 0.8},
        "active_speaker": {"algorithm": "talknet", "threshold": 0.6},
        "fusion": {
            "confidence_threshold": 0.7,
            "weights": {"face": 1.2, "voice": 1.5, "active_speaker": 1.0},
        },
        "device": "cuda",
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = AnalysisConfig.from_yaml(cfg_file)
    assert cfg.asr.backend == "whisper"
    assert cfg.asr.model_name == "large"
    assert cfg.diarization.algorithm == "pyannote"
    assert cfg.diarization.clustering.method == "spectral"
    assert cfg.face.algorithm == "facenet"
    assert cfg.voice.algorithm == "ecapa_tdnn"
    assert cfg.active_speaker.algorithm == "talknet"
    assert cfg.fusion.confidence_threshold == 0.7
    assert cfg.device == "cuda"
    assert cfg.algorithm_summary()["asr_backend"] == "whisper"


def test_invalid_backend_rejected(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"asr": {"backend": "not_a_backend"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        AnalysisConfig.from_yaml(cfg_file)


def test_registry_contains_expected_families():
    assert "whisper" in supported_backend_names("asr")
    assert "pyannote" in supported_backend_names("diarization")
    assert "arcface" in supported_backend_names("face")
    assert "resemblyzer" in supported_backend_names("voice")
