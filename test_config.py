from pathlib import Path

import yaml

from WhoSpoke.config import AnalysisConfig


def test_load_default_config(tmp_path: Path):
    # An empty YAML should produce default configuration
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("{}", encoding="utf-8")
    cfg = AnalysisConfig.from_yaml(cfg_file)
    assert cfg.asr.backend == "mock"
    assert cfg.fusion.mode == "balanced"
    assert cfg.device == "cpu"


def test_load_partial_config(tmp_path: Path):
    # Provide custom ASR backend and fusion threshold
    data = {
        "asr": {"backend": "dummy", "language": "en"},
        "fusion": {"confidence_threshold": 0.7},
        "device": "cuda",
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = AnalysisConfig.from_yaml(cfg_file)
    assert cfg.asr.backend == "dummy"
    assert cfg.asr.language == "en"
    assert cfg.fusion.confidence_threshold == 0.7
    assert cfg.device == "cuda"