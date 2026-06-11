"""Backend registry for WhoSpoke.

This module centralizes the list of algorithm families that WhoSpoke can
coordinate.  The registry is intentionally lightweight: it does not import any
heavy ML frameworks and does not require optional dependencies.  Instead, it
records names, categories, expected optional extras and descriptions so the
configuration layer, CLI, tests and documentation can stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class BackendSpec:
    """Description of a supported backend or algorithm.

    Parameters
    ----------
    name:
        Machine-readable identifier used in YAML config files and CLI flags.
    display_name:
        Human-readable model or algorithm name.
    category:
        One of ``asr``, ``diarization``, ``clustering``, ``face``, ``voice`` or
        ``active_speaker``.
    optional_extra:
        Suggested pip extra for installing the backend dependencies.  This is
        informational only in the current skeleton.
    description:
        Short description shown by the CLI.
    """

    name: str
    display_name: str
    category: str
    optional_extra: str | None = None
    description: str = ""


ASR_BACKENDS: Dict[str, BackendSpec] = {
    "mock": BackendSpec("mock", "Mock ASR", "asr", None, "Placeholder backend used for tests and demos."),
    "whisper": BackendSpec("whisper", "OpenAI Whisper", "asr", "asr-whisper", "Multilingual encoder-decoder transformer ASR."),
    "faster_whisper": BackendSpec("faster_whisper", "faster-whisper", "asr", "asr-whisper", "CTranslate2 implementation of Whisper for faster inference."),
    "kaldi": BackendSpec("kaldi", "Kaldi", "asr", "asr-kaldi", "Highly configurable classical ASR toolkit."),
    "vosk": BackendSpec("vosk", "Vosk", "asr", "asr-vosk", "Lightweight offline and streaming ASR."),
    "deepspeech": BackendSpec("deepspeech", "DeepSpeech", "asr", "asr-deepspeech", "RNN/CTC speech recognition engine."),
    "wav2letter": BackendSpec("wav2letter", "Wav2Letter++", "asr", "asr-wav2letter", "Fast end-to-end convolutional ASR."),
    "open_seq2seq": BackendSpec("open_seq2seq", "OpenSeq2Seq", "asr", "asr-open-seq2seq", "Sequence-to-sequence toolkit for ASR and related tasks."),
    "wav2vec": BackendSpec("wav2vec", "Wav2Vec / Wav2Vec2", "asr", "asr-wav2vec", "Self-supervised speech representation model."),
}

DIARIZATION_ALGORITHMS: Dict[str, BackendSpec] = {
    "mock": BackendSpec("mock", "Mock diarization", "diarization", None, "No-op diarization for tests and demos."),
    "pyannote": BackendSpec("pyannote", "pyannote.audio", "diarization", "diarization-pyannote", "Neural segmentation plus embedding and clustering pipeline."),
    "uis_rnn": BackendSpec("uis_rnn", "UIS-RNN", "diarization", "diarization-uis-rnn", "Supervised recurrent diarization / online clustering."),
    "eend": BackendSpec("eend", "EEND", "diarization", "diarization-eend", "End-to-end neural diarization supporting overlapping speech."),
    "vb_hmm": BackendSpec("vb_hmm", "VB-HMM", "diarization", "diarization-vb-hmm", "Variational Bayes HMM re-clustering."),
    "spectral": BackendSpec("spectral", "Spectral diarization", "diarization", "diarization-spectral", "Embedding pipeline with spectral clustering."),
    "kmeans": BackendSpec("kmeans", "K-means diarization", "diarization", "diarization-basic", "Embedding pipeline with k-means clustering."),
    "affinity": BackendSpec("affinity", "Affinity-propagation diarization", "diarization", "diarization-basic", "Embedding pipeline with affinity propagation."),
}

CLUSTERING_METHODS: Dict[str, BackendSpec] = {
    "ahc": BackendSpec("ahc", "Agglomerative hierarchical clustering", "clustering", None, "Bottom-up clustering with a distance threshold."),
    "kmeans": BackendSpec("kmeans", "K-means", "clustering", None, "Centroid-based clustering."),
    "spectral": BackendSpec("spectral", "Spectral clustering", "clustering", None, "Affinity-matrix and eigenvector clustering."),
    "affinity": BackendSpec("affinity", "Affinity propagation", "clustering", None, "Message-passing clustering without specifying k."),
    "uis_rnn": BackendSpec("uis_rnn", "UIS-RNN", "clustering", None, "Supervised recurrent online clustering."),
    "vb_hmm": BackendSpec("vb_hmm", "VB-HMM", "clustering", None, "Variational Bayesian HMM re-clustering."),
    "mock": BackendSpec("mock", "Mock clustering", "clustering", None, "No-op clustering."),
}

FACE_BACKBONES: Dict[str, BackendSpec] = {
    "mock": BackendSpec("mock", "Mock face matcher", "face", None, "Placeholder face matcher."),
    "vgg_face": BackendSpec("vgg_face", "VGG-Face", "face", "face-deepface", "DeepFace VGG-Face backend."),
    "facenet": BackendSpec("facenet", "FaceNet", "face", "face-deepface", "DeepFace FaceNet backend."),
    "facenet512": BackendSpec("facenet512", "FaceNet512", "face", "face-deepface", "512-dimensional FaceNet backend."),
    "openface": BackendSpec("openface", "OpenFace", "face", "face-deepface", "Lightweight OpenFace backend."),
    "deepface": BackendSpec("deepface", "DeepFace", "face", "face-deepface", "Original DeepFace model backend."),
    "deepid": BackendSpec("deepid", "DeepID", "face", "face-deepface", "DeepID face verification backend."),
    "arcface": BackendSpec("arcface", "ArcFace", "face", "face-deepface", "Additive angular margin face embedding model."),
    "dlib": BackendSpec("dlib", "Dlib", "face", "face-deepface", "Dlib face recognition backend."),
    "sface": BackendSpec("sface", "SFace", "face", "face-deepface", "SFace backend."),
    "ghostfacenet": BackendSpec("ghostfacenet", "GhostFaceNet", "face", "face-deepface", "Lightweight GhostFaceNet backend."),
    "buffalo_l": BackendSpec("buffalo_l", "Buffalo_L", "face", "face-deepface", "InsightFace Buffalo_L-style backend."),
}

VOICE_BACKENDS: Dict[str, BackendSpec] = {
    "mock": BackendSpec("mock", "Mock voice matcher", "voice", None, "Placeholder voice matcher."),
    "resemblyzer": BackendSpec("resemblyzer", "Resemblyzer", "voice", "voice-resemblyzer", "GE2E speaker embedding model."),
    "xvector": BackendSpec("xvector", "x-vector", "voice", "voice-speechbrain", "DNN speaker embedding model."),
    "ecapa_tdnn": BackendSpec("ecapa_tdnn", "ECAPA-TDNN", "voice", "voice-speechbrain", "TDNN speaker verification model with channel attention."),
}

ACTIVE_SPEAKER_BACKENDS: Dict[str, BackendSpec] = {
    "mock": BackendSpec("mock", "Mock active speaker detector", "active_speaker", None, "No-op active speaker detection."),
    "fast_asd": BackendSpec("fast_asd", "Fast-ASD / TalkNet", "active_speaker", "active-speaker", "Optimized TalkNet-based active speaker detection."),
    "talknet": BackendSpec("talknet", "TalkNet", "active_speaker", "active-speaker", "Audio-visual active speaker detection model."),
}

REGISTRIES: Dict[str, Mapping[str, BackendSpec]] = {
    "asr": ASR_BACKENDS,
    "diarization": DIARIZATION_ALGORITHMS,
    "clustering": CLUSTERING_METHODS,
    "face": FACE_BACKBONES,
    "voice": VOICE_BACKENDS,
    "active_speaker": ACTIVE_SPEAKER_BACKENDS,
}


def supported_backend_names(category: str) -> List[str]:
    """Return the supported backend names for a category."""
    try:
        return sorted(REGISTRIES[category].keys())
    except KeyError as exc:
        raise ValueError(f"Unknown backend category: {category}") from exc


def validate_backend_choice(category: str, name: str) -> str:
    """Validate and normalize a backend name.

    The function returns the original string if it is supported.  It raises a
    ``ValueError`` with a helpful message if the choice is invalid.
    """
    normalized = name.lower().replace("-", "_")
    valid = supported_backend_names(category)
    if normalized not in valid:
        raise ValueError(
            f"Unsupported {category} backend '{name}'. Supported values: {', '.join(valid)}"
        )
    return normalized


def iter_backend_specs(categories: Iterable[str] | None = None) -> Iterable[BackendSpec]:
    """Yield backend specifications for one or more categories."""
    selected = categories or REGISTRIES.keys()
    for category in selected:
        for spec in REGISTRIES[category].values():
            yield spec


def summarize_config_algorithms(config: object) -> Dict[str, str]:
    """Return a compact summary of algorithm choices in an ``AnalysisConfig``.

    This helper avoids importing ``AnalysisConfig`` directly and therefore keeps
    the registry free of circular imports.
    """
    return {
        "asr_backend": getattr(getattr(config, "asr", None), "backend", "unknown"),
        "asr_model_name": getattr(getattr(config, "asr", None), "model_name", None) or getattr(getattr(config, "asr", None), "model", None) or "default",
        "diarization_algorithm": getattr(getattr(config, "diarization", None), "algorithm", "unknown"),
        "clustering_method": getattr(getattr(getattr(config, "diarization", None), "clustering", None), "method", "unknown"),
        "face_algorithm": getattr(getattr(config, "face", None), "algorithm", "unknown"),
        "voice_algorithm": getattr(getattr(config, "voice", None), "algorithm", "unknown"),
        "active_speaker_algorithm": getattr(getattr(config, "active_speaker", None), "algorithm", "unknown"),
    }
