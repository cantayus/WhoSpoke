"""Core analysis pipeline for WhoSpoke.

This module defines the ``SpeakerVideoAnalyzer`` class, which coordinates
multimodal speaker identification for video.  The current implementation remains
a safe lightweight skeleton: it validates configuration, records which algorithm
families were selected, and returns a structured mock result.  Future milestones
can replace the private stage methods with real ASR, diarization, face, voice,
active‑speaker, and fusion backends without changing the public API.
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
    """High‑level interface for analyzing videos.

    Parameters
    ----------
    config:
        Optional :class:`AnalysisConfig`.  The configuration controls which ASR,
        diarization, face, voice, active‑speaker and fusion algorithms should be
        used.  This implementation attempts to import and install optional
        dependencies on demand.  When dependencies are available, it will
        delegate to the selected backend; otherwise it falls back to a mock
        analysis that returns a dummy result.  This allows users to run the
        package with full capabilities by simply specifying their desired
        backends and ensuring that pip installation is permitted.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or AnalysisConfig()

    # ------------------------------------------------------------------
    # Dependency management
    #
    # The following helper functions attempt to import optional dependencies
    # required by specific backends.  If a module is missing, they will
    # install it using pip.  Because network access may be restricted in
    # some environments, installation may fail; in that case, the pipeline
    # will fall back to the mock implementation.
    #
    @staticmethod
    def _ensure_package(package: str) -> bool:
        """Ensure a Python package is importable.

        Returns ``True`` if the package is importable after this call, and
        ``False`` otherwise.  If the package is not found, this method
        attempts to install it using ``pip``.  Installation may fail if
        network or permissions are restricted.
        """
        import importlib
        import subprocess
        import sys

        try:
            importlib.import_module(package)
            return True
        except ImportError:
            try:
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package,
                ])
                importlib.invalidate_caches()
                importlib.import_module(package)
                return True
            except Exception:
                return False

    def _ensure_backend_dependencies(self) -> bool:
        """Ensure that optional dependencies for the configured backends are installed.

        Returns ``True`` if all required dependencies are available.  This
        method maps backend identifiers to the corresponding Python packages.
        Only the most common combinations are covered; additional backends
        may require manual installation by the user.  If any dependency
        cannot be installed, this returns ``False``.
        """
        success = True
        # ASR backends
        asr_backend = self.config.asr.backend
        if asr_backend == "whisper":
            # Try both whisper and openai-whisper; whichever succeeds first will import the module "whisper"
            success &= (self._ensure_package("whisper") or self._ensure_package("openai-whisper"))
        elif asr_backend == "faster_whisper":
            success &= self._ensure_package("faster-whisper")
        elif asr_backend == "vosk":
            success &= self._ensure_package("vosk")
        elif asr_backend == "deepspeech":
            success &= self._ensure_package("deepspeech")
        elif asr_backend == "wav2vec":
            # wav2vec requires both transformers and torchaudio
            success &= self._ensure_package("transformers")
            success &= self._ensure_package("torchaudio")
        elif asr_backend in {"wav2letter", "open_seq2seq", "kaldi"}:
            # Not implemented; cannot ensure packages here
            success = False
        # Diarization backends
        diar_algo = self.config.diarization.algorithm
        if diar_algo == "pyannote":
            success &= self._ensure_package("pyannote.audio")
        elif diar_algo in {"uis_rnn", "eend", "vb_hmm"}:
            # Attempt to install potential packages; they may not exist on PyPI
            success &= self._ensure_package("uis-rnn") if diar_algo == "uis_rnn" else success
            success &= self._ensure_package("eend") if diar_algo == "eend" else success
            success &= self._ensure_package("pyrodiarization") if diar_algo == "vb_hmm" else success
        # Face recognition backends
        face_backend = self.config.face.algorithm
        if face_backend not in {"mock", "dlib"}:
            success &= self._ensure_package("deepface")
        # Voice matching backends
        voice_backend = self.config.voice.algorithm
        if voice_backend == "resemblyzer":
            success &= self._ensure_package("resemblyzer")
        elif voice_backend in {"xvector", "ecapa_tdnn"}:
            success &= self._ensure_package("speechbrain")
            success &= self._ensure_package("torchaudio")
        # Active speaker backends
        as_backend = self.config.active_speaker.algorithm
        if as_backend in {"fast_asd", "talknet"}:
            # Try a couple of likely package names.  Some users may need to install
            # their own implementation; these names are placeholders.
            success &= (self._ensure_package("fast-asd") or self._ensure_package("fastasd") or self._ensure_package("talknet-asd"))
        return success

    # ------------------------------------------------------------------
    # Backend implementations
    #
    def _transcribe(self, audio_path: str) -> Optional[list[TranscriptSegment]]:
        """Transcribe audio using the configured ASR backend.

        Returns a list of :class:`TranscriptSegment` or ``None`` if the
        backend is unsupported or unavailable.
        """
        backend = self.config.asr.backend
        language = self.config.asr.language
        if backend in {"whisper", "faster_whisper"}:
            try:
                if backend == "whisper":
                    import whisper  # type: ignore
                    model_name = self.config.asr.model_name or "small"
                    model = whisper.load_model(model_name)
                    result = model.transcribe(audio_path, language=language)
                    segments = []
                    for idx, seg in enumerate(result.get("segments", [])):
                        segments.append(
                            TranscriptSegment(
                                segment_id=str(uuid.uuid4()),
                                start=seg.get("start", 0.0),
                                end=seg.get("end", 0.0),
                                text=seg.get("text", ""),
                                speaker_id=None,
                                language=language or seg.get("language"),
                                asr_confidence=seg.get("avg_logprob"),
                            )
                        )
                    return segments
                else:  # faster_whisper
                    from faster_whisper import WhisperModel  # type: ignore
                    model_name = self.config.asr.model_name or "small"
                    device = self.config.device if self.config.device in {"cuda", "cpu"} else "cpu"
                    model = WhisperModel(model_name, device=device)
                    segments, _ = model.transcribe(audio_path, beam_size=self.config.asr.beam_size or 5)
                    transcript_segments = []
                    for seg in segments:
                        transcript_segments.append(
                            TranscriptSegment(
                                segment_id=str(uuid.uuid4()),
                                start=seg.start,
                                end=seg.end,
                                text=seg.text,
                                speaker_id=None,
                                language=language,
                                asr_confidence=None,
                            )
                        )
                    return transcript_segments
            except Exception:
                return None
        # TODO: implement other ASR backends (vosk, deepspeech, wav2vec) as needed
        return None

    def _run_diarization(self, audio_path: str) -> Optional[list[tuple[float, float, int]]]:
        """Perform speaker diarization and return (start, end, speaker_idx) tuples.

        Currently only supports the pyannote pipeline when installed.  Returns
        ``None`` if diarization is unsupported or fails.
        """
        if self.config.diarization.algorithm == "pyannote":
            try:
                from pyannote.audio import Pipeline  # type: ignore
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                diarization = pipeline(audio_path)
                segments = []
                # itertracks yields segments with speaker labels
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append((segment.start, segment.end, speaker))
                return segments
            except Exception:
                return None
        # TODO: implement other diarization backends (uis_rnn, eend, vb_hmm, spectral, etc.)
        return None

    def _match_faces(self, frame_paths: list[str], people: Iterable[PersonProfile]) -> dict:
        """Match detected faces against known people using DeepFace.

        Returns a dictionary mapping frame index to (person_id, score) if
        matching succeeds.  If face recognition is unavailable, returns an
        empty dictionary.
        """
        if self.config.face.algorithm == "mock":
            return {}
        try:
            from deepface import DeepFace  # type: ignore
            matches: dict[int, tuple[str, float]] = {}
            for idx, frame in enumerate(frame_paths):
                try:
                    # Represent each face using the selected model
                    representation = DeepFace.represent(
                        img_path=frame,
                        model_name=self.config.face.algorithm,
                        detector_backend=self.config.face.detector_backend,
                        enforce_detection=self.config.face.enforce_detection,
                    )
                    # Compare with each enrolment portrait
                    best_match: tuple[str, float] | None = None
                    for person in people:
                        for portrait in person.portraits:
                            try:
                                result = DeepFace.verify(
                                    img1_path=frame,
                                    img2_path=portrait,
                                    model_name=self.config.face.algorithm,
                                    distance_metric=self.config.face.distance_metric,
                                    enforce_detection=self.config.face.enforce_detection,
                                )
                                if result.get("verified"):
                                    score = result.get("distance", 0.0)
                                    # lower distance implies better match
                                    if best_match is None or score < best_match[1]:
                                        best_match = (person.id, score)
                            except Exception:
                                continue
                    if best_match is not None:
                        matches[idx] = best_match
                except Exception:
                    continue
            return matches
        except Exception:
            return {}

    def _match_voice(self, audio_segments: list[tuple[float, float]], audio_path: str, people: Iterable[PersonProfile]) -> dict:
        """Match voice segments to known people using speaker embeddings.

        Returns a dictionary mapping segment index to (person_id, score) if
        matching succeeds.  This uses Resemblyzer when available.  If voice
        matching is unavailable, returns an empty dictionary.
        """
        if self.config.voice.algorithm == "mock":
            return {}
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
            # Load full audio
            wav, sr = sf.read(audio_path)
            encoder = VoiceEncoder()
            # Precompute embeddings for enrolment samples
            enrolment_embeddings: dict[str, np.ndarray] = {}
            for person in people:
                embeddings = []
                for sample in person.voice_samples:
                    try:
                        wav_sample, sr_sample = sf.read(sample)
                        embed = encoder.embed_utterance(preprocess_wav(wav_sample, sr_sample))
                        embeddings.append(embed)
                    except Exception:
                        continue
                if embeddings:
                    enrolment_embeddings[person.id] = np.mean(embeddings, axis=0)
            matches: dict[int, tuple[str, float]] = {}
            for idx, (start, end) in enumerate(audio_segments):
                start_idx = int(start * sr)
                end_idx = int(end * sr)
                seg = wav[start_idx:end_idx]
                try:
                    seg_embed = encoder.embed_utterance(preprocess_wav(seg, sr))
                except Exception:
                    continue
                best_match: tuple[str, float] | None = None
                for person_id, enrol_embed in enrolment_embeddings.items():
                    # cosine similarity
                    sim = float(np.dot(seg_embed, enrol_embed) / (np.linalg.norm(seg_embed) * np.linalg.norm(enrol_embed)))
                    # higher similarity implies better match
                    if best_match is None or sim > best_match[1]:
                        best_match = (person_id, sim)
                if best_match is not None:
                    matches[idx] = best_match
            return matches
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def analyze(
        self,
        video_path: str | Path,
        people: Iterable[PersonProfile] | None = None,
        language: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
    ) -> AnalysisResult:
        """Analyze a video and return a structured result.

        This method attempts to run the full pipeline.  If optional
        dependencies are missing and cannot be installed, it falls back
        to returning a single dummy segment.  The returned
        :class:`AnalysisResult` always conforms to the API, but will have
        minimal content when backends are unavailable.
        """
        video_path = Path(video_path)
        video_metadata = VideoMetadata(video_id=video_path.stem, path=video_path)
        people_list = list(people or [])

        # Try to ensure dependencies are present
        dependencies_ok = self._ensure_backend_dependencies()

        algorithm_trace = summarize_config_algorithms(self.config)

        # If dependencies are available, attempt to run each stage
        transcript_segments: list[TranscriptSegment] | None = None
        diarization_segments: list[tuple[float, float, int]] | None = None
        face_matches: dict = {}
        voice_matches: dict = {}

        if dependencies_ok:
            try:
                # Extract audio track from video to a temporary file
                import subprocess
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    tmp_audio_path = tmp_audio.name
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    tmp_audio_path,
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                # Run ASR
                transcript_segments = self._transcribe(tmp_audio_path)
                # Run diarization
                diarization_segments = self._run_diarization(tmp_audio_path)
                # Match voices (if diarization succeeded and voice backend enabled)
                if diarization_segments and self.config.voice.algorithm != "mock":
                    voice_matches = self._match_voice([(s, e) for s, e, _ in diarization_segments], tmp_audio_path, people_list)
                # Face recognition: extract one frame at the midpoint of each diarization segment
                face_matches = {}
                if diarization_segments and self.config.face.algorithm != "mock":
                    frame_paths: list[str] = []
                    for idx, (start, end, _spk) in enumerate(diarization_segments):
                        midpoint = (start + end) / 2.0
                        frame_path = str(Path(tmp_audio_path).with_name(f"frame_{idx}.jpg"))
                        # Use ffmpeg to extract a single frame
                        frame_cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(video_path),
                            "-ss",
                            str(midpoint),
                            "-frames:v",
                            "1",
                            frame_path,
                        ]
                        subprocess.run(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        frame_paths.append(frame_path)
                    face_matches = self._match_faces(frame_paths, people_list)
                    # Cleanup frame files afterwards
                    for f in frame_paths:
                        try:
                            os.unlink(f)
                        except Exception:
                            pass
                # Clean up temporary audio file
                os.unlink(tmp_audio_path)
            except Exception:
                pass
        # Build results
        if transcript_segments is None:
            # Fallback: dummy result
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
        else:
            final_segments: list[FinalSpeakerSegment] = []
            errors: list[str] = []
            warnings: list[str] = []
            # If diarization is unavailable, treat each transcript as its own segment
            if not diarization_segments:
                for ts in transcript_segments:
                    final_segments.append(
                        FinalSpeakerSegment(
                            segment_id=ts.segment_id,
                            start=ts.start,
                            end=ts.end,
                            text=ts.text,
                            speaker_id=None,
                            speaker_name=None,
                            confidence=None,
                            evidence_scores={
                                "asr": ts.asr_confidence or 1.0,
                                "diarization": 0.0,
                                "face": 0.0,
                                "voice": 0.0,
                                "active_speaker": 0.0,
                            },
                            algorithm_trace=algorithm_trace,
                            notes="ASR only; no diarization available.",
                        )
                    )
            else:
                # Basic fusion: assign diarized speaker indices to transcripts based on overlap
                # and then match voices if possible
                for ts in transcript_segments:
                    seg_match = None
                    for diar_idx, (start, end, spk_idx) in enumerate(diarization_segments):
                        # simple overlap check
                        overlap = max(0.0, min(ts.end, end) - max(ts.start, start))
                        if overlap > 0.0:
                            seg_match = diar_idx
                            break
                    # Determine predicted speaker based on face and voice evidence
                    speaker_id: Optional[str] = None
                    speaker_name: Optional[str] = None
                    voice_score = 0.0
                    face_score = 0.0
                    if seg_match is not None:
                        # Retrieve voice candidate for this diarization index
                        if isinstance(voice_matches, dict) and seg_match in voice_matches:
                            candidate_id, candidate_score = voice_matches[seg_match]
                            if candidate_id is not None:
                                speaker_id = candidate_id
                                voice_score = candidate_score or 0.0
                        # Retrieve face candidate for this diarization index
                        if isinstance(face_matches, dict) and seg_match in face_matches:
                            fc_id, fc_score = face_matches[seg_match]
                            face_score = fc_score or 0.0
                            # If no speaker predicted yet or face evidence is stronger, update
                            weights = self.config.fusion.weights
                            voice_weighted = voice_score * weights.get("voice", 1.0)
                            face_weighted = face_score * weights.get("face", 1.0)
                            if fc_id is not None and (speaker_id is None or face_weighted > voice_weighted):
                                speaker_id = fc_id
                    # Map speaker_id to name
                    if speaker_id is not None:
                        for person in people_list:
                            if person.id == speaker_id:
                                speaker_name = person.name
                                break
                    # Compute weighted confidence as in fusion section
                    evidence_scores = {
                        "asr": ts.asr_confidence or 1.0,
                        "diarization": 1.0 if seg_match is not None else 0.0,
                        "face": face_score,
                        "voice": voice_score,
                        "active_speaker": 0.0,
                    }
                    weights = self.config.fusion.weights
                    numerator = sum(evidence_scores[k] * weights.get(k, 0.0) for k in evidence_scores)
                    denominator = sum(weights.get(k, 0.0) for k in evidence_scores)
                    combined_confidence = numerator / denominator if denominator > 0 else None
                    final_segments.append(
                        FinalSpeakerSegment(
                            segment_id=ts.segment_id,
                            start=ts.start,
                            end=ts.end,
                            text=ts.text,
                            speaker_id=speaker_id,
                            speaker_name=speaker_name,
                            confidence=combined_confidence,
                            evidence_scores=evidence_scores,
                            algorithm_trace=algorithm_trace,
                            notes="Combined ASR, diarization, and optional voice/face matching.",
                        )
                    )
            return AnalysisResult(
                video_metadata=video_metadata,
                transcript_segments=transcript_segments,
                final_segments=final_segments,
                warnings=warnings,
                errors=errors,
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