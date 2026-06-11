"""Command-line interface for the WhoSpoke package."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .backend_registry import REGISTRIES, iter_backend_specs
from .config import AnalysisConfig
from .pipeline import SpeakerVideoAnalyzer
from .schemas import PersonProfile


app = typer.Typer(add_completion=False, help="WhoSpoke command-line interface")


def load_people(people_path: Path) -> List[PersonProfile]:
    """Load a list of ``PersonProfile`` objects from a YAML file."""
    import yaml

    if not people_path.exists():
        raise typer.BadParameter(f"People file not found: {people_path}")

    with people_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or "people" not in data:
        raise typer.BadParameter("People YAML must contain a top-level 'people' list")

    profiles: List[PersonProfile] = []
    for person_data in data.get("people", []):
        try:
            profiles.append(PersonProfile.parse_obj(person_data))
        except Exception as exc:
            raise typer.BadParameter(f"Invalid person entry: {exc}") from exc
    return profiles


def _apply_overrides(
    cfg: AnalysisConfig,
    *,
    asr_backend: Optional[str] = None,
    asr_model_name: Optional[str] = None,
    diarization_algorithm: Optional[str] = None,
    clustering_method: Optional[str] = None,
    face_algorithm: Optional[str] = None,
    voice_algorithm: Optional[str] = None,
    active_speaker_algorithm: Optional[str] = None,
) -> AnalysisConfig:
    """Apply CLI overrides to an ``AnalysisConfig`` in place and return it."""
    if asr_backend:
        cfg.asr.backend = asr_backend
    if asr_model_name:
        cfg.asr.model_name = asr_model_name
    if diarization_algorithm:
        cfg.diarization.algorithm = diarization_algorithm
    if clustering_method:
        cfg.diarization.clustering.method = clustering_method
    if face_algorithm:
        cfg.face.algorithm = face_algorithm
    if voice_algorithm:
        cfg.voice.algorithm = voice_algorithm
    if active_speaker_algorithm:
        cfg.active_speaker.algorithm = active_speaker_algorithm
    return cfg


@app.command()
def list_backends(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Limit results to one category: asr, diarization, clustering, face, voice, active_speaker",
    )
) -> None:
    """List configurable algorithms and backends supported by WhoSpoke."""
    console = Console()
    categories = [category] if category else list(REGISTRIES.keys())
    table = Table(title="WhoSpoke supported algorithm registry")
    table.add_column("Category")
    table.add_column("Name")
    table.add_column("Display name")
    table.add_column("Extra")
    table.add_column("Description")
    for spec in iter_backend_specs(categories):
        table.add_row(spec.category, spec.name, spec.display_name, spec.optional_extra or "", spec.description)
    console.print(table)


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to the video file to analyze"),
    people: Optional[str] = typer.Option(None, "--people", help="Path to a YAML file describing known people"),
    output: Optional[str] = typer.Option(None, "--output", help="Directory where outputs will be written"),
    config_file: Optional[str] = typer.Option(None, "--config", "--config-file", help="Path to an analysis configuration YAML file"),
    language: Optional[str] = typer.Option(None, "--language", help="Override language code (e.g. 'en') or 'auto'"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress most console output"),
    # Algorithm overrides
    asr_backend: Optional[str] = typer.Option(None, "--asr-backend", help="ASR backend: whisper, kaldi, vosk, etc."),
    asr_model_name: Optional[str] = typer.Option(None, "--asr-model-name", help="ASR model name or size"),
    diarization_algorithm: Optional[str] = typer.Option(None, "--diarization-algorithm", help="Diarization algorithm: pyannote, eend, uis_rnn, etc."),
    clustering_method: Optional[str] = typer.Option(None, "--clustering-method", help="Clustering method: ahc, spectral, kmeans, etc."),
    face_algorithm: Optional[str] = typer.Option(None, "--face-algorithm", help="Face backbone: arcface, facenet, vgg_face, etc."),
    voice_algorithm: Optional[str] = typer.Option(None, "--voice-algorithm", help="Voice model: resemblyzer, xvector, ecapa_tdnn, etc."),
    active_speaker_algorithm: Optional[str] = typer.Option(None, "--active-speaker-algorithm", help="Active-speaker algorithm: fast_asd, talknet, mock"),
) -> None:
    """Analyze a video and write JSON and CSV outputs."""
    console = Console(force_terminal=True) if not quiet else Console(record=True)

    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Video file not found:[/red] {video_path}")
        raise typer.Exit(code=1)

    cfg = AnalysisConfig.from_yaml(config_file) if config_file else AnalysisConfig()
    try:
        cfg = _apply_overrides(
            cfg,
            asr_backend=asr_backend,
            asr_model_name=asr_model_name,
            diarization_algorithm=diarization_algorithm,
            clustering_method=clustering_method,
            face_algorithm=face_algorithm,
            voice_algorithm=voice_algorithm,
            active_speaker_algorithm=active_speaker_algorithm,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid backend selection:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    people_profiles: List[PersonProfile] = load_people(Path(people)) if people else []

    analyzer = SpeakerVideoAnalyzer(config=cfg)
    result = analyzer.analyze(video_path=video_path, people=people_profiles, language=language)

    out_dir = Path(output) if output else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{video_path.stem}_result.json"
    csv_path = out_dir / f"{video_path.stem}_result.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)

    if not quiet:
        console.print("[green]Analysis complete![/green]")
        console.print(f"JSON output written to: {json_path}")
        console.print(f"CSV output written to: {csv_path}")
        console.print("Configured algorithms:")
        for key, value in cfg.algorithm_summary().items():
            console.print(f"  - {key}: {value}")
