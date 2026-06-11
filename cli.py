"""Command‑line interface for the WhoSpoke package.

This module uses Typer to expose a friendly CLI. The current implementation
supports a single ``analyze`` command which runs the mock analysis pipeline
and writes JSON and CSV outputs. Future milestones will expand the CLI with
additional subcommands for enrollment, diarization, transcription, inspection,
and demonstrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from .config import AnalysisConfig
from .pipeline import SpeakerVideoAnalyzer
from .schemas import PersonProfile


app = typer.Typer(add_completion=False, help="WhoSpoke command‑line interface")


def load_people(people_path: Path) -> List[PersonProfile]:
    """Load a list of ``PersonProfile`` objects from a YAML file.

    The YAML should have a top‑level key ``people`` containing a list of
    dictionaries. Unknown fields are ignored. If no file is provided an
    empty list is returned.
    """
    import yaml

    if not people_path.exists():
        raise typer.BadParameter(f"People file not found: {people_path}")

    with people_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or "people" not in data:
        raise typer.BadParameter("People YAML must contain a top‑level 'people' list")
    people_list = data.get("people", [])
    profiles: List[PersonProfile] = []
    for person_data in people_list:
        try:
            profiles.append(PersonProfile.parse_obj(person_data))
        except Exception as e:
            raise typer.BadParameter(f"Invalid person entry: {e}") from e
    return profiles


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to the video file to analyze"),
    people: Optional[str] = typer.Option(None, help="Path to a YAML file describing known people"),
    output: Optional[str] = typer.Option(None, help="Directory where outputs will be written"),
    config_file: Optional[str] = typer.Option(None, help="Path to an analysis configuration YAML file"),
    language: Optional[str] = typer.Option(None, help="Override language code (e.g. 'en') or 'auto'"),
    quiet: bool = typer.Option(False, help="Suppress most console output"),
) -> None:
    """Analyze a video and output results to files.

    This command runs the WhoSpoke analysis pipeline on the provided video. If
    a people YAML file is supplied, known speaker profiles will be loaded and
    used in the analysis. The results are written to JSON and CSV files in
    the specified output directory.
    """
    console = Console(force_terminal=True) if not quiet else Console(record=True)

    video_path = Path(video)
    if not video_path.exists():
        console.print(f"[red]Video file not found:[/red] {video_path}")
        raise typer.Exit(code=1)

    # Load configuration if provided
    cfg: AnalysisConfig
    if config_file:
        cfg = AnalysisConfig.from_yaml(config_file)
    else:
        cfg = AnalysisConfig()

    # Load people profiles
    people_profiles: List[PersonProfile] = []
    if people:
        people_profiles = load_people(Path(people))

    analyzer = SpeakerVideoAnalyzer(config=cfg)
    result = analyzer.analyze(video_path=video_path, people=people_profiles, language=language)

    # Determine output directory
    out_dir = Path(output) if output else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write outputs
    json_path = out_dir / f"{video_path.stem}_result.json"
    csv_path = out_dir / f"{video_path.stem}_result.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)

    if not quiet:
        console.print(f"[green]Analysis complete![/green]")
        console.print(f"JSON output written to: {json_path}")
        console.print(f"CSV output written to: {csv_path}")
