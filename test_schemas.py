import json
from pathlib import Path

import pytest

from WhoSpoke.schemas import AnalysisResult, FinalSpeakerSegment, PersonProfile, TranscriptSegment, VideoMetadata


def test_person_profile_requires_id_and_name():
    # Valid instantiation
    person = PersonProfile(id="abc", name="Alice")
    assert person.id == "abc"
    assert person.name == "Alice"
    # Paths should be converted to Path objects
    person = PersonProfile(id="bob", name="Bob", portraits=["portrait.jpg"], voice_samples=["voice.wav"])
    assert all(isinstance(p, Path) for p in person.portraits)
    assert all(isinstance(v, Path) for v in person.voice_samples)

    # Missing id should raise an error
    with pytest.raises(Exception):
        PersonProfile(name="NoID")  # type: ignore


def test_analysis_result_serialization(tmp_path: Path):
    meta = VideoMetadata(video_id="vid", path=Path("video.mp4"))
    transcript = TranscriptSegment(segment_id="s1", start=0.0, end=1.0, text="hello")
    final = FinalSpeakerSegment(segment_id="s1", start=0.0, end=1.0, text="hello")
    result = AnalysisResult(video_metadata=meta, transcript_segments=[transcript], final_segments=[final])

    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)
    # JSON file should exist and contain the expected keys
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "video_metadata" in data
    assert "transcript_segments" in data
    assert "final_segments" in data
    # CSV file should exist and start with header
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("segment_id")