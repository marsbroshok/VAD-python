"""Initial smoke tests for VAD-python package and wav-sample.wav fixture."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from vad import VoiceActivityDetector


def test_package_metadata_and_fixture_loading() -> None:
    """Verify package metadata is installed and wav-sample.wav loads cleanly."""
    version = importlib.metadata.version("vad-python")
    assert version == "0.2.0"

    sample_path = Path(__file__).resolve().parent.parent / "wav-sample.wav"
    assert sample_path.exists()

    detector = VoiceActivityDetector(str(sample_path))
    assert detector.rate > 0
    assert len(detector.data) > 0
