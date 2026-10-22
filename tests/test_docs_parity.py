"""Documentation, CLI help parity, and golden test.json sync tests."""

from __future__ import annotations

import json
from pathlib import Path

import detectVoiceInWave
from vad import VoiceActivityDetector


def test_readme_modernized_and_no_local_paths() -> None:
    """README.md has no abandonment notice, no local paths, and modern APIs."""
    root = Path(__file__).resolve().parent.parent
    readme_text = (root / "README.md").read_text(encoding="utf-8")

    assert "not working on this project anymore" not in readme_text
    assert "/Users/" not in readme_text
    assert "/home/" not in readme_text

    for expected_symbol in (
        "VoiceActivityDetector",
        "detect_intervals",
        "strip_silence",
        "extract_speech_segments",
        "StreamingVAD",
        "--threshold",
        "--speech-window",
        "--output-wav",
        "--plot",
    ):
        assert expected_symbol in readme_text, (
            f"Missing {expected_symbol} in README.md"
        )


def test_cli_help_exposes_all_documented_flags() -> None:
    """CLI --help output exposes all positional and optional flags."""
    help_text = detectVoiceInWave.build_parser().format_help()
    for flag in (
        "INPUTWAVE",
        "OUTPUTFILE",
        "--threshold",
        "--energy-threshold",
        "--speech-window",
        "--output-wav",
        "--pad-ms",
        "--plot",
    ):
        assert flag in help_text


def test_golden_test_json_matches_wav_sample() -> None:
    """test.json in repo root matches output for wav-sample.wav."""
    root = Path(__file__).resolve().parent.parent
    detector = VoiceActivityDetector(root / "wav-sample.wav")
    expected_intervals = detector.detect_intervals()

    actual_json = json.loads((root / "test.json").read_text(encoding="utf-8"))
    assert actual_json == expected_intervals
