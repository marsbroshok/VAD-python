"""Integration tests for detectVoiceInWave CLI and console script entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

import detectVoiceInWave


def test_cli_positional_args_and_save_json(tmp_path: Path) -> None:
    """Running main([input, output]) writes speech intervals JSON to outputfile."""
    sample_wav = Path(__file__).resolve().parent.parent / "wav-sample.wav"
    out_json = tmp_path / "results.json"

    exit_code = detectVoiceInWave.main([str(sample_wav), str(out_json)])
    assert exit_code == 0
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "speech_begin" in data[0]
    assert "speech_end" in data[0]
    assert "start_sample" in data[0]
    assert "end_sample" in data[0]


def test_cli_optional_flags_threshold_speech_window_output_wav_and_plot(
    tmp_path: Path,
) -> None:
    """CLI supports --threshold, --speech-window, --output-wav, and --plot flags."""
    sample_wav = Path(__file__).resolve().parent.parent / "wav-sample.wav"
    out_json = tmp_path / "custom.json"
    out_wav = tmp_path / "stripped.wav"

    mock_plt = MagicMock()
    with patch.dict(sys.modules, {"matplotlib.pyplot": mock_plt}):
        exit_code = detectVoiceInWave.main(
            [
                str(sample_wav),
                str(out_json),
                "--threshold",
                "0.5",
                "--speech-window",
                "0.2",
                "--output-wav",
                str(out_wav),
                "--pad-ms",
                "10.0",
                "--plot",
            ]
        )

    assert exit_code == 0
    assert out_json.exists()
    assert out_wav.exists()
    assert mock_plt.show.called

    # Also test running without positional outputfile (stdout only)
    exit_code_no_out = detectVoiceInWave.main([str(sample_wav)])
    assert exit_code_no_out == 0


def test_cli_missing_input_file_raises_system_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing input WAV file prints error to stderr and exits with code 1."""
    missing_wav = tmp_path / "does_not_exist.wav"
    out_json = tmp_path / "out.json"

    with pytest.raises(SystemExit) as exc_info:
        detectVoiceInWave.main([str(missing_wav), str(out_json)])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()
