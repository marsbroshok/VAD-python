"""Unit and property tests for median smoothing, interval closure, and lazy plotting."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vad import VoiceActivityDetector


def test_speech_at_end_of_audio_is_closed_and_no_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Speech active until the end of the audio stream is closed with no print output."""
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    # 0.5s silence followed by 0.5s of 1000 Hz speech until the very last sample
    audio = np.zeros(sr, dtype=np.float64)
    audio[sr // 2 :] = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t[sr // 2 :])

    detector = VoiceActivityDetector(audio, sample_rate=sr, speech_window=0.1)
    windows = detector.detect_speech()
    labels = detector.convert_windows_to_readible_labels(windows)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(labels) == 1
    interval = labels[0]
    assert 0.45 <= interval["speech_begin"] <= 0.55
    assert np.isclose(interval["speech_end"], 1.0)
    assert interval["start_sample"] == int(round(interval["speech_begin"] * sr))
    assert interval["end_sample"] == sr


def test_readable_labels_alias_and_detect_intervals() -> None:
    """convert_windows_to_readable_labels and detect_intervals return identical rich dicts."""
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    audio = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)

    detector = VoiceActivityDetector(audio, sample_rate=sr, speech_window=0.1)
    windows = detector.detect_speech()
    legacy = detector.convert_windows_to_readible_labels(windows)
    modern = detector.convert_windows_to_readable_labels(windows)
    direct = detector.detect_intervals()

    assert legacy == modern == direct
    assert len(direct) == 1
    assert direct[0] == {
        "speech_begin": 0.0,
        "speech_end": 1.0,
        "start_sample": 0,
        "end_sample": sr,
    }


def test_speech_followed_by_silence_closes_before_end() -> None:
    """Speech in the first half followed by silence closes around 0.5s."""
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    audio = np.zeros(sr, dtype=np.float64)
    audio[: sr // 2] = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t[: sr // 2])

    detector = VoiceActivityDetector(audio, sample_rate=sr, speech_window=0.1)
    intervals = detector.detect_intervals()

    assert len(intervals) == 1
    assert intervals[0]["speech_begin"] == 0.0
    assert 0.45 <= intervals[0]["speech_end"] <= 0.55
    assert intervals[0]["start_sample"] == 0
    assert int(sr * 0.45) <= intervals[0]["end_sample"] <= int(sr * 0.55)


def test_short_clip_shorter_than_speech_window() -> None:
    """Audio shorter than speech_window (0.5s) is smoothed safely without crashing."""
    sr = 16000
    # 0.1s clip (1600 samples) while default speech_window is 0.5s
    t = np.arange(1600, dtype=np.float64) / sr
    short_speech = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)

    detector = VoiceActivityDetector(short_speech, sample_rate=sr, speech_window=0.5)
    intervals = detector.detect_intervals()

    assert len(intervals) == 1
    assert intervals[0]["speech_begin"] == 0.0
    assert np.isclose(intervals[0]["speech_end"], 0.1)


def test_median_filter_uses_sample_overlap_step() -> None:
    """_smooth_speech_detection calculates median window using sample_overlap (10ms)."""
    sr = 16000
    audio = np.zeros(sr, dtype=np.float64)
    detector = VoiceActivityDetector(
        audio,
        sample_rate=sr,
        sample_window=0.02,
        sample_overlap=0.01,
        speech_window=0.5,
    )
    # 0.5s / 0.01s = 50 -> odd kernel 49 (which filters out a 20-frame / 200ms spike)
    # Under the old bug (0.5 / 0.02 = 25), a 15-frame (150ms) spike in 50 frames would pass
    # if kernel was 25 vs 49: a 20-frame spike is > 12 (half of 25) so old bug kept it,
    # whereas 20 < 25 (half of 49) so fixed 49-frame kernel smooths it out!
    windows = np.zeros((100, 2), dtype=np.float64)
    windows[:, 0] = np.arange(100) * 160
    windows[40:60, 1] = 1.0  # 20 frames (200ms < 250ms half-window of 0.5s)

    smoothed = detector._smooth_speech_detection(windows)
    assert np.all(smoothed == 0.0)

    # Edge cases for _median_filter and empty windows
    with pytest.raises(ValueError, match="odd"):
        detector._median_filter(np.ones(5), 4)
    with pytest.raises(ValueError, match="one-dimensional"):
        detector._median_filter(np.ones((2, 2)), 3)
    assert len(detector._median_filter(np.array([]), 3)) == 0
    assert len(detector._smooth_speech_detection(np.empty((0, 2)))) == 0
    assert detector.convert_windows_to_readible_labels(np.empty((0, 2))) == []


def test_headless_lazy_plotting_and_full_window_masking() -> None:
    """plot_detected_speech_regions lazy-loads matplotlib and fills full frame slices."""
    sr = 16000
    t = np.arange(3200, dtype=np.float64) / sr
    speech = np.zeros(3200, dtype=np.float64)
    speech[:1600] = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t[:1600])
    detector = VoiceActivityDetector(speech, sample_rate=sr, speech_window=0.05)

    mock_plt = MagicMock()
    with patch.dict(sys.modules, {"matplotlib.pyplot": mock_plt}):
        result = detector.plot_detected_speech_regions()

    assert result is detector
    assert mock_plt.figure.called
    assert mock_plt.plot.call_count == 2
    assert mock_plt.show.called

    # First plot call should be raw waveform, second should be masked speech waveform
    first_plotted = mock_plt.plot.call_args_list[0][0][0]
    second_plotted = mock_plt.plot.call_args_list[1][0][0]
    assert np.allclose(first_plotted, detector.data)
    # First half is speech (filled frames), second half is silence (zeros)
    assert np.count_nonzero(second_plotted[:1600]) > 1400
    assert np.count_nonzero(second_plotted[2000:]) == 0


@given(
    st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=80),
)
@settings(max_examples=50)
def test_property_intervals_are_ordered_and_disjoint(flags: list[float]) -> None:
    """Converted intervals always have start < end and strictly increasing non-overlapping bounds."""
    sr = 16000
    detector = VoiceActivityDetector(np.zeros(len(flags) * 160), sample_rate=sr)
    windows = np.column_stack(
        (np.arange(len(flags), dtype=np.float64) * 160.0, np.array(flags, dtype=np.float64))
    )
    intervals = detector.convert_windows_to_readable_labels(windows)

    prev_end = -1.0
    for item in intervals:
        assert item["speech_begin"] >= 0.0
        assert item["speech_end"] > item["speech_begin"]
        assert item["start_sample"] < item["end_sample"]
        assert item["speech_begin"] >= prev_end
        prev_end = float(item["speech_end"])
