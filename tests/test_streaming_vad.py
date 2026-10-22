"""Unit and property-based tests for real-time StreamingVAD."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import vad


def test_streaming_vad_exported_and_frame_by_frame() -> None:
    """StreamingVAD is exported from vad and detects frame transitions."""
    from vad_streaming import StreamingVAD

    assert vad.StreamingVAD is StreamingVAD

    sr = 16000
    streamer = StreamingVAD(sample_rate=sr, speech_window=0.05)
    frame_len = streamer.frame_samples  # 320 samples (20ms)

    silent_frame = np.zeros(frame_len, dtype=np.float64)
    t = np.arange(frame_len, dtype=np.float64) / sr
    speech_frame = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)

    events: list[str] = []
    # Feed 5 silent frames
    for _ in range(5):
        res = streamer.process_frame(silent_frame)
        assert res["is_speech"] is False
        assert res["event"] is None

    # Feed 10 speech frames -> should transition to speech_begin
    for _ in range(10):
        res = streamer.process_frame(speech_frame)
        if res["event"] is not None:
            events.append(str(res["event"]))
    assert streamer.is_speech is True
    assert events == ["speech_begin"]

    # Feed 10 silent frames -> should transition to speech_end after hangover
    for _ in range(10):
        res = streamer.process_frame(silent_frame)
        if res["event"] is not None:
            events.append(str(res["event"]))
            assert res["interval"] is not None
    assert streamer.is_speech is False
    assert events == ["speech_begin", "speech_end"]
    assert len(streamer.completed_intervals) == 1


def test_streaming_vad_arbitrary_chunk_sizes_and_flush() -> None:
    """process_chunk handles arbitrary chunk sizes, stereo/bytes, and flush."""
    from vad_streaming import StreamingVAD

    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    # 0.3s silence, 0.4s speech, 0.3s speech until end
    audio = np.zeros(sr, dtype=np.float64)
    audio[int(sr * 0.3) :] = 0.5 * np.sin(
        2.0 * np.pi * 1000.0 * t[int(sr * 0.3) :]
    )

    streamer = StreamingVAD(sample_rate=sr, speech_window=0.05)
    emitted: list[dict[str, float | int | str]] = []

    # Feed in irregular 137-sample chunks
    idx = 0
    while idx < len(audio):
        chunk = audio[idx : idx + 137]
        emitted.extend(streamer.process_chunk(chunk))
        idx += 137

    assert streamer.is_speech is True
    assert len(emitted) == 1
    assert emitted[0]["event"] == "speech_begin"

    final_interval = streamer.flush()
    assert final_interval is not None
    assert streamer.is_speech is False
    assert np.isclose(float(final_interval["speech_end"]), 1.0)
    assert len(streamer.completed_intervals) == 1

    # Calling flush() when not in speech returns None
    assert streamer.flush() is None

    # Test reset() and bytes/stereo chunk support
    streamer.reset()
    assert streamer.is_speech is False
    assert len(streamer.completed_intervals) == 0

    pcm_bytes = struct.pack("<320h", *([16000] * 320))
    res_bytes = streamer.process_frame(pcm_bytes)
    assert res_bytes["is_speech"] is False  # constant DC has 0 AC energy

    stereo_chunk = np.column_stack((audio[:640], audio[:640]))
    streamer.process_chunk(stereo_chunk)


def test_streaming_vad_validation_errors() -> None:
    """Invalid parameters or chunk types raise ValueError / TypeError."""
    from vad_streaming import StreamingVAD

    with pytest.raises(ValueError, match="sample_rate"):
        StreamingVAD(sample_rate=0)

    with pytest.raises(ValueError, match="positive"):
        StreamingVAD(sample_rate=16000, sample_window=0.0)

    streamer = StreamingVAD(sample_rate=16000)
    with pytest.raises(ValueError, match="even number of bytes"):
        streamer.process_chunk(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="1D or 2D"):
        streamer.process_chunk(np.zeros((2, 2, 2)))

    with pytest.raises(TypeError, match="Unsupported audio chunk"):
        streamer.process_chunk(12345)  # type: ignore[arg-type]

    # Empty chunk and short/custom frame edge cases + uint8/int16/int32 dtypes
    even_win_streamer = StreamingVAD(sample_rate=16000, speech_window=0.04)
    assert even_win_streamer.history_len == 5
    assert streamer.process_chunk(np.array([], dtype=np.float64)) == []
    short_res = streamer.process_frame(np.array([0.0], dtype=np.float64))
    assert short_res["is_speech"] is False

    custom_len_frame = np.zeros(160, dtype=np.uint8) + 128
    assert streamer.process_frame(custom_len_frame)["is_speech"] is False
    assert (
        streamer.process_frame(np.zeros(160, dtype=np.int16))["is_speech"]
        is False
    )
    assert (
        streamer.process_frame(np.zeros(160, dtype=np.int32))["is_speech"]
        is False
    )


@given(st.integers(min_value=50, max_value=800))
@settings(max_examples=25)
def test_property_streaming_chunk_size_invariance(chunk_size: int) -> None:
    """Feeding the same stream in any chunk_size yields identical intervals."""
    from vad_streaming import StreamingVAD

    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    audio = np.zeros(sr, dtype=np.float64)
    audio[int(sr * 0.2) : int(sr * 0.7)] = 0.5 * np.sin(
        2.0 * np.pi * 1000.0 * t[int(sr * 0.2) : int(sr * 0.7)]
    )

    ref_streamer = StreamingVAD(sample_rate=sr, speech_window=0.05)
    ref_streamer.process_chunk(audio)
    ref_streamer.flush()

    chunk_streamer = StreamingVAD(sample_rate=sr, speech_window=0.05)
    for start in range(0, len(audio), chunk_size):
        chunk_streamer.process_chunk(audio[start : start + chunk_size])
    chunk_streamer.flush()

    assert (
        chunk_streamer.completed_intervals == ref_streamer.completed_intervals
    )
