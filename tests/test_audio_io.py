"""Unit and property tests for audio loading and PCM mono conversion."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vad import VoiceActivityDetector


def test_stereo_int16_max_amplitude_no_overflow() -> None:
    """Stereo int16 [32767, 32767] must not overflow during mono downmix."""
    stereo_max = np.array(
        [[32767, 32767], [16384, 16384], [0, 0]], dtype=np.int16
    )
    detector = VoiceActivityDetector(stereo_max, sample_rate=16000)

    assert detector.channels == 1
    assert detector.data.ndim == 1
    assert detector.data.dtype == np.float64
    assert np.isclose(detector.data[0], 32767.0 / 32768.0)
    assert np.isclose(detector.data[1], 16384.0 / 32768.0)
    assert detector.data[2] == 0.0


def test_uint8_dc_centering() -> None:
    """Unsigned 8-bit PCM (uint8) centers 128 at 0.0 and scales to [-1, 1]."""
    raw_uint8 = np.array([0, 128, 255], dtype=np.uint8)
    detector = VoiceActivityDetector(raw_uint8, sample_rate=11025)

    assert detector.data.dtype == np.float64
    assert np.isclose(detector.data[0], -1.0)
    assert detector.data[1] == 0.0
    assert np.isclose(detector.data[2], 127.0 / 128.0)


def test_int32_and_float32_normalization() -> None:
    """Signed 32-bit PCM and float32 arrays normalize cleanly to float64."""
    raw_int32 = np.array([-2147483648, 0, 1073741824], dtype=np.int32)
    det_int32 = VoiceActivityDetector(raw_int32, sample_rate=16000)
    assert np.allclose(det_int32.data, [-1.0, 0.0, 0.5])

    raw_f32 = np.array([-0.5, 0.0, 0.75], dtype=np.float32)
    det_f32 = VoiceActivityDetector(raw_f32, sample_rate=16000)
    assert det_f32.data.dtype == np.float64
    assert np.allclose(det_f32.data, [-0.5, 0.0, 0.75])


def test_multichannel_4ch_to_mono() -> None:
    """Multi-channel (e.g., 4-channel) arrays average cleanly across axis=1."""
    quad = np.array(
        [[0.2, 0.4, 0.6, 0.8], [-0.5, -0.5, 0.5, 0.5]], dtype=np.float64
    )
    detector = VoiceActivityDetector(quad, sample_rate=16000)

    assert detector.channels == 1
    assert detector.data.shape == (2,)
    assert np.allclose(detector.data, [0.5, 0.0])


def test_bytes_input_and_pathlib_input(tmp_path: Path) -> None:
    """Support raw 16-bit PCM bytes and pathlib.Path inputs."""
    samples = [0, 16384, -16384, 32767]
    pcm_bytes = struct.pack(f"<{len(samples)}h", *samples)

    det_bytes = VoiceActivityDetector(pcm_bytes, sample_rate=16000)
    assert det_bytes.rate == 16000
    assert np.allclose(
        det_bytes.data, np.array(samples, dtype=np.float64) / 32768.0
    )

    wav_path = tmp_path / "sample_path.wav"
    with wave.open(str(wav_path), "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(16000)
        wf_out.writeframes(pcm_bytes)

    det_path = VoiceActivityDetector(wav_path)
    assert det_path.rate == 16000
    assert np.allclose(det_path.data, det_bytes.data)


def test_stdlib_wave_fallback(tmp_path: Path) -> None:
    """Fallback to stdlib wave module when scipy.io.wavfile.read fails."""
    samples = [32767, 32767, -16384, -16384]
    pcm_bytes = struct.pack(f"<{len(samples)}h", *samples)
    wav_path = tmp_path / "stereo_fallback.wav"
    with wave.open(str(wav_path), "wb") as wf_out:
        wf_out.setnchannels(2)
        wf_out.setsampwidth(2)
        wf_out.setframerate(8000)
        wf_out.writeframes(pcm_bytes)

    with patch(
        "scipy.io.wavfile.read", side_effect=ValueError("Unknown WAV chunk")
    ):
        detector = VoiceActivityDetector(wav_path)

    assert detector.rate == 8000
    assert detector.channels == 1
    assert np.allclose(detector.data, [32767.0 / 32768.0, -16384.0 / 32768.0])


def test_stdlib_wave_fallback_uint8_and_int32(tmp_path: Path) -> None:
    """Standard-library wave fallback supports 8-bit and 32-bit PCM WAVs."""
    wav_u8 = tmp_path / "u8.wav"
    with wave.open(str(wav_u8), "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(1)
        wf_out.setframerate(11025)
        wf_out.writeframes(bytes([0, 128, 255]))

    wav_i32 = tmp_path / "i32.wav"
    with wave.open(str(wav_i32), "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(4)
        wf_out.setframerate(16000)
        wf_out.writeframes(struct.pack("<2i", -2147483648, 1073741824))

    with patch(
        "scipy.io.wavfile.read", side_effect=ValueError("force fallback")
    ):
        det_u8 = VoiceActivityDetector(wav_u8)
        det_i32 = VoiceActivityDetector(wav_i32)

    assert np.allclose(det_u8.data, [-1.0, 0.0, 127.0 / 128.0])
    assert np.allclose(det_i32.data, [-1.0, 0.5])

    wav_24 = tmp_path / "i24.wav"
    with wave.open(str(wav_24), "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(3)
        wf_out.setframerate(16000)
        wf_out.writeframes(b"\x00\x00\x00")

    with (
        patch(
            "scipy.io.wavfile.read", side_effect=ValueError("force fallback")
        ),
        pytest.raises(ValueError, match="Unsupported WAV sample width"),
    ):
        VoiceActivityDetector(wav_24)


def test_invalid_inputs_raise_errors() -> None:
    """Missing sample_rate or invalid types/shapes raise clear exceptions."""
    with pytest.raises(ValueError, match="sample_rate"):
        VoiceActivityDetector(np.zeros(100, dtype=np.float32))

    with pytest.raises(ValueError, match="sample_rate"):
        VoiceActivityDetector(b"\x00\x00", sample_rate=0)

    with pytest.raises(ValueError, match="even number of bytes"):
        VoiceActivityDetector(b"\x00\x01\x02", sample_rate=16000)

    with pytest.raises(ValueError, match="1D or 2D"):
        VoiceActivityDetector(np.zeros((2, 2, 2)), sample_rate=16000)

    with pytest.raises(TypeError, match="Unsupported audio source"):
        VoiceActivityDetector(12345)  # type: ignore[arg-type]


@given(
    st.lists(
        st.integers(min_value=-32768, max_value=32767),
        min_size=1,
        max_size=200,
    ),
    st.integers(min_value=2, max_value=6),
)
@settings(max_examples=50)
def test_property_identical_multichannel_equals_mono(
    samples: list[int], num_channels: int
) -> None:
    """Duplicating a mono int16 signal across N channels matches mono."""
    mono_arr = np.array(samples, dtype=np.int16)
    multi_arr = np.tile(mono_arr[:, np.newaxis], (1, num_channels))

    det_mono = VoiceActivityDetector(mono_arr, sample_rate=16000)
    det_multi = VoiceActivityDetector(multi_arr, sample_rate=16000)

    assert np.allclose(det_mono.data, det_multi.data)
    assert np.all(det_multi.data >= -1.0)
    assert np.all(det_multi.data <= 1.0)
