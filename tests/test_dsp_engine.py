"""Unit, property, and benchmark tests for the vectorized DSP engine."""

from __future__ import annotations

import time
import warnings

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vad import VoiceActivityDetector


def test_digital_silence_zeros_and_dc_no_warnings() -> None:
    """Digital silence (zeros or constant DC) yields 0.0 flags and zero RuntimeWarnings."""
    sr = 16000
    zeros = np.zeros(sr, dtype=np.float64)
    constant_dc = np.full(sr, 0.42, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        det_zeros = VoiceActivityDetector(zeros, sample_rate=sr)
        win_zeros = det_zeros.detect_speech()

        det_dc = VoiceActivityDetector(constant_dc, sample_rate=sr)
        win_dc = det_dc.detect_speech()

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0
    assert win_zeros.shape[0] > 0
    assert np.all(win_zeros[:, 1] == 0.0)
    assert np.all(win_dc[:, 1] == 0.0)


def test_quiet_midband_noise_rejected_by_rms_gate() -> None:
    """Quiet 1000 Hz tone below RMS floor is rejected despite 100% in-band ratio."""
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    quiet_sine = 5e-5 * np.sin(2.0 * np.pi * 1000.0 * t)

    detector = VoiceActivityDetector(quiet_sine, sample_rate=sr, energy_threshold=1e-4)
    windows = detector.detect_speech()
    assert np.all(windows[:, 1] == 0.0)


def test_adaptive_rms_gate_rejects_background_hum() -> None:
    """Adaptive RMS floor rejects faint in-band background hum while preserving loud speech."""
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    signal = 0.005 * np.sin(2.0 * np.pi * 1000.0 * t)
    # Add loud speech-band tone in the second half (1.0s to 2.0s)
    signal[sr:] = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t[sr:])

    detector = VoiceActivityDetector(signal, sample_rate=sr)
    windows = detector.detect_speech()

    first_half = windows[windows[:, 0] < sr * 0.7]
    second_half = windows[windows[:, 0] > sr * 1.3]
    assert np.all(first_half[:, 1] == 0.0)
    assert np.all(second_half[:, 1] == 1.0)


def test_synthetic_voice_band_vs_out_of_band_sine() -> None:
    """1000 Hz sine is detected as speech; 100 Hz and 5000 Hz sines are rejected."""
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr

    in_band = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    low_freq = 0.5 * np.sin(2.0 * np.pi * 100.0 * t)
    high_freq = 0.5 * np.sin(2.0 * np.pi * 5000.0 * t)

    det_in = VoiceActivityDetector(in_band, sample_rate=sr)
    det_low = VoiceActivityDetector(low_freq, sample_rate=sr)
    det_high = VoiceActivityDetector(high_freq, sample_rate=sr)

    assert np.all(det_in.detect_speech()[:, 1] == 1.0)
    assert np.all(det_low.detect_speech()[:, 1] == 0.0)
    assert np.all(det_high.detect_speech()[:, 1] == 0.0)


def test_white_box_helper_methods() -> None:
    """White-box helper methods compute frequencies, Hann-windowed energy, and znorm safely."""
    sr = 16000
    t = np.arange(320, dtype=np.float64) / sr
    frame = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)

    detector = VoiceActivityDetector(frame, sample_rate=sr)
    freqs = detector._calculate_frequencies(frame)
    ampl = detector._calculate_amplitude(frame)
    energy = detector._calculate_energy(frame)
    znorm = detector._znormalize_energy(energy)
    znorm_zero = detector._znormalize_energy(np.zeros(10, dtype=np.float64))
    energy_freq = detector._calculate_normalized_energy(frame)
    band_energy = detector._sum_energy_in_band(energy_freq, 300, 3000)
    total_energy = sum(energy_freq.values())

    assert len(freqs) == len(ampl) == len(energy)
    assert np.isclose(np.mean(znorm), 0.0, atol=1e-7)
    assert np.all(znorm_zero == 0.0)
    assert band_energy / total_energy > 0.95

    # Edge cases: empty, 1-sample, and disabled RMS gate (energy_threshold=0.0)
    assert len(detector._calculate_frequencies(np.array([0.1]))) == 0
    assert len(detector._calculate_amplitude(np.array([0.1]))) == 0
    dup_freq = detector._connect_energy_with_frequencies(
        np.array([100.0, -100.0]), np.array([2.0, 2.0])
    )
    assert dup_freq == {100.0: 4.0}

    det_empty = VoiceActivityDetector(np.array([], dtype=np.float64), sample_rate=sr)
    assert det_empty.detect_speech().shape == (0, 2)

    det_one = VoiceActivityDetector(np.array([0.5], dtype=np.float64), sample_rate=10)
    assert det_one.detect_speech().shape == (1, 2)

    det_no_rms = VoiceActivityDetector(frame, sample_rate=sr, energy_threshold=0.0)
    assert det_no_rms.detect_speech().shape == (1, 2)


def test_performance_benchmark_1min_16khz_under_50ms() -> None:
    """Vectorized detect_speech processes 1 minute of 16 kHz mono audio in <50ms."""
    sr = 16000
    rng = np.random.default_rng(42)
    t = np.arange(sr * 60, dtype=np.float64) / sr
    audio = 0.001 * rng.standard_normal(sr * 60)
    audio[sr * 10 : sr * 25] += 0.4 * np.sin(2.0 * np.pi * 800.0 * t[sr * 10 : sr * 25])
    audio[sr * 35 : sr * 50] += 0.4 * np.sin(2.0 * np.pi * 1200.0 * t[sr * 35 : sr * 50])

    detector = VoiceActivityDetector(audio, sample_rate=sr)
    # Warm up once
    detector.detect_speech()

    start_time = time.perf_counter()
    windows = detector.detect_speech()
    elapsed = time.perf_counter() - start_time

    assert windows.shape[0] >= 5990
    assert elapsed < 0.050, f"Expected <50ms, took {elapsed * 1000:.2f}ms"


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=400,
        max_size=2000,
    )
)
@settings(max_examples=40)
def test_property_detect_speech_output_invariants(samples: list[float]) -> None:
    """detect_speech always returns finite (N, 2) array with binary {0.0, 1.0} flags."""
    arr = np.array(samples, dtype=np.float64)
    detector = VoiceActivityDetector(arr, sample_rate=8000, speech_window=0.05)
    windows = detector.detect_speech()

    assert windows.ndim == 2
    assert windows.shape[1] == 2
    assert np.all(np.isfinite(windows))
    assert set(np.unique(windows[:, 1])).issubset({0.0, 1.0})
