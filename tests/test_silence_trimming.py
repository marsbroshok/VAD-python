"""Unit and property tests for extract_speech_segments and strip_silence."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from vad import VoiceActivityDetector


def test_extract_speech_segments_and_strip_silence(tmp_path: Path) -> None:
    """extract_speech_segments and strip_silence isolate speech bursts and write WAV."""
    sr = 16000
    t = np.arange(sr * 3, dtype=np.float64) / sr
    # 3s audio: speech in [0.5s, 1.2s] and [2.0s, 2.7s], silence elsewhere
    audio = np.zeros(sr * 3, dtype=np.float64)
    audio[int(sr * 0.5) : int(sr * 1.2)] = 0.5 * np.sin(
        2.0 * np.pi * 1000.0 * t[int(sr * 0.5) : int(sr * 1.2)]
    )
    audio[int(sr * 2.0) : int(sr * 2.7)] = 0.5 * np.sin(
        2.0 * np.pi * 1200.0 * t[int(sr * 2.0) : int(sr * 2.7)]
    )

    detector = VoiceActivityDetector(audio, sample_rate=sr, speech_window=0.1)
    segments = detector.extract_speech_segments()

    assert len(segments) == 2
    for seg in segments:
        assert isinstance(seg, np.ndarray)
        assert int(sr * 0.6) <= len(seg) <= int(sr * 0.8)

    out_wav = tmp_path / "trimmed.wav"
    trimmed_no_pad = detector.strip_silence(pad_ms=0.0)
    trimmed_padded = detector.strip_silence(output_path=out_wav, pad_ms=20.0)

    assert len(trimmed_no_pad) == sum(len(s) for s in segments)
    assert len(trimmed_padded) > len(trimmed_no_pad)
    assert len(trimmed_padded) < len(audio)
    assert out_wav.exists()

    with wave.open(str(out_wav), "rb") as wf_in:
        assert wf_in.getnchannels() == 1
        assert wf_in.getsampwidth() == 2
        assert wf_in.getframerate() == sr
        assert wf_in.getnframes() == len(trimmed_padded)


def test_strip_silence_on_pure_silence_and_overlapping_padding(tmp_path: Path) -> None:
    """strip_silence handles pure silence and merges overlapping padded intervals."""
    sr = 16000
    silent = np.zeros(sr, dtype=np.float64)
    det_silent = VoiceActivityDetector(silent, sample_rate=sr)

    assert det_silent.extract_speech_segments() == []
    out_silent = tmp_path / "empty.wav"
    trimmed_silent = det_silent.strip_silence(output_path=out_silent)
    assert len(trimmed_silent) == 0
    assert out_silent.exists()

    # Two speech segments separated by a 100ms pause; with pad_ms=100.0 they merge
    t = np.arange(sr * 2, dtype=np.float64) / sr
    audio = np.zeros(sr * 2, dtype=np.float64)
    audio[int(sr * 0.2) : int(sr * 0.7)] = 0.5 * np.sin(
        2.0 * np.pi * 1000.0 * t[int(sr * 0.2) : int(sr * 0.7)]
    )
    audio[int(sr * 0.85) : int(sr * 1.4)] = 0.5 * np.sin(
        2.0 * np.pi * 1000.0 * t[int(sr * 0.85) : int(sr * 1.4)]
    )
    det_close = VoiceActivityDetector(audio, sample_rate=sr, speech_window=0.05)
    unmerged = det_close.extract_speech_segments(pad_ms=0.0)
    merged = det_close.extract_speech_segments(pad_ms=150.0)
    assert len(unmerged) == 2
    assert len(merged) == 1
