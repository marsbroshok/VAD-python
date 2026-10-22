# AGENTS.md

Instructions and reference for AI coding agents working with `VAD-python`.

## Quick Verification Commands

```bash
uv sync --python 3.13 --extra dev
CI=true uv run pytest --cov --cov-report=term-missing
CI=true uv run ruff check .
CI=true uv run ruff format --check .
CI=true uv run mypy vad.py vad_streaming.py detectVoiceInWave.py tests/
```

## Module Map & Context Pointers

- [`vad.py`](vad.py): Batch `VoiceActivityDetector` class (`detect_intervals`, `detect_speech`, `extract_speech_segments`, `strip_silence`, `plot_detected_speech_regions`).
- [`vad_streaming.py`](vad_streaming.py): Stateful `StreamingVAD` class (`process_chunk`, `process_frame`, `flush`, `reset`) for live audio streams. Re-exported in `vad.py`.
- [`detectVoiceInWave.py`](detectVoiceInWave.py): CLI parser and `vad-python` console script entry point (`main`, `build_parser`, `save_to_file`).
- [`docs/adr/0001-spectral-dsp-vectorization-and-rms-gating.md`](docs/adr/0001-spectral-dsp-vectorization-and-rms-gating.md): DSP design decisions and rejected alternatives.

## Core Invariants

1. **Normalized `float64` Internal Audio**:
   Normalize integer PCM inputs (`uint8` centered at `128`, `int16 / 32768.0`, `int32 / 2147483648.0`) to `float64` in `[-1.0, 1.0]` before averaging channels to mono. This prevents 16-bit integer overflow and keeps `energy_threshold` scale-invariant across bit depths.
2. **Zero Heavy Runtime Dependencies**:
   Keep core detection dependent only on `numpy` and Python standard library `wave` (with optional `scipy.io.wavfile` and lazy-imported `matplotlib.pyplot` inside `plot_detected_speech_regions()`).
3. **Backward Compatibility**:
   Preserve `VoiceActivityDetector(filename)`, `detect_speech()`, and `convert_windows_to_readible_labels()` alongside `convert_windows_to_readable_labels()`, `detect_intervals()`, `extract_speech_segments()`, `strip_silence()`, and `StreamingVAD`.
4. **Code & Test Quality**:
   Keep line length `<= 80` characters (`ruff`), pass `mypy` with zero errors, and maintain 100% statement and branch coverage in `pytest`.
