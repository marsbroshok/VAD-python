# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-22

### Added
- `StreamingVAD` class in `vad_streaming.py` (re-exported in `vad.py`) for real-time frame-by-frame (`process_frame`) and arbitrary-chunk (`process_chunk`) voice activity detection with ring-buffer hangover smoothing.
- `VoiceActivityDetector.detect_intervals()` returning rich interval dictionaries with both timestamps (`speech_begin`, `speech_end`) and sample indices (`start_sample`, `end_sample`).
- `VoiceActivityDetector.convert_windows_to_readable_labels()` properly spelled alias alongside `convert_windows_to_readible_labels()`.
- `VoiceActivityDetector.extract_speech_segments()` and `VoiceActivityDetector.strip_silence()` for slicing speech segments and exporting silence-trimmed WAV files.
- Support for `pathlib.Path`, in-memory `numpy.ndarray`, and raw 16-bit PCM `bytes` audio inputs, plus a standard-library `wave` fallback when `scipy` encounters non-standard WAV header chunks.
- Configurable RMS noise-floor gate (`energy_threshold` and adaptive 10th-percentile floor) and Hann windowing (`np.hanning`).
- PEP 621 `pyproject.toml` packaging with `vad-python` console script entry point and CLI flags (`--threshold`, `--energy-threshold`, `--speech-window`, `--output-wav`, `--pad-ms`, `--plot`).
- Automated unit, CLI, documentation parity, and Hypothesis property-based test suite (`tests/`).

### Changed
- Vectorized `VoiceActivityDetector.detect_speech()` using `sliding_window_view` and `np.fft.rfft`, accelerating 1 minute of 16 kHz audio processing to <15 ms on CPU.
- Lazy-loaded `matplotlib.pyplot` inside `plot_detected_speech_regions()` so headless imports never require `tkinter`.

### Fixed
- Closed trailing speech segments active at the end of the audio stream in `convert_windows_to_readible_labels()` and removed stray debug `print()` output.
- Fixed 16-bit signed integer overflow during stereo-to-mono conversion and centered unsigned 8-bit (`uint8`) PCM around `128`.
- Guarded zero-energy frames against `0.0 / 0.0` division (`RuntimeWarning` / `NaN`) on digital silence and constant DC inputs.
- Fixed median filter window size calculation to use `sample_overlap` (`10 ms` step) rather than `sample_window` (`20 ms`) and added short-clip bounds clamping.
- Fixed `plot_detected_speech_regions()` to fill full speech frame slices and overlay detected speech on top of the raw waveform.
