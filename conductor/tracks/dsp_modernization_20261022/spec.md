# Specification: Core DSP Modernization, Ergonomic APIs & Packaging (`dsp_modernization_20261022`)

## 1. Overview
Revive and modernize `VAD-python` from a 2015 student prototype into a fast, well-tested, zero-PyTorch Spectral Voice Activity Detection library and CLI tool. This track resolves all 6 bugs identified in the codebase audit, upgrades the DSP pipeline with Hann windowing, vectorized `rfft`, and an adaptive RMS noise-floor gate ("10 lines of DSP"), adds in-memory/streaming and silence-trimming APIs, and packages the project with `pyproject.toml` and automated tests.

## 2. Functional Requirements

### FR-1: Core DSP Bug Fixes & "10 Lines of DSP" Algorithmic Upgrade
1. **End-of-Audio Speech Interval Closure**: Fix `convert_windows_to_readible_labels()` (and its properly spelled alias `convert_windows_to_readable_labels()` / `detect_intervals()`) so speech segments active at the end of the audio stream are properly closed at the final sample/timestamp instead of being silently dropped. Remove stray `print()` statements from library methods.
2. **Overflow-Safe Stereo/Multi-Channel to Mono Conversion**: Convert multi-channel audio (`ndim == 2`) to mono using `float64`/`float32` accumulation without 16-bit integer wrap-around overflow, and normalize unsigned 8-bit PCM (`uint8`, centered at 128) and signed integer PCM (`int16`, `int32`) consistently.
3. **Safe Division on Digital Silence**: Prevent `0.0 / 0.0` (`NaN` / `RuntimeWarning`) when total window energy is zero by guarding the denominator (`sum_full_energy <= 1e-12 -> ratio = 0.0`).
4. **Accurate Median Filter Window & Short-Clip Safety**: Calculate `median_window` using `sample_overlap` (step size, 10ms) rather than `sample_window` (20ms), ensure odd filter kernel length, and gracefully handle audio clips shorter than `speech_window` without crashing.
5. **Hann Windowing + Vectorized `rfft` + RMS Noise-Floor Gate**:
   - Apply a Hann window (`np.hanning`) to each frame before computing `np.fft.rfft` to suppress spectral leakage.
   - Vectorize frame extraction and `rfft` band-energy ratio computation across frames (while retaining the helper methods for backward compatibility and white-box clarity).
   - Add a configurable RMS energy floor gate (`energy_threshold` / adaptive noise-floor margin, enabled by default) alongside `speech_energy_threshold` (default `0.6` in `300–3000 Hz`) so silent frames or faint mid-band background hiss do not false-trigger as speech.
6. **Headless-Safe Lazy Plotting & Fixed Region Masking**:
   - Lazy-import `matplotlib.pyplot` inside `plot_detected_speech_regions()` so importing `vad` never fails on headless environments without `tkinter`.
   - Fill the full frame slice (`start:end`) in `data_speech` rather than a single sample per step, and plot speech regions visibly on top of the raw waveform.

### FR-2: Flexible Input Formats, Rich Outputs & Silence Trimming
1. **Multi-Format Input**: Allow `VoiceActivityDetector(source, sample_rate=None, ...)` to accept:
   - File paths (`str` or `pathlib.Path`), using `scipy.io.wavfile` with standard-library `wave` fallback for non-standard WAV chunks.
   - In-memory `numpy.ndarray` (1D mono or 2D multi-channel) or raw 16-bit PCM `bytes` when `sample_rate` is provided.
2. **Rich Interval Output**: Return interval dictionaries containing both timestamps in seconds and sample indices:
   `{"speech_begin": float, "speech_end": float, "start_sample": int, "end_sample": int}` (preserving `speech_begin` and `speech_end` keys for 100% backward compatibility).
3. **Audio Slicing & Silence Removal Helpers**:
   - `extract_speech_segments() -> list[np.ndarray]`: Returns a list of audio array slices corresponding to each detected speech interval.
   - `strip_silence(output_path: str | Path | None = None, pad_ms: float = 20.0) -> np.ndarray`: Concatenates detected speech segments (with optional padding) and optionally writes the cleaned WAV file to disk.

### FR-3: Real-Time Frame-by-Frame `StreamingVAD`
1. Provide a modular `StreamingVAD` class in its own module (`vad_streaming.py`, re-exported in `vad.py`) that accepts sequential audio chunks (`process_frame(frame)` / `process_chunk(chunk)`), maintains a ring buffer for median/hangover smoothing, and yields speech state transitions (`speech_begin`, `speech_end`) in real time.

### FR-4: CLI Modernization, Packaging & Documentation
1. **Packaging (`pyproject.toml`)**: Configure PEP 621 metadata, `numpy` core dependency, optional extras (`scipy`, `matplotlib`, `dev`), and console script `vad-python`.
2. **CLI (`detectVoiceInWave.py`)**: Support `--threshold`, `--speech-window`, `--output-wav` (for silence stripping), and `--plot` while keeping positional `INPUTWAVE OUTPUTFILE` backward-compatible.
3. **Documentation (`README.md`)**: Remove the abandonment banner; document quickstart, CLI usage, silence trimming, streaming pre-filtering for Whisper/LLMs, white-box DSP explanation, and honest guidance on when to use spectral DSP vs. neural VADs.

## 3. Non-Functional Requirements
- **Performance**: Vectorized `detect_speech()` must process 1 minute of 16 kHz mono audio in <50 ms on CPU.
- **Test Coverage & Quality**: 100% unit + property-based (`hypothesis`) test coverage on newly authored code; zero `ruff` / `mypy` errors.
- **Backward Compatibility**: Existing tutorial snippets (`from vad import VoiceActivityDetector; v = VoiceActivityDetector('wav-sample.wav'); v.detect_speech(); v.convert_windows_to_readible_labels(...)`) must continue to work without modification.

## 4. Out of Scope
- Bundling heavy deep-learning frameworks (`torch`, `torchaudio`, `tensorflow`) or large pre-trained model weights.
