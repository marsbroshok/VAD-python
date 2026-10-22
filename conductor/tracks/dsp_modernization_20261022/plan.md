# Implementation Plan: Core DSP Modernization, Ergonomic APIs & Packaging (`dsp_modernization_20261022`)

## Phase 1: Project Scaffolding, Core DSP Bug Fixes & "10 Lines of DSP" Vectorization [checkpoint: 82005af]
- [x] Task: Configure `pyproject.toml`, `.gitignore`, and test harness (`pytest`, `pytest-cov`, `hypothesis`) 25cd536
  - [x] Write initial smoke test verifying test runner and `wav-sample.wav` fixture loading
  - [x] Create `pyproject.toml` and update `.gitignore` for Python build/test artifacts
  - [x] Verify `pytest` execution and baseline coverage report
- [x] Task: Fix audio loading, `uint8`/`int16` normalization, and stereo-to-mono integer overflow (`_read_wav`, `_convert_to_mono`) 540229f
  - [x] Write Failing Tests (Red Phase): Unit and Hypothesis property tests for stereo `int16` max-amplitude overflow (`[32767, 32767]`), `uint8` DC centering, multi-channel arrays, and `np.ndarray` / `bytes` / `Path` inputs
  - [x] Implement to Pass Tests (Green Phase): Refactor `_read_wav` and `_convert_to_mono` to support file paths (with `wave` fallback), `np.ndarray`, and `bytes`, converting safely without integer overflow
  - [x] Refactor & Verify Coverage
- [x] Task: Upgrade DSP engine with Hann windowing, vectorized `rfft`, zero-division guard, and RMS noise-floor gate d450bc5
  - [x] Write Failing Tests (Red Phase): Tests for digital silence (all zeros / constant DC producing `0.0` ratio with zero `RuntimeWarning`s), quiet mid-band noise rejection via RMS gate, 300–3000 Hz synthetic sine wave detection, and performance benchmark (<50ms per minute of audio)
  - [x] Implement to Pass Tests (Green Phase): Implement Hann windowing, vectorized `rfft` band-energy ratio, safe denominator guard, and RMS noise-floor gating in `VoiceActivityDetector`
  - [x] Refactor & Verify Coverage
- [x] Task: Fix median filter step calculation, short-clip safety, end-of-audio interval closure, and lazy plotting 82005af
  - [x] Write Failing Tests (Red Phase): Tests for speech continuing until the final sample of the audio (Issues #4 & #6), audio shorter than `speech_window`, sample index inclusion (`start_sample`, `end_sample`), `convert_windows_to_readable_labels` alias, and headless import without `tkinter` + mocked `plot_detected_speech_regions()`
  - [x] Implement to Pass Tests (Green Phase): Fix `_smooth_speech_detection`, `_median_filter`, `convert_windows_to_readible_labels`, `detect_intervals`, and lazy-loaded `plot_detected_speech_regions`
  - [x] Refactor & Verify Coverage
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) 82005af

## Phase 2: Silence Trimming Helpers & Real-Time `StreamingVAD` [checkpoint: 6b14568]
- [x] Task: Implement `extract_speech_segments()` and `strip_silence()` audio helpers 441f436
  - [x] Write Failing Tests (Red Phase): Unit tests for extracting speech numpy segments, stripping silence with configurable padding, and saving trimmed WAV output to disk
  - [x] Implement to Pass Tests (Green Phase): Add `extract_speech_segments` and `strip_silence` methods to `VoiceActivityDetector`
  - [x] Refactor & Verify Coverage
- [x] Task: Implement `StreamingVAD` class in `vad_streaming.py` for real-time frame-by-frame detection 6b14568
  - [x] Write Failing Tests (Red Phase): Unit and property tests feeding chunked audio (10ms/20ms/arbitrary chunk sizes) into `StreamingVAD`, verifying state transitions (`is_speech`, speech start/end events) and silence hangover smoothing
  - [x] Implement to Pass Tests (Green Phase): Create `vad_streaming.py` with `StreamingVAD` class and export it from `vad.py`
  - [x] Refactor & Verify Coverage
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) 6b14568

## Phase 3: CLI Modernization, Documentation & Final Quality Gate [checkpoint: c7dd173]
- [x] Task: Modernize `detectVoiceInWave.py` CLI and console script entrypoint 3d1b13f
  - [x] Write Failing Tests (Red Phase): CLI integration tests testing positional `INPUTWAVE OUTPUTFILE` args, `--threshold`, `--speech-window`, `--output-wav`, and error handling on missing files
  - [x] Implement to Pass Tests (Green Phase): Enhance `detectVoiceInWave.py` with `main()` entrypoint and new optional CLI flags while keeping full backward compatibility
  - [x] Refactor & Verify Coverage
- [x] Task: Revitalize `README.md` and synchronize `test.json` c7dd173
  - [x] Write Failing Tests (Red Phase): Documentation & interface parity test verifying CLI `--help` flags and `README.md` code examples execute cleanly
  - [x] Implement to Pass Tests (Green Phase): Rewrite `README.md` (removing shutdown notice, adding quickstart, streaming & silence-trimming examples, DSP explanation, and realistic comparison) and update `test.json`
  - [x] Refactor & Verify Coverage
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) c7dd173

## Assumptions & Domain Invariants Ledger
- **PCM Normalization Before Mono Downmix**: Unsigned 8-bit (`uint8`, centered at `128`), signed 16-bit (`int16`, scaled by `32768.0`), and signed 32-bit (`int32`, scaled by `2147483648.0`) inputs are converted to `float64` in `[-1.0, 1.0]` prior to channel averaging so stereo/multi-channel downmixing never overflows 16-bit integer bounds and RMS thresholds remain bit-depth invariant.
- **Per-Frame DC Mean Removal**: Each frame's DC mean is subtracted before applying the Hann window (`np.hanning`) so residual DC offsets do not leak energy into low-frequency FFT bins or inflate frame RMS on constant DC signals.
- **Capped Adaptive RMS Noise Floor**: The effective RMS gate is `max(energy_threshold, min(2.0 * percentile_10(frame_rms), 0.25 * max(frame_rms)))` when `energy_threshold > 0`, ensuring quiet mid-band background hum is rejected in dynamic recordings while continuous speech or constant-amplitude test tones are never suppressed.
- **Trailing Speech Closure**: Speech segments still active at the final window close at `len(self.data)` (or `last_win_start + step_samples` for synthetic window arrays exceeding `len(self.data)`).
- **Padding Merge on Silence Trimming**: When `pad_ms > 0` in `extract_speech_segments()` or `strip_silence()`, overlapping or adjacent padded intervals are merged so audio samples are never duplicated.
