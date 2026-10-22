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

## Phase 2: Silence Trimming Helpers & Real-Time `StreamingVAD`
- [~] Task: Implement `extract_speech_segments()` and `strip_silence()` audio helpers
  - [ ] Write Failing Tests (Red Phase): Unit tests for extracting speech numpy segments, stripping silence with configurable padding, and saving trimmed WAV output to disk
  - [ ] Implement to Pass Tests (Green Phase): Add `extract_speech_segments` and `strip_silence` methods to `VoiceActivityDetector`
  - [ ] Refactor & Verify Coverage
- [ ] Task: Implement `StreamingVAD` class in `vad_streaming.py` for real-time frame-by-frame detection
  - [ ] Write Failing Tests (Red Phase): Unit and property tests feeding chunked audio (10ms/20ms/arbitrary chunk sizes) into `StreamingVAD`, verifying state transitions (`is_speech`, speech start/end events) and silence hangover smoothing
  - [ ] Implement to Pass Tests (Green Phase): Create `vad_streaming.py` with `StreamingVAD` class and export it from `vad.py`
  - [ ] Refactor & Verify Coverage
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: CLI Modernization, Documentation & Final Quality Gate
- [ ] Task: Modernize `detectVoiceInWave.py` CLI and console script entrypoint
  - [ ] Write Failing Tests (Red Phase): CLI integration tests testing positional `INPUTWAVE OUTPUTFILE` args, `--threshold`, `--speech-window`, `--output-wav`, and error handling on missing files
  - [ ] Implement to Pass Tests (Green Phase): Enhance `detectVoiceInWave.py` with `main()` entrypoint and new optional CLI flags while keeping full backward compatibility
  - [ ] Refactor & Verify Coverage
- [ ] Task: Revitalize `README.md` and synchronize `test.json`
  - [ ] Write Failing Tests (Red Phase): Documentation & interface parity test verifying CLI `--help` flags and `README.md` code examples execute cleanly
  - [ ] Implement to Pass Tests (Green Phase): Rewrite `README.md` (removing shutdown notice, adding quickstart, streaming & silence-trimming examples, DSP explanation, and realistic comparison) and update `test.json`
  - [ ] Refactor & Verify Coverage
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
