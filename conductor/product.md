# Product Definition: VAD-python

## Vision & Overview
`VAD-python` is a lightweight, explainable Spectral Voice Activity Detector (VAD) library and CLI toolkit written in pure Python. It detects human speech segments by combining spectral band-energy ratios (300–3000 Hz) with an adaptive RMS noise-floor gate, Hann windowing, and median-filter smoothing.

## Target Users & Core Use Cases
1. **ASR & Voice-Agent Pre-filtering (Stage-1 Audio Gate)**: Developers building speech pipelines (Whisper, LLM voice interfaces, edge/IoT voice triggers) who need a fast, zero-PyTorch pre-filter to strip silence and prevent downstream hallucinations or unnecessary API calls.
2. **Clean Audio Silence Trimming & Splitting**: Podcasters, researchers, and dataset curators splitting recordings into speech intervals or stripping dead air from WAV/audio files in milliseconds.
3. **DSP Education & White-Box Reference**: Students and engineers learning speech processing who want a clean, readable, math-first implementation of classical spectral VAD rather than an opaque binary or neural network weight file.

## Core Capabilities & Invariants
- **Backward Compatibility**: Preserve the classic `VoiceActivityDetector` interface (`detect_speech()`, `convert_windows_to_readible_labels()`, `plot_detected_speech_regions()`) cited in existing external tutorials while providing clean modern APIs (`detect_intervals()`, `convert_windows_to_readable_labels()`, `extract_speech_segments()`, `strip_silence()`, and `StreamingVAD`).
- **Zero Heavy ML Dependencies**: The core detector relies strictly on `numpy` (and standard library `wave` / `scipy`) with zero model weight downloads and microsecond-level execution.
- **Headless-Safe Imports**: `matplotlib` is lazy-loaded only when plotting is explicitly invoked, ensuring CLI and server deployments never fail due to missing GUI backends (`tkinter`).
