# ADR 0001: Vectorized Hann-Windowed rFFT, PCM Normalization, and Adaptive RMS Gating

- **Status**: Accepted
- **Date**: 2026-10-22

## Context
The original 2015 `VAD-python` implementation operated as a single-frame Python loop computing rectangular-windowed full complex `np.fft.fft`, mapping frequencies through a Python dictionary, and thresholding solely on the `300–3000 Hz` band-energy ratio. Audits identified six architectural defects:
1. Trailing speech active at the end of a recording was silently dropped (`convert_windows_to_readible_labels`).
2. Stereo-to-mono conversion (`np.mean(self.data, axis=1, dtype=self.data.dtype)`) accumulated in `int16`, wrapping high-amplitude peaks (`[32767, 32767] -> -1`), and unsigned 8-bit PCM (`uint8`, such as `wav-sample.wav`) retained a `+128` DC offset.
3. Digital silence (`0.0 / 0.0`) emitted `RuntimeWarning` and `NaN` ratios.
4. Median filter kernel sizing divided `speech_window` by `sample_window` (`20 ms`) instead of `sample_overlap` (`10 ms` step), and crashed on clips shorter than `speech_window`.
5. Pure band-energy ratios without an RMS energy floor caused quiet mid-band background hiss (`300–3000 Hz`) to trigger false speech detections, while rectangular windowing leaked out-of-band low-frequency energy into the speech band.
6. Top-level `import matplotlib.pyplot as plt` broke headless server imports when `tkinter` was absent.

## Decision
1. **Normalized `float64` Internal Audio Representation**:
   All integer PCM inputs (`uint8` centered at `128`, `int16` scaled by `32768.0`, `int32` scaled by `2147483648.0`) are normalized to `float64` in `[-1.0, 1.0]` before channel averaging.
   - *Negative Constraint*: **Do NOT** use `np.mean(data, axis=1, dtype=data.dtype)` on integer PCM arrays; integer accumulation overflows 16-bit signed bounds.
2. **Per-Frame DC Removal + Hann Windowing + Vectorized `rfft`**:
   Extract frames using `np.lib.stride_tricks.sliding_window_view`, subtract per-frame mean, multiply by `np.hanning(frame_len)`, and compute `np.fft.rfft` across all frames in a single vectorized call.
3. **Dual Gate (Band-Energy Ratio + Adaptive RMS Floor)**:
   Guard zero-energy denominators (`full_energy <= 1e-12 -> 0.0`) and require both `speech_ratio > speech_energy_threshold` and `frame_rms > max(energy_threshold, min(2.0 * percentile_10(frame_rms), 0.25 * max(frame_rms)))`.
4. **Modular Separation of Batch (`vad.py`) and Streaming (`vad_streaming.py`)**:
   Keep batch `VoiceActivityDetector` in `vad.py` and real-time `StreamingVAD` in `vad_streaming.py` (re-exported in `vad.py` for ergonomic imports).

## Considered & Rejected Alternatives
- **Z-Normalization of Spectral Energy (`_znormalize_energy` in loop)**:
  Rejected for the primary detection loop because z-normalizing flat noise spectra amplifies random in-band bins above threshold. Retained `_znormalize_energy` (with a zero-variance guard) only as a backward-compatible white-box helper.
- **Uncapped Percentile Noise Floor (`rms_gate = 2.0 * percentile(frame_rms, 10)`)**:
  Rejected because an uncapped percentile threshold fails on recordings containing continuous speech or constant-amplitude test tones (`10th percentile == 100th percentile`). Capping the adaptive floor at `0.25 * max(frame_rms)` preserves continuous speech while still rejecting background hum when speech-to-silence dynamic range is present.
- **Bundling Pre-Trained Neural Weights (PyTorch / ONNX)**:
  Rejected to preserve the zero-ML-dependency, microsecond-latency, white-box DSP niche of `VAD-python`.
