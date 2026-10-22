# Architecture Overview: VAD-python

## Module Structure

- **`vad.py` (`VoiceActivityDetector`)**:
  Batch spectral Voice Activity Detector, audio loader/normalizer (`str`, `Path`, `np.ndarray`, `bytes`), interval converter, silence trimmer (`extract_speech_segments`, `strip_silence`), and lazy matplotlib visualizer (`plot_detected_speech_regions`). Re-exports `StreamingVAD`.
- **`vad_streaming.py` (`StreamingVAD`)**:
  Stateful, real-time frame-by-frame (`process_frame`) and chunk-based (`process_chunk`) Voice Activity Detector using a causal odd-length ring buffer (`collections.deque`) for median hangover smoothing.
- **`detectVoiceInWave.py` (`main`, `build_parser`, `save_to_file`)**:
  Command-line interface and `vad-python` console script entry point supporting JSON interval export, silence-stripped WAV output (`--output-wav`), threshold tuning, and optional waveform plotting (`--plot`).

## Data Flow

```
Audio Input (WAV Path | np.ndarray | PCM bytes)
       │
       ▼
PCM Normalization (uint8 - 128, int16 / 32768, int32 / 2^31 -> float64 [-1, 1])
       │
       ▼
Overflow-Safe Mono Downmix (np.mean across channels in float64)
       │
       ▼
Frame Extraction (20 ms window, 10 ms step) & Per-Frame DC Removal
       │
       ▼
Hann Windowing + Real FFT (np.fft.rfft) + Time-Domain Frame RMS
       │
       ▼
Dual Gate: (300-3000 Hz Energy Ratio > threshold) AND (RMS > noise_floor)
       │
       ▼
1D Median Filter Smoothing (speech_window / sample_overlap frames)
       │
       ▼
Outputs: Rich Interval Dicts | Speech Array Segments | Silence-Stripped WAV
```
