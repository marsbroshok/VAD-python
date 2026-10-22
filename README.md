# VAD-python: Spectral Voice Activity Detector

`VAD-python` is a lightweight, explainable Spectral Voice Activity Detector (VAD) library and CLI toolkit written in pure Python. It detects human speech segments in microseconds using **Hann-windowed rFFT band-energy ratios (300–3000 Hz)**, an **adaptive RMS noise-floor gate**, and **median-filter hangover smoothing**—with **zero PyTorch/ONNX dependencies** and **zero model weight downloads**.

---

## Features

- **Zero Heavy ML Dependencies**: Relies solely on `numpy` (and standard-library `wave` / optional `scipy`), making it ideal for serverless functions, edge/IoT devices, and Stage-1 audio pre-filtering before Whisper or LLM voice APIs.
- **Fast Vectorized DSP**: Processes 1 minute of 16 kHz mono audio in **<15 ms** on CPU using strided NumPy frame views and vectorized `rfft`.
- **Flexible Audio Inputs**: Accepts WAV file paths (`str` or `pathlib.Path`), in-memory `numpy.ndarray` buffers (1D mono or 2D multi-channel `uint8`, `int16`, `int32`, `float32`, `float64`), or raw 16-bit PCM `bytes`.
- **Silence Trimming & Segment Extraction**: Slice speech arrays with `extract_speech_segments()` or export silence-stripped WAV files with `strip_silence()`.
- **Real-Time Streaming (`StreamingVAD`)**: Feed sequential audio frames (`process_frame`) or arbitrary-sized streaming chunks (`process_chunk`) with causal ring-buffer smoothing.
- **Headless-Safe Visualization**: `matplotlib` is lazy-loaded only when `plot_detected_speech_regions()` is called—never crashing headless servers without `tkinter`.

---

## Installation

```bash
# Core library + CLI (requires only numpy)
pip install .

# With optional scipy WAV reader and matplotlib plotting support
pip install ".[scipy,matplotlib]"

# Development & test suite
pip install ".[dev]"
```

---

## Quickstart

### 1. Detect Speech Intervals (File or NumPy Array)

```python
from vad import VoiceActivityDetector

# Load from WAV path (or pass a numpy.ndarray with sample_rate=16000)
v = VoiceActivityDetector("wav-sample.wav")

# High-level ergonomic API returning timestamps (seconds) and sample indices
intervals = v.detect_intervals()
print(intervals)
# [
#   {
#     "speech_begin": 0.11972789115646258,
#     "speech_end": 1.9256235827664399,
#     "start_sample": 1320,
#     "end_sample": 21230
#   }
# ]

# Classic API (100% backward-compatible with existing tutorials)
raw_windows = v.detect_speech()
labels = v.convert_windows_to_readable_labels(raw_windows)
```

### 2. Strip Silence or Extract Speech Segments for Whisper / ASR

```python
from vad import VoiceActivityDetector

v = VoiceActivityDetector("wav-sample.wav")

# Extract individual speech segments as a list of 1D float64 numpy arrays
segments = v.extract_speech_segments(pad_ms=20.0)

# Concatenate all speech and optionally save a cleaned 16-bit PCM WAV file
cleaned_audio = v.strip_silence(output_path="cleaned.wav", pad_ms=20.0)
```

### 3. Real-Time Streaming VAD (`StreamingVAD`)

```python
import numpy as np
from vad import StreamingVAD

streamer = StreamingVAD(sample_rate=16000, speech_window=0.1)

# Feed arbitrary-sized audio chunks (numpy arrays or 16-bit PCM bytes)
chunk = np.zeros(1600, dtype=np.float32)
events = streamer.process_chunk(chunk)
for event in events:
    print(
        event
    )  # {"event": "speech_begin", ...} or {"event": "speech_end", ...}

# Or inspect frame-by-frame diagnostics
frame_status = streamer.process_frame(chunk[:320])
print(
    frame_status["is_speech"], frame_status["speech_ratio"], frame_status["rms"]
)

# Flush any active speech interval when the stream ends
final_interval = streamer.flush()
```

### 4. Plot Detected Speech Regions

```python
from vad import VoiceActivityDetector

v = VoiceActivityDetector("wav-sample.wav")
v.plot_detected_speech_regions()
```

---

## Command-Line Interface (`detectVoiceInWave.py` / `vad-python`)

Analyze a WAV file, write speech intervals to JSON, and optionally strip silence into a new WAV file:

```bash
# Classic positional usage
python detectVoiceInWave.py wav-sample.wav results.json

# Or via the installed console script with custom tuning and silence stripping
vad-python wav-sample.wav results.json \
  --threshold 0.6 \
  --energy-threshold 0.0001 \
  --speech-window 0.5 \
  --output-wav cleaned.wav \
  --pad-ms 20.0 \
  --plot
```

---

## How the White-Box DSP Pipeline Works

1. **PCM Normalization & Overflow-Safe Mono Downmix**:
   Converts unsigned 8-bit (`uint8`, centered at `128`) and signed integer PCM (`int16`, `int32`) to normalized `float64` in `[-1.0, 1.0]` before averaging multi-channel audio, eliminating 16-bit integer overflow.
2. **Sliding Frame Extraction & DC Removal**:
   Slices overlapping `20 ms` frames (`sample_window=0.02`) stepped every `10 ms` (`sample_overlap=0.01`) and subtracts each frame's DC mean.
3. **Hann Windowing & Real FFT (`rfft`)**:
   Applies a Hann window ($\text{hann}(N)$) to suppress spectral sidelobe leakage and computes positive frequency power spectra via vectorized `np.fft.rfft`.
4. **Dual Gate (Band-Energy Ratio + Adaptive RMS Floor)**:
   Computes the ratio of voice-band energy (`300–3000 Hz`) to total spectral energy (guarded against `0.0 / 0.0` on digital silence) and combines `speech_energy_threshold` (`0.6`) with an RMS energy gate (`energy_threshold=1e-4` and adaptive noise-floor margin).
5. **Median Filter Hangover Smoothing**:
   Applies an odd-kernel 1D median filter across `speech_window` (`0.5 s` / `49` frames at `10 ms` step) to bridge brief plosive gaps and suppress isolated transient clicks.

---

## When to Use Spectral DSP vs. Neural VAD

| Use Case | Recommended Approach |
| :--- | :--- |
| **Clean / Podcast / Studio Audio Silence Trimming** | **`VAD-python` (Spectral DSP)** — microsecond execution, deterministic, zero dependencies. |
| **Stage-1 Gate Before Whisper / Cloud ASR** | **`VAD-python` (Spectral DSP)** — fast pre-filter to drop dead air and silent chunks before GPU inference. |
| **DSP Education & Embedded / Constrained Python** | **`VAD-python` (Spectral DSP)** — ~250 lines of readable NumPy math; no C++/PyTorch runtime. |
| **Heavy Non-Stationary Mid-Band Noise (Music, Cafeteria, Siren)** | **Neural VAD (e.g., Silero VAD / WebRTC)** — acoustic classifiers distinguish harmonic vocal formants from non-vocal 300–3000 Hz noise. |

---

## License

Alexander Usoltsev (c) 2015–2026 [MIT License](https://opensource.org/licenses/MIT)
