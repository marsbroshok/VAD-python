# Voice Activity Detector (VAD-python)

Python code and CLI tool to detect voice activity in WAV files or audio streams.

The detector measures the ratio between energy in the human speech band (`300–3000 Hz`) and total energy, combined with an RMS volume floor and a `0.5s` median filter.

## Why use a Spectral VAD instead of Deep Learning?

Neural VADs (like Silero or Pyannote) are great for noisy street audio, but this classic spectral approach is still useful when you need:

1. **Stage-1 Gate for Whisper & Voice Agents**:
   Streaming audio 24/7 into a neural model or cloud API wastes CPU, battery, and money. Moreover, Whisper often hallucinates when you feed it long silence. You can run this detector first at almost zero CPU cost to drop 90% of dead air and low-frequency rumble (AC hum, mic noise, wind `< 200 Hz`), and only wake up Stage 2 (Whisper / LLM) when speech energy appears.
2. **Zero Heavy Dependencies & Instant Start**:
   Installing `torch` or `onnxruntime` in a small CLI tool, AWS Lambda, or Docker container adds hundreds of megabytes of C++ binaries. This code uses only `numpy`, needs `0 MB` of model weights, and starts immediately on any machine.
3. **Splitting Clean Audio (Podcasts, TTS, Zoom, Datasets)**:
   If you just want to trim silence from studio recordings, TTS output, or interview WAV files, a neural network is overkill. Spectral energy + RMS gating processes a 1-hour file in milliseconds.
4. **Learning How DSP Works (White-Box Code)**:
   If you want to understand how Voice Activity Detection works in Python, a black-box `.onnx` weight file does not help. Here you can read ~250 lines of math, change the frequency band or window size, and plot the result.

## Requirements

Core detection only needs:
* `numpy`

Optional packages:
* `scipy` (faster WAV reading; falls back to Python standard `wave` module if not installed)
* `matplotlib` (only needed if you call `plot_detected_speech_regions()`)

Install with pip:

```bash
pip install .
# Or with optional plotting and scipy support:
pip install ".[scipy,matplotlib]"
```

## Basic Idea

Input audio data is processed in 6 steps:

1. Convert stereo (or multi-channel) audio to mono and normalize PCM values to `[-1.0, 1.0]`.
2. Move a window of `20ms` (`sample_window=0.02`) along the audio data with a `10ms` step (`sample_overlap=0.01`).
3. Apply a Hann window and real FFT (`rfft`) to get frequencies and energy for each window.
4. Check two simple conditions:
   - The window RMS volume is above the silence floor (`energy_threshold=1e-4`).
   - The ratio of energy in the speech band (`300–3000 Hz`) to total energy is above the threshold (`0.6` by default).
5. Apply a median filter with length of `0.5s` (`speech_window=0.5`) to smooth out short clicks or pauses.
6. Return speech regions as time intervals, audio arrays, or a silence-stripped WAV file.

## Examples

### 1. Get speech timestamps from a WAV file

```python
from vad import VoiceActivityDetector

v = VoiceActivityDetector("wav-sample.wav")
intervals = v.detect_intervals()
print(intervals)
# [{'speech_begin': 0.12, 'speech_end': 1.93, 'start_sample': 1320, 'end_sample': 21230}]
```

*(Note: the older methods `v.detect_speech()` and `v.convert_windows_to_readable_labels()` / `v.convert_windows_to_readible_labels()` still work the same way).*

### 2. Trim silence from a podcast or recording before Whisper

Remove dead air so ASR models do not hallucinate on silence, or split a recording into separate speech chunks:

```python
from vad import VoiceActivityDetector

v = VoiceActivityDetector("wav-sample.wav")

# Save a new WAV file with all silence removed (keeps 20ms padding around words)
clean_audio = v.strip_silence(output_path="cleaned.wav", pad_ms=20.0)

# Or get a list of numpy arrays, one for each spoken phrase:
segments = v.extract_speech_segments(pad_ms=20.0)
for i, chunk in enumerate(segments):
    print(f"Segment {i}: {len(chunk)} samples")
```

### 3. Stage-1 live microphone / stream gate (`StreamingVAD`)

Filter a live audio stream frame-by-frame or chunk-by-chunk before sending audio to a heavier model:

```python
import numpy as np
from vad import StreamingVAD

vad_gate = StreamingVAD(sample_rate=16000, speech_window=0.1)

# Feed audio chunks of any size (numpy array or 16-bit PCM bytes)
audio_chunk = np.zeros(1600, dtype=np.float32)  # 100ms chunk at 16kHz
events = vad_gate.process_chunk(audio_chunk)

for event in events:
    if event["event"] == "speech_begin":
        print(f"Voice started at {event['speech_begin']:.2f}s -> wake up ASR")
    elif event["event"] == "speech_end":
        print(f"Voice ended at {event['speech_end']:.2f}s")

# Or check a single 20ms frame directly:
status = vad_gate.process_frame(audio_chunk[:320])
if status["is_speech"]:
    pass  # forward frame to Stage 2
```

### 4. Inspect and plot the DSP detection (Education / Debugging)

```python
from vad import VoiceActivityDetector

v = VoiceActivityDetector(
    "wav-sample.wav",
    speech_energy_threshold=0.6,  # 60% energy in voice band
    speech_start_band=300,  # Hz
    speech_end_band=3000,  # Hz
    speech_window=0.5,  # 0.5s median smoothing
)
v.plot_detected_speech_regions()
```

## Command Line Usage

Save detected speech intervals to a JSON file:

```bash
./detectVoiceInWave.py ./wav-sample.wav ./results.json
```

Or use the optional flags to tune thresholds, strip silence into a new WAV file, or plot the wave:

```bash
vad-python ./wav-sample.wav ./results.json \
  --threshold 0.6 \
  --speech-window 0.5 \
  --output-wav ./cleaned.wav \
  --plot
```

-------
Alexander USOLTSEV 2015–2026 (c) [MIT License](https://opensource.org/licenses/MIT)
