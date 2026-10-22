"""Real-time frame-by-frame and chunk-based Spectral Voice Activity Detector."""

from __future__ import annotations

from collections import deque

import numpy as np


class StreamingVAD:
    """Real-time spectral Voice Activity Detector with ring-buffer hangover smoothing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        sample_window: float = 0.02,
        sample_overlap: float = 0.01,
        speech_window: float = 0.1,
        speech_energy_threshold: float = 0.6,
        speech_start_band: int = 300,
        speech_end_band: int = 3000,
        energy_threshold: float = 1e-4,
    ) -> None:
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if sample_window <= 0.0 or sample_overlap <= 0.0 or speech_window <= 0.0:
            raise ValueError(
                "sample_window, sample_overlap, and speech_window must be positive."
            )

        self.sample_rate = int(sample_rate)
        self.sample_window = float(sample_window)
        self.sample_overlap = float(sample_overlap)
        self.speech_window = float(speech_window)
        self.speech_energy_threshold = float(speech_energy_threshold)
        self.speech_start_band = int(speech_start_band)
        self.speech_end_band = int(speech_end_band)
        self.energy_threshold = float(energy_threshold)

        self.frame_samples = max(2, int(round(self.sample_rate * self.sample_window)))
        self.step_samples = max(1, int(round(self.sample_rate * self.sample_overlap)))
        history_len = max(1, int(round(self.speech_window / self.sample_overlap)))
        if history_len % 2 == 0:
            history_len += 1
        self.history_len = history_len

        self._hann = np.hanning(self.frame_samples)
        freqs = np.fft.rfftfreq(self.frame_samples, 1.0 / self.sample_rate)[1:]
        self._band_mask = (freqs > self.speech_start_band) & (
            freqs < self.speech_end_band
        )

        self._sample_buffer = np.array([], dtype=np.float64)
        self._buffer_start_sample = 0
        self._history: deque[int] = deque(maxlen=self.history_len)
        self.is_speech = False
        self._active_start_sample: int | None = None
        self.completed_intervals: list[dict[str, float | int]] = []

    def reset(self) -> None:
        """Reset all internal streaming buffers and speech state."""
        self._sample_buffer = np.array([], dtype=np.float64)
        self._buffer_start_sample = 0
        self._history.clear()
        self.is_speech = False
        self._active_start_sample = None
        self.completed_intervals.clear()

    @staticmethod
    def _normalize_chunk(
        chunk: np.ndarray | bytes | bytearray | memoryview,
    ) -> np.ndarray:
        """Normalize incoming audio chunk to 1D float64 mono samples."""
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            if len(chunk) % 2 != 0:
                raise ValueError(
                    "Raw 16-bit PCM byte buffer must contain an even number of bytes."
                )
            arr: np.ndarray = np.frombuffer(chunk, dtype="<i2").astype(np.float64) / 32768.0
        elif isinstance(chunk, np.ndarray):
            if chunk.dtype == np.uint8:
                arr = (chunk.astype(np.float64) - 128.0) / 128.0
            elif chunk.dtype == np.int16:
                arr = chunk.astype(np.float64) / 32768.0
            elif chunk.dtype == np.int32:
                arr = chunk.astype(np.float64) / 2147483648.0
            else:
                arr = chunk.astype(np.float64)
        else:
            raise TypeError(f"Unsupported audio chunk type: {type(chunk)!r}")

        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return np.mean(arr, axis=1, dtype=np.float64)
        raise ValueError(f"Audio chunk must be 1D or 2D array, got {arr.ndim}D.")

    def _evaluate_frame_array(
        self, frame: np.ndarray, sample_index: int
    ) -> dict[str, float | int | bool | str | None | dict[str, float | int]]:
        """Evaluate a 1D mono float64 frame and update ring-buffer speech state."""
        n = len(frame)
        if n <= 1:
            rms = 0.0
            speech_ratio = 0.0
        else:
            frame_ac = frame - np.mean(frame)
            rms = float(np.sqrt(np.mean(frame_ac**2)))
            if n == self.frame_samples:
                hann = self._hann
                band_mask = self._band_mask
            else:
                hann = np.hanning(n)
                freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)[1:]
                band_mask = (freqs > self.speech_start_band) & (
                    freqs < self.speech_end_band
                )
            spectrum = np.abs(np.fft.rfft(frame_ac * hann))[1:] ** 2
            full_energy = float(np.sum(spectrum))
            if full_energy <= 1e-12:
                speech_ratio = 0.0
            else:
                voice_energy = float(np.sum(spectrum[band_mask]))
                speech_ratio = voice_energy / full_energy

        raw_is_speech = bool(
            speech_ratio > self.speech_energy_threshold and rms > self.energy_threshold
        )
        self._history.append(1 if raw_is_speech else 0)
        smoothed_speech = sum(self._history) > (len(self._history) // 2)

        event: str | None = None
        closed_interval: dict[str, float | int] | None = None

        if smoothed_speech and not self.is_speech:
            self.is_speech = True
            self._active_start_sample = sample_index
            event = "speech_begin"
        elif not smoothed_speech and self.is_speech:
            self.is_speech = False
            start_sample = (
                self._active_start_sample
                if self._active_start_sample is not None
                else sample_index
            )
            end_sample = sample_index
            closed_interval = {
                "speech_begin": float(start_sample / self.sample_rate),
                "speech_end": float(end_sample / self.sample_rate),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
            }
            self.completed_intervals.append(closed_interval)
            self._active_start_sample = None
            event = "speech_end"

        return {
            "is_speech": self.is_speech,
            "raw_is_speech": raw_is_speech,
            "speech_ratio": speech_ratio,
            "rms": rms,
            "event": event,
            "timestamp": float(sample_index / self.sample_rate),
            "sample_index": sample_index,
            "interval": closed_interval,
        }

    def process_frame(
        self, frame: np.ndarray | bytes | bytearray | memoryview
    ) -> dict[str, float | int | bool | str | None | dict[str, float | int]]:
        """Process a single audio frame immediately and return state diagnostics."""
        mono = self._normalize_chunk(frame)
        sample_index = self._buffer_start_sample
        self._buffer_start_sample += max(1, len(mono))
        return self._evaluate_frame_array(mono, sample_index)

    def process_chunk(
        self, chunk: np.ndarray | bytes | bytearray | memoryview
    ) -> list[dict[str, float | int | str]]:
        """Append an arbitrary-length audio chunk and return any transition events."""
        mono = self._normalize_chunk(chunk)
        if len(mono) == 0:
            return []

        if len(self._sample_buffer) == 0:
            self._sample_buffer = mono.copy()
        else:
            self._sample_buffer = np.concatenate((self._sample_buffer, mono))

        events: list[dict[str, float | int | str]] = []
        while len(self._sample_buffer) >= self.frame_samples:
            window = self._sample_buffer[: self.frame_samples]
            win_start = self._buffer_start_sample
            self._sample_buffer = self._sample_buffer[self.step_samples :]
            self._buffer_start_sample += self.step_samples

            res = self._evaluate_frame_array(window, win_start)
            if res["event"] == "speech_begin":
                events.append(
                    {
                        "event": "speech_begin",
                        "speech_begin": float(win_start / self.sample_rate),
                        "start_sample": win_start,
                    }
                )
            elif res["event"] == "speech_end" and isinstance(res["interval"], dict):
                interval_event: dict[str, float | int | str] = {"event": "speech_end"}
                interval_event.update(res["interval"])
                events.append(interval_event)

        return events

    def flush(self) -> dict[str, float | int] | None:
        """Close and return any active speech interval at the end of the audio stream."""
        total_samples = self._buffer_start_sample + len(self._sample_buffer)
        self._sample_buffer = np.array([], dtype=np.float64)
        self._buffer_start_sample = total_samples

        if not self.is_speech:
            return None

        start_sample = (
            self._active_start_sample
            if self._active_start_sample is not None
            else total_samples
        )
        end_sample = max(total_samples, start_sample + self.step_samples)
        closed_interval: dict[str, float | int] = {
            "speech_begin": float(start_sample / self.sample_rate),
            "speech_end": float(end_sample / self.sample_rate),
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
        }
        self.completed_intervals.append(closed_interval)
        self.is_speech = False
        self._active_start_sample = None
        return closed_interval
