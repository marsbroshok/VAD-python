from __future__ import annotations

import importlib
import wave
from pathlib import Path

import numpy as np

from vad_streaming import StreamingVAD

try:
    import scipy.io.wavfile as wf
except ImportError:  # pragma: no cover
    wf = None  # type: ignore[assignment]

__all__ = ["VoiceActivityDetector", "StreamingVAD"]


class VoiceActivityDetector:
    """Use spectral band-energy ratio and RMS gating to detect voice."""

    def __init__(
        self,
        wave_input_filename: str
        | Path
        | np.ndarray
        | bytes
        | bytearray
        | memoryview,
        sample_rate: int | None = None,
        sample_window: float = 0.02,
        sample_overlap: float = 0.01,
        speech_window: float = 0.5,
        speech_energy_threshold: float = 0.6,
        speech_start_band: int = 300,
        speech_end_band: int = 3000,
        energy_threshold: float = 1e-4,
    ) -> None:
        self.sample_window = float(sample_window)
        self.sample_overlap = float(sample_overlap)
        self.speech_window = float(speech_window)
        self.speech_energy_threshold = float(speech_energy_threshold)
        self.speech_start_band = int(speech_start_band)
        self.speech_end_band = int(speech_end_band)
        self.energy_threshold = float(energy_threshold)
        self._read_wav(
            wave_input_filename, sample_rate=sample_rate
        )._convert_to_mono()

    @staticmethod
    def _read_wav_stdlib(file_path: str) -> tuple[int, np.ndarray]:
        """Read a PCM WAV file using the Python standard-library wave module."""
        with wave.open(file_path, "rb") as wav_in:
            n_channels = wav_in.getnchannels()
            sampwidth = wav_in.getsampwidth()
            rate = wav_in.getframerate()
            n_frames = wav_in.getnframes()
            raw_bytes = wav_in.readframes(n_frames)

        if sampwidth == 1:
            data: np.ndarray = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
        elif sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype="<i2").copy()
        elif sampwidth == 4:
            data = np.frombuffer(raw_bytes, dtype="<i4").copy()
        else:
            raise ValueError(
                f"Unsupported WAV sample width: {sampwidth} bytes."
            )

        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        return int(rate), data

    def _read_wav(
        self,
        wave_file: str | Path | np.ndarray | bytes | bytearray | memoryview,
        sample_rate: int | None = None,
    ) -> VoiceActivityDetector:
        """Load audio from a file path, numpy array, or 16-bit PCM bytes."""
        if isinstance(wave_file, (str, Path)):
            path_str = str(wave_file)
            self.filename: str | None = path_str
            if wf is not None:
                try:
                    rate, data = wf.read(path_str)
                    self.rate = int(rate)
                    self.data = np.asarray(data)
                except ValueError:
                    self.rate, self.data = self._read_wav_stdlib(path_str)
            else:  # pragma: no cover
                self.rate, self.data = self._read_wav_stdlib(path_str)
        elif isinstance(wave_file, np.ndarray):
            if sample_rate is None or int(sample_rate) <= 0:
                raise ValueError(
                    "A positive sample_rate is required for ndarray input."
                )
            self.filename = None
            self.rate = int(sample_rate)
            self.data = np.asarray(wave_file).copy()
        elif isinstance(wave_file, (bytes, bytearray, memoryview)):
            if sample_rate is None or int(sample_rate) <= 0:
                raise ValueError(
                    "A positive sample_rate is required for PCM bytes input."
                )
            if len(wave_file) % 2 != 0:
                raise ValueError(
                    "Raw 16-bit PCM buffer must have an even number of bytes."
                )
            self.filename = None
            self.rate = int(sample_rate)
            self.data = np.frombuffer(wave_file, dtype="<i2").copy()
        else:
            raise TypeError(
                f"Unsupported audio source type: {type(wave_file)!r}"
            )

        if self.data.ndim == 1:
            self.channels = 1
        elif self.data.ndim == 2:
            self.channels = int(self.data.shape[1])
        else:
            raise ValueError(
                f"Audio data must be a 1D or 2D array, got {self.data.ndim}D."
            )
        return self

    @staticmethod
    def _normalize_pcm(data: np.ndarray) -> np.ndarray:
        """Normalize integer or float PCM arrays to float64 without overflow."""
        if data.dtype == np.uint8:
            return (data.astype(np.float64) - 128.0) / 128.0
        if data.dtype == np.int16:
            return data.astype(np.float64) / 32768.0
        if data.dtype == np.int32:
            return data.astype(np.float64) / 2147483648.0
        return data.astype(np.float64)

    def _convert_to_mono(self) -> VoiceActivityDetector:
        """Convert normalized audio to a 1D float64 mono signal safely."""
        normalized = self._normalize_pcm(self.data)
        if normalized.ndim == 2:
            self.data = np.asarray(
                np.mean(normalized, axis=1, dtype=np.float64), dtype=np.float64
            )
            self.channels = 1
        else:
            self.data = normalized
            self.channels = 1
        return self

    def _calculate_frequencies(self, audio_data: np.ndarray) -> np.ndarray:
        """Calculate positive FFT frequencies (excluding DC) for a frame."""
        if len(audio_data) <= 1:
            return np.array([], dtype=np.float64)
        data_freq = np.fft.rfftfreq(len(audio_data), 1.0 / self.rate)
        return data_freq[1:]

    def _calculate_amplitude(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply DC removal and Hann window, returning rFFT magnitudes."""
        arr = np.asarray(audio_data, dtype=np.float64)
        if len(arr) <= 1:
            return np.array([], dtype=np.float64)
        zero_mean = arr - np.mean(arr)
        windowed = zero_mean * np.hanning(len(arr))
        data_ampl = np.abs(np.fft.rfft(windowed))
        return data_ampl[1:]

    def _calculate_energy(self, data: np.ndarray) -> np.ndarray:
        """Compute spectral power for each positive frequency bin."""
        data_amplitude = self._calculate_amplitude(data)
        return data_amplitude**2

    def _znormalize_energy(self, data_energy: np.ndarray) -> np.ndarray:
        """Z-normalize an energy vector, returning zeros when std is ~0."""
        arr = np.asarray(data_energy, dtype=np.float64)
        energy_mean = np.mean(arr)
        energy_std = np.std(arr)
        if energy_std <= 1e-12:
            return np.zeros_like(arr, dtype=np.float64)
        return np.asarray((arr - energy_mean) / energy_std, dtype=np.float64)

    def _connect_energy_with_frequencies(
        self, data_freq: np.ndarray, data_energy: np.ndarray
    ) -> dict[float, float]:
        """Map positive frequency bins (Hz) to their spectral energy."""
        energy_freq: dict[float, float] = {}
        for i, freq in enumerate(data_freq):
            key = float(abs(freq))
            if key not in energy_freq:
                energy_freq[key] = float(data_energy[i] * 2.0)
        return energy_freq

    def _calculate_normalized_energy(
        self, data: np.ndarray
    ) -> dict[float, float]:
        """Compute frequency-to-energy mapping for a single audio frame."""
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        return self._connect_energy_with_frequencies(data_freq, data_energy)

    def _sum_energy_in_band(
        self,
        energy_frequencies: dict[float, float],
        start_band: int | float,
        end_band: int | float,
    ) -> float:
        """Sum spectral energy strictly within (start_band, end_band) Hz."""
        sum_energy = 0.0
        for f, val in energy_frequencies.items():
            if start_band < f < end_band:
                sum_energy += float(val)
        return sum_energy

    def _median_filter(self, x: np.ndarray, k: int) -> np.ndarray:
        """Apply a 1D edge-padded median filter of odd kernel length k."""
        if k % 2 != 1:
            raise ValueError("Median filter length must be odd.")
        if x.ndim != 1:
            raise ValueError("Input must be one-dimensional.")
        if len(x) == 0 or k <= 1:
            return x.astype(np.float64, copy=True)
        k2 = (k - 1) // 2
        padded = np.pad(x.astype(np.float64, copy=False), k2, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, k)
        return np.asarray(np.median(windows, axis=1), dtype=np.float64)

    def _smooth_speech_detection(
        self, detected_windows: np.ndarray
    ) -> np.ndarray:
        """Smooth binary frame flags with a median filter over speech_window."""
        if len(detected_windows) == 0:
            return np.array([], dtype=np.float64)
        median_window = max(1, int(self.speech_window / self.sample_overlap))
        median_window = min(median_window, len(detected_windows))
        if median_window % 2 == 0:
            median_window = max(1, median_window - 1)
        return self._median_filter(detected_windows[:, 1], median_window)

    def convert_windows_to_readible_labels(
        self, detected_windows: np.ndarray
    ) -> list[dict[str, float | int]]:
        """Convert [sample_start, is_speech] rows into speech intervals.

        Preserves the historical method spelling ('readible') for backward
        compatibility while closing trailing speech segments and including both
        seconds ('speech_begin', 'speech_end') and sample indices
        ('start_sample', 'end_sample').
        """
        speech_time: list[dict[str, float | int]] = []
        if len(detected_windows) == 0:
            return speech_time

        is_speech = 0
        start_sample = 0
        speech_begin = 0.0

        for window in detected_windows:
            win_sample = int(window[0])
            flag = float(window[1])
            if flag == 1.0 and is_speech == 0:
                is_speech = 1
                start_sample = win_sample
                speech_begin = float(start_sample / self.rate)
            elif flag == 0.0 and is_speech == 1:
                is_speech = 0
                end_sample = win_sample
                speech_end = float(end_sample / self.rate)
                speech_time.append(
                    {
                        "speech_begin": speech_begin,
                        "speech_end": speech_end,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                    }
                )

        if is_speech == 1:
            last_win_start = int(detected_windows[-1, 0])
            step_samples = max(1, int(self.rate * self.sample_overlap))
            end_sample = max(len(self.data), last_win_start + step_samples)
            speech_end = float(end_sample / self.rate)
            speech_time.append(
                {
                    "speech_begin": speech_begin,
                    "speech_end": speech_end,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                }
            )

        return speech_time

    def convert_windows_to_readable_labels(
        self, detected_windows: np.ndarray
    ) -> list[dict[str, float | int]]:
        """Convert [sample_start, is_speech] rows into speech intervals."""
        return self.convert_windows_to_readible_labels(detected_windows)

    def detect_intervals(self) -> list[dict[str, float | int]]:
        """Detect speech and return a list of interval dictionaries."""
        detected_windows = self.detect_speech()
        return self.convert_windows_to_readable_labels(detected_windows)

    def extract_speech_segments(self, pad_ms: float = 0.0) -> list[np.ndarray]:
        """Extract audio waveform slices for each detected speech interval.

        Args:
            pad_ms: Optional padding in milliseconds to include before and after
                each speech interval. Overlapping padded intervals are merged.

        Returns:
            List of 1D float64 numpy arrays for each speech segment.
        """
        intervals = self.detect_intervals()
        if not intervals:
            return []

        pad_samples = max(0, int(round(self.rate * (float(pad_ms) / 1000.0))))
        n_samples = len(self.data)
        merged_ranges: list[list[int]] = []

        for item in intervals:
            start_idx = max(0, int(item["start_sample"]) - pad_samples)
            end_idx = min(n_samples, int(item["end_sample"]) + pad_samples)
            if merged_ranges and start_idx <= merged_ranges[-1][1]:
                merged_ranges[-1][1] = max(merged_ranges[-1][1], end_idx)
            else:
                merged_ranges.append([start_idx, end_idx])

        return [self.data[s:e].copy() for s, e in merged_ranges]

    def strip_silence(
        self,
        output_path: str | Path | None = None,
        pad_ms: float = 20.0,
    ) -> np.ndarray:
        """Concatenate speech segments and optionally write a cleaned WAV file.

        Args:
            output_path: Optional file path to write 16-bit PCM mono WAV output.
            pad_ms: Padding in milliseconds retained around each speech segment.

        Returns:
            Concatenated 1D float64 numpy array containing only speech segments.
        """
        segments = self.extract_speech_segments(pad_ms=pad_ms)
        if segments:
            trimmed: np.ndarray = np.asarray(
                np.concatenate(segments), dtype=np.float64
            )
        else:
            trimmed = np.array([], dtype=np.float64)

        if output_path is not None:
            pcm_int16 = np.clip(
                np.round(trimmed * 32767.0), -32768, 32767
            ).astype("<i2")
            with wave.open(str(output_path), "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(self.rate)
                wav_out.writeframes(pcm_int16.tobytes())

        return trimmed

    def plot_detected_speech_regions(self) -> VoiceActivityDetector:
        """Detect speech and plot the waveform with speech regions overlaid."""
        plt = importlib.import_module("matplotlib.pyplot")
        data = self.data
        detected_windows = self.detect_speech()
        data_speech = np.zeros(len(data), dtype=np.float64)
        sample_window = max(1, int(self.rate * self.sample_window))

        for window in detected_windows:
            if float(window[1]) == 1.0:
                start_idx = int(window[0])
                end_idx = min(len(data), start_idx + sample_window)
                data_speech[start_idx:end_idx] = data[start_idx:end_idx]

        plt.figure()
        plt.plot(data)
        plt.plot(data_speech)
        plt.show()
        return self

    def detect_speech(self) -> np.ndarray:
        """Detect speech windows using vectorized Hann-windowed rFFT and RMS.

        Returns:
            2D numpy array of shape (num_windows, 2) where column 0 is the
            window start sample index and column 1 is the smoothed binary flag
            (1.0 for speech, 0.0 for non-speech).
        """
        data = self.data
        n_samples = len(data)
        if n_samples == 0:
            return np.empty((0, 2), dtype=np.float64)

        sample_window = max(1, int(self.rate * self.sample_window))
        sample_overlap = max(1, int(self.rate * self.sample_overlap))

        if n_samples <= sample_window:
            sample_starts = np.array([0], dtype=np.float64)
            frames = data[np.newaxis, :]
        else:
            all_windows = np.lib.stride_tricks.sliding_window_view(
                data, sample_window
            )
            frames = all_windows[::sample_overlap]
            sample_starts = np.arange(len(frames), dtype=np.float64) * float(
                sample_overlap
            )

        frame_len = frames.shape[1]
        frames_ac = frames - np.mean(frames, axis=1, keepdims=True)
        frame_rms = np.sqrt(np.mean(frames_ac**2, axis=1))

        if frame_len <= 1:
            speech_ratio: np.ndarray = np.zeros(len(frames), dtype=np.float64)
        else:
            hann = np.hanning(frame_len)
            spectrum = np.abs(np.fft.rfft(frames_ac * hann, axis=1))[:, 1:] ** 2
            freqs = np.fft.rfftfreq(frame_len, 1.0 / self.rate)[1:]
            band_mask = (freqs > self.speech_start_band) & (
                freqs < self.speech_end_band
            )
            voice_energy = np.sum(spectrum[:, band_mask], axis=1)
            full_energy = np.sum(spectrum, axis=1)
            safe_denom = np.where(full_energy > 1e-12, full_energy, 1.0)
            speech_ratio = np.asarray(
                np.where(full_energy > 1e-12, voice_energy / safe_denom, 0.0),
                dtype=np.float64,
            )

        if self.energy_threshold > 0.0 and len(frame_rms) > 0:
            adaptive_floor = min(
                float(np.percentile(frame_rms, 10)) * 2.0,
                float(np.max(frame_rms)) * 0.25,
            )
            rms_gate = max(self.energy_threshold, adaptive_floor)
            is_speech = (
                (speech_ratio > self.speech_energy_threshold)
                & (frame_rms > rms_gate)
            ).astype(np.float64)
        else:
            is_speech = (speech_ratio > self.speech_energy_threshold).astype(
                np.float64
            )

        detected_windows = np.column_stack((sample_starts, is_speech))
        detected_windows[:, 1] = self._smooth_speech_detection(detected_windows)
        return detected_windows
