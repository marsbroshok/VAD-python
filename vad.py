from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

try:
    import scipy.io.wavfile as wf
except ImportError:  # pragma: no cover
    wf = None  # type: ignore[assignment]


class VoiceActivityDetector:
    """Use spectral band-energy ratio and RMS gating to detect voice activity."""

    def __init__(
        self,
        wave_input_filename: str | Path | np.ndarray | bytes | bytearray | memoryview,
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
        self._read_wav(wave_input_filename, sample_rate=sample_rate)._convert_to_mono()

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
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes.")

        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        return int(rate), data

    def _read_wav(
        self,
        wave_file: str | Path | np.ndarray | bytes | bytearray | memoryview,
        sample_rate: int | None = None,
    ) -> VoiceActivityDetector:
        """Load audio from a file path, numpy array, or 16-bit PCM byte buffer."""
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
                    "A positive sample_rate must be provided for numpy.ndarray input."
                )
            self.filename = None
            self.rate = int(sample_rate)
            self.data = np.asarray(wave_file).copy()
        elif isinstance(wave_file, (bytes, bytearray, memoryview)):
            if sample_rate is None or int(sample_rate) <= 0:
                raise ValueError(
                    "A positive sample_rate must be provided for raw PCM bytes input."
                )
            if len(wave_file) % 2 != 0:
                raise ValueError(
                    "Raw 16-bit PCM byte buffer must contain an even number of bytes."
                )
            self.filename = None
            self.rate = int(sample_rate)
            self.data = np.frombuffer(wave_file, dtype="<i2").copy()
        else:
            raise TypeError(f"Unsupported audio source type: {type(wave_file)!r}")

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
        """Normalize integer or floating-point PCM arrays to float64 without overflow."""
        if data.dtype == np.uint8:
            return (data.astype(np.float64) - 128.0) / 128.0
        if data.dtype == np.int16:
            return data.astype(np.float64) / 32768.0
        if data.dtype == np.int32:
            return data.astype(np.float64) / 2147483648.0
        return data.astype(np.float64)

    def _convert_to_mono(self) -> VoiceActivityDetector:
        """Convert normalized audio data to a 1D float64 mono signal without overflow."""
        normalized = self._normalize_pcm(self.data)
        if normalized.ndim == 2:
            self.data = np.mean(normalized, axis=1, dtype=np.float64)
            self.channels = 1
        else:
            self.data = normalized
            self.channels = 1
        return self
    
    def _calculate_frequencies(self, audio_data: np.ndarray) -> np.ndarray:
        """Calculate positive FFT frequencies (excluding DC bin 0) for a frame."""
        if len(audio_data) <= 1:
            return np.array([], dtype=np.float64)
        data_freq = np.fft.rfftfreq(len(audio_data), 1.0 / self.rate)
        return data_freq[1:]

    def _calculate_amplitude(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply DC removal and Hann windowing, then return positive rFFT magnitudes."""
        arr = np.asarray(audio_data, dtype=np.float64)
        if len(arr) <= 1:
            return np.array([], dtype=np.float64)
        zero_mean = arr - np.mean(arr)
        windowed = zero_mean * np.hanning(len(arr))
        data_ampl = np.abs(np.fft.rfft(windowed))
        return data_ampl[1:]

    def _calculate_energy(self, data: np.ndarray) -> np.ndarray:
        """Compute spectral power (squared amplitude) for each positive frequency bin."""
        data_amplitude = self._calculate_amplitude(data)
        return data_amplitude**2

    def _znormalize_energy(self, data_energy: np.ndarray) -> np.ndarray:
        """Z-normalize an energy vector, safely returning zeros when std is ~0."""
        arr = np.asarray(data_energy, dtype=np.float64)
        energy_mean = np.mean(arr)
        energy_std = np.std(arr)
        if energy_std <= 1e-12:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - energy_mean) / energy_std

    def _connect_energy_with_frequencies(
        self, data_freq: np.ndarray, data_energy: np.ndarray
    ) -> dict[float, float]:
        """Map positive frequency bins (Hz) to their corresponding spectral energy."""
        energy_freq: dict[float, float] = {}
        for i, freq in enumerate(data_freq):
            key = float(abs(freq))
            if key not in energy_freq:
                energy_freq[key] = float(data_energy[i] * 2.0)
        return energy_freq

    def _calculate_normalized_energy(self, data: np.ndarray) -> dict[float, float]:
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
    
    def _median_filter (self, x, k):
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros ((len (x), k), dtype=x.dtype)
        y[:,k2] = x
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]
        return np.median (y, axis=1)
        
    def _smooth_speech_detection(self, detected_windows):
        median_window=int(self.speech_window/self.sample_window)
        if median_window%2==0: median_window=median_window-1
        median_energy = self._median_filter(detected_windows[:,1], median_window)
        return median_energy
        
    def convert_windows_to_readible_labels(self, detected_windows):
        """ Takes as input array of window numbers and speech flags from speech
        detection and convert speech flags to time intervals of speech.
        Output is array of dictionaries with speech intervals.
        """
        speech_time = []
        is_speech = 0
        for window in detected_windows:
            if (window[1]==1.0 and is_speech==0): 
                is_speech = 1
                speech_label = {}
                speech_time_start = window[0] / self.rate
                speech_label['speech_begin'] = speech_time_start
                print(window[0], speech_time_start)
                #speech_time.append(speech_label)
            if (window[1]==0.0 and is_speech==1):
                is_speech = 0
                speech_time_end = window[0] / self.rate
                speech_label['speech_end'] = speech_time_end
                speech_time.append(speech_label)
                print(window[0], speech_time_end)
        return speech_time
      
    def plot_detected_speech_regions(self):
        """ Performs speech detection and plot original signal and speech regions.
        """
        data = self.data
        detected_windows = self.detect_speech()
        data_speech = np.zeros(len(data))
        it = np.nditer(detected_windows[:,0], flags=['f_index'])
        while not it.finished:
            data_speech[int(it[0])] = data[int(it[0])] * detected_windows[it.index,1]
            it.iternext()
        plt.figure()
        plt.plot(data_speech)
        plt.plot(data)
        plt.show()
        return self
       
    def detect_speech(self) -> np.ndarray:
        """Detect speech windows using vectorized Hann-windowed rFFT and RMS gating.

        Returns:
            2D numpy array of shape (num_windows, 2) where column 0 is the window
            start sample index and column 1 is the smoothed binary speech flag
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
            all_windows = np.lib.stride_tricks.sliding_window_view(data, sample_window)
            frames = all_windows[::sample_overlap]
            sample_starts = (
                np.arange(len(frames), dtype=np.float64) * float(sample_overlap)
            )

        frame_len = frames.shape[1]
        frames_ac = frames - np.mean(frames, axis=1, keepdims=True)
        frame_rms = np.sqrt(np.mean(frames_ac**2, axis=1))

        if frame_len <= 1:
            speech_ratio = np.zeros(len(frames), dtype=np.float64)
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
            speech_ratio = np.where(full_energy > 1e-12, voice_energy / safe_denom, 0.0)

        if self.energy_threshold > 0.0 and len(frame_rms) > 0:
            adaptive_floor = min(
                float(np.percentile(frame_rms, 10)) * 2.0,
                float(np.max(frame_rms)) * 0.25,
            )
            rms_gate = max(self.energy_threshold, adaptive_floor)
            is_speech = (
                (speech_ratio > self.speech_energy_threshold) & (frame_rms > rms_gate)
            ).astype(np.float64)
        else:
            is_speech = (speech_ratio > self.speech_energy_threshold).astype(np.float64)

        detected_windows = np.column_stack((sample_starts, is_speech))
        detected_windows[:, 1] = self._smooth_speech_detection(detected_windows)
        return detected_windows
 
