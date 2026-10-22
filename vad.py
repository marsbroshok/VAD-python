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
    
    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)
        data_freq = data_freq[1:]
        return data_freq    
    
    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl
        
    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy
        
    def _znormalize_energy(self, data_energy):
        energy_mean = np.mean(data_energy)
        energy_std = np.std(data_energy)
        energy_znorm = (data_energy - energy_mean) / energy_std
        return energy_znorm
    
    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq
    
    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        #data_energy = self._znormalize_energy(data_energy) #znorm brings worse results
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq
    
    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band<f<end_band:
                sum_energy += energy_frequencies[f]
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
       
    def detect_speech(self):
        """ Detects speech regions based on ratio between speech band energy
        and total energy.
        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).
        """
        detected_windows = np.array([])
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        data = self.data
        sample_start = 0
        start_band = self.speech_start_band
        end_band = self.speech_end_band
        while (sample_start < (len(data) - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end>=len(data): sample_end = len(data)-1
            data_window = data[sample_start:sample_end]
            energy_freq = self._calculate_normalized_energy(data_window)
            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
            sum_full_energy = sum(energy_freq.values())
            speech_ratio = sum_voice_energy/sum_full_energy
            # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
            speech_ratio = speech_ratio>self.speech_energy_threshold
            detected_windows = np.append(detected_windows,[sample_start, speech_ratio])
            sample_start += sample_overlap
        detected_windows = detected_windows.reshape(int(len(detected_windows)/2),2)
        detected_windows[:,1] = self._smooth_speech_detection(detected_windows)
        return detected_windows
 
