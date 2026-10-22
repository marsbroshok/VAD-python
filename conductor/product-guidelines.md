# Product Guidelines: VAD-python

## Architectural & UX Principles
1. **Explainable White-Box DSP**: Every signal processing step (windowing, rFFT, band-energy ratio, RMS gating, median filtering) must be modular, well-documented with docstrings and type annotations, and easy to inspect.
2. **Sensible Defaults, Full Tunability**: Default parameters (`sample_window=0.02`, `sample_overlap=0.01`, `speech_window=0.5`, `speech_energy_threshold=0.6`, `speech_start_band=300`, `speech_end_band=3000`) work out of the box, while all thresholds remain configurable via constructor arguments and CLI flags.
3. **Flexible Input & Output Ergonomics**:
   - Accept file paths (`str` / `pathlib.Path`), in-memory `numpy.ndarray` buffers, or streaming audio frames.
   - Return structured speech intervals containing both time in seconds (`speech_begin`, `speech_end`) and sample indices (`start_sample`, `end_sample`).
4. **Honest Documentation**: Clearly explain in `README.md` when spectral energy-ratio VAD excels (clean speech, silence trimming, low-CPU Stage-1 gating, educational DSP) and when users should pair it with or use a neural VAD (heavy non-stationary noise in the 300–3000 Hz band).
