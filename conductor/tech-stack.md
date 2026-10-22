# Technology Stack: VAD-python

- **Language**: Python >= 3.10 (with full static type annotations)
- **Core DSP & Math**: `numpy` (vectorized `rfft`, Hann windowing, RMS calculation, median filtering)
- **Audio I/O**: Python standard library `wave` + `scipy.io.wavfile` (with direct `numpy.ndarray` input support)
- **Visualization (Optional / Lazy-Loaded)**: `matplotlib`
- **CLI**: `argparse` (`detectVoiceInWave.py` and `vad-python` console script entry point)
- **Packaging**: `pyproject.toml` (PEP 621 standard)
- **Testing & Quality Gates**: `pytest`, `pytest-cov` (100% coverage target on core logic), `hypothesis` (property-based DSP invariant testing), `ruff` / `mypy`
