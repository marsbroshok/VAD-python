"""CLI entrypoint to analyze WAV files, save speech intervals, and trim."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vad import VoiceActivityDetector


def save_to_file(
    data: list[dict[str, float | int]], filename: str | Path
) -> None:
    """Serialize detected speech intervals as JSON to the target file path."""
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(data, fp)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for VAD-python."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze input wave-file, save detected speech intervals to JSON, "
            "or strip silence into a cleaned WAV file."
        )
    )
    parser.add_argument(
        "inputfile",
        metavar="INPUTWAVE",
        help="Path to the input WAV file.",
    )
    parser.add_argument(
        "outputfile",
        metavar="OUTPUTFILE",
        nargs="?",
        default=None,
        help="Optional path to output JSON file for detected speech intervals.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Speech band (300-3000 Hz) energy ratio threshold (default: 0.6).",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=1e-4,
        help="Minimum RMS energy floor gate (default: 1e-4).",
    )
    parser.add_argument(
        "--speech-window",
        type=float,
        default=0.5,
        help="Median filter smoothing window in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--output-wav",
        type=str,
        default=None,
        help="Optional path to save silence-stripped WAV file.",
    )
    parser.add_argument(
        "--pad-ms",
        type=float,
        default=20.0,
        help="Padding in milliseconds when stripping silence (default: 20.0).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot waveform and detected speech regions using matplotlib.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the VAD CLI workflow and return exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.inputfile)
    if not input_path.is_file():
        print(
            f"Error: Input WAV file not found: {args.inputfile}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    detector = VoiceActivityDetector(
        input_path,
        speech_energy_threshold=args.threshold,
        speech_window=args.speech_window,
        energy_threshold=args.energy_threshold,
    )
    raw_detection = detector.detect_speech()
    speech_labels = detector.convert_windows_to_readible_labels(raw_detection)
    print(speech_labels)

    if args.outputfile is not None:
        save_to_file(speech_labels, args.outputfile)

    if args.output_wav is not None:
        detector.strip_silence(output_path=args.output_wav, pad_ms=args.pad_ms)

    if args.plot:
        detector.plot_detected_speech_regions()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
