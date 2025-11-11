"""
CLI for prediction on single THz waveforms or spectra.

Usage:
    python -m src.predict --input path/to/waveform.csv --domain time
    python -m src.predict --input path/to/spectrum.csv --domain freq
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import extract_features
from src.model import load_artifacts
from src.preprocess import compute_fft, preprocess_waveform


def main():
    parser = argparse.ArgumentParser(
        description="Predict material class from THz waveform or spectrum."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file (single waveform or spectrum).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["time", "freq"],
        required=True,
        help="Input domain: 'time' for time-domain waveform, 'freq' for frequency-domain spectrum.",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="Path to artifacts directory containing model, scaler, and PCA.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/prediction.json",
        help="Path to output JSON file for prediction results.",
    )

    args = parser.parse_args()

    # Load artifacts
    try:
        model, scaler, pca = load_artifacts(args.artifacts)
    except Exception as e:
        print(f"Error loading artifacts: {e}", file=sys.stderr)
        sys.exit(1)

    # Load input data
    try:
        df = pd.read_csv(args.input)
        if df.shape[0] < 2:
            raise ValueError("Input CSV must have at least 2 rows.")
    except Exception as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Process based on domain
    try:
        if args.domain == "time":
            # Expect two columns: time, amplitude
            if df.shape[1] < 2:
                raise ValueError("Time-domain CSV must have at least 2 columns: time, amplitude.")

            time_col = df.iloc[:, 0].values
            signal_col = df.iloc[:, 1].values

            # Preprocess
            signal_proc = preprocess_waveform(signal_col)

            # Compute FFT
            dt = time_col[1] - time_col[0]
            freqs_hz, mag = compute_fft(signal_proc, dt)
            freq_thz = freqs_hz / 1e12

        elif args.domain == "freq":
            # Expect two columns: frequency (THz), magnitude
            if df.shape[1] < 2:
                raise ValueError("Frequency-domain CSV must have at least 2 columns: frequency, magnitude.")

            freq_thz = df.iloc[:, 0].values
            mag = df.iloc[:, 1].values

        else:
            raise ValueError(f"Unknown domain: {args.domain}")

    except Exception as e:
        print(f"Error processing input: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract features
    try:
        features = extract_features(freq_thz, mag, topk=5)
        features = features.reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}", file=sys.stderr)
        sys.exit(1)

    # Scale and predict
    try:
        features_scaled = scaler.transform(features)
        pred_label = model.predict(features_scaled)[0]
        pred_proba = model.predict_proba(features_scaled)[0]

        # Build result
        result = {
            "predicted_label": str(pred_label),
            "probabilities": {
                str(cls): float(prob) for cls, prob in zip(model.classes_, pred_proba)
            },
        }

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

    # Print to console
    print(json.dumps(result, indent=2))

    # Save to file
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nPrediction saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save output file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
