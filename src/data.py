from __future__ import annotations

"""
Data utilities: DETRIS Zenodo loader with synthetic fallback.

The DETRIS dataset (DOI: 10.5281/zenodo.5079558) contains frequency-domain
THz spectra for various materials. We download a subset, parse the two-column
txt files (frequency in THz, complex amplitude), perform Hermitian IFFT to
obtain real time-domain waveforms, and resample to a common grid.

If the download fails or the user disables DETRIS, we generate synthetic
THz-like waveforms with class-specific characteristics.
"""

import os
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# DETRIS Zenodo dataset DOI: 10.5281/zenodo.5079558
DETRIS_ZENODO_URL = "https://zenodo.org/record/5079558/files/DETRIS.zip"
DETRIS_MATERIALS = [
    "PTFE",  # Polymer
    "Al2O3",  # Ceramic
    "GFRP",  # Composite
]

SPEED_OF_LIGHT = 3.0e8  # m/s
TAIL_DECAY_COEFF = 8e-12  # s·cm, empirical scaling for absorption-driven decay
ATTENUATION_PATH_CM = 0.1  # convert mm thickness to cm for amplitude attenuation

# Representative dielectric parameters drawn from terahertz spectroscopy literature,
# Adjusted to match experimental THz-TDS characteristics (peak ~0.4 THz, broader pulses)
# Based on typical PVA/MXene nanocomposite measurements
material_physics: Dict[str, Dict[str, float]] = {
    "Polymer": {
        "n": 1.55,
        "alpha_cm-1": 8.0,
        "thickness_mm": 0.5,
        "central_freq_thz": 0.42,  # Match experimental peak ~0.4 THz
        "bandwidth_thz": 0.35,
    },
    "Ceramic": {
        "n": 1.75,
        "alpha_cm-1": 18.0,
        "thickness_mm": 0.5,
        "central_freq_thz": 0.38,  # Slightly lower due to higher absorption
        "bandwidth_thz": 0.32,
    },
    "Composite": {
        "n": 1.65,
        "alpha_cm-1": 12.0,
        "thickness_mm": 0.5,
        "central_freq_thz": 0.40,
        "bandwidth_thz": 0.34,
    },
}


def load_detris_subset(
    materials: int = 3,
    files_per_class: int = 24,
    target_points: int = 1024,
    fresh: bool = True,
    timeout: int = 120,
    seed: int = 42,
    cache_dir: Path | str = Path("./data_cache"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download DETRIS dataset from Zenodo, parse frequency-domain spectra,
    perform Hermitian IFFT to get real time-domain waveforms, and resample.

    Parameters
    ----------
    materials : int
        Number of material classes to load (1-3).
    files_per_class : int
        Number of files to load per class.
    target_points : int
        Number of points to resample each waveform to.
    fresh : bool
        If True, always download fresh data. If False, reuse cached zip if available.
    timeout : int
        Request timeout in seconds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray (target_points,)
        Time axis in seconds.
    X_time : np.ndarray (n_samples, target_points)
        Time-domain waveforms (real-valued).
    y_labels : np.ndarray (n_samples,)
        Class labels (str).
    freq_thz : np.ndarray (target_points//2 + 1,)
        Frequency axis in THz (non-negative frequencies only).
    spectra : np.ndarray (n_samples, target_points//2 + 1)
        Magnitude spectra.

    Raises
    ------
    RuntimeError
        If download fails or data cannot be parsed.
    """
    rng = np.random.default_rng(seed)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "DETRIS.zip"

    if fresh or not cache_path.exists():
        print(f"Downloading DETRIS dataset from Zenodo ({DETRIS_ZENODO_URL})...")
        try:
            response = requests.get(DETRIS_ZENODO_URL, timeout=timeout, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with cache_path.open("wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download DETRIS dataset: {e}") from e
        print(f"Using cached DETRIS archive at {cache_path.resolve()}")

    # Extract and parse files
    X_time_list = []
    y_labels_list = []
    spectra_list = []

    with zipfile.ZipFile(cache_path, "r") as zf:
        # List all txt files
        all_files = [f for f in zf.namelist() if f.endswith(".txt")]

        # Select materials
        selected_materials = DETRIS_MATERIALS[:materials]

        for material in selected_materials:
            # Find files for this material
            material_files = [f for f in all_files if material in f]
            if not material_files:
                warnings.warn(f"No files found for material {material}, skipping.")
                continue

            selected_files = rng.choice(
                material_files, size=min(files_per_class, len(material_files)), replace=False
            )

            for file_path in selected_files:
                try:
                    with zf.open(file_path) as f:
                        # Parse two-column txt: frequency (THz), complex amplitude
                        data = np.loadtxt(f, dtype=float)
                        if data.shape[1] < 2:
                            continue

                        freq_orig = data[:, 0]  # THz
                        # Assume real and imaginary parts or magnitude/phase
                        # DETRIS format: typically frequency, real, imaginary
                        # or frequency, magnitude. Check shape:
                        if data.shape[1] == 2:
                            # Assume magnitude only
                            mag = data[:, 1]
                            complex_spectrum = mag + 0j
                        elif data.shape[1] >= 3:
                            # Assume real, imaginary
                            real_part = data[:, 1]
                            imag_part = data[:, 2]
                            complex_spectrum = real_part + 1j * imag_part
                        else:
                            continue

                        # Resample frequency to common grid
                        freq_uniform = np.linspace(
                            freq_orig.min(), freq_orig.max(), target_points // 2 + 1
                        )
                        # Interpolate real and imaginary separately
                        real_interp = np.interp(freq_uniform, freq_orig, complex_spectrum.real)
                        imag_interp = np.interp(freq_uniform, freq_orig, complex_spectrum.imag)
                        spectrum_uniform = real_interp + 1j * imag_interp

                        # Create Hermitian symmetric spectrum for IFFT
                        # Full spectrum: [DC, positive freqs, negative freqs (conjugate)]
                        if target_points % 2 == 0:
                            # Even: [0, 1, ..., N/2, -(N/2-1), ..., -1]
                            neg_freqs = np.conj(spectrum_uniform[-2:0:-1])
                        else:
                            # Odd: [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1]
                            neg_freqs = np.conj(spectrum_uniform[-1:0:-1])

                        full_spectrum = np.concatenate([spectrum_uniform, neg_freqs])

                        # IFFT to time domain (should be real-valued)
                        time_signal = np.fft.ifft(full_spectrum).real

                        # Normalise
                        time_signal = time_signal - np.mean(time_signal)
                        std = np.std(time_signal)
                        if std > 1e-12:
                            time_signal = time_signal / std

                        X_time_list.append(time_signal)
                        y_labels_list.append(material)
                        spectra_list.append(np.abs(spectrum_uniform))

                except Exception as e:
                    warnings.warn(f"Failed to parse {file_path}: {e}")
                    continue

    if not X_time_list:
        raise RuntimeError("No valid DETRIS files parsed.")

    X_time = np.array(X_time_list)
    y_labels = np.array(y_labels_list)
    spectra = np.array(spectra_list)

    # Create time and frequency axes
    # Assume max frequency from DETRIS is ~5 THz, dt = 1/(2*f_max)
    f_max_hz = freq_uniform[-1] * 1e12  # Convert THz to Hz
    dt = 1 / (2 * f_max_hz)
    times = np.arange(target_points) * dt  # seconds
    freq_thz = freq_uniform  # THz

    print(
        f"Loaded DETRIS subset: {X_time.shape[0]} samples, {X_time.shape[1]} points each."
    )
    print(f"Materials: {sorted(set(y_labels.tolist()))}")

    return times, X_time, y_labels, freq_thz, spectra


def simulate_thz_waveform(
    length: int,
    fs: float,
    class_label: str,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a single THz waveform matching experimental characteristics.
    
    Generates Gaussian-windowed few-cycle THz pulses matching PVA/MXene nanocomposite data:
    - Spectral peak at ~0.4 THz (controlled by carrier frequency)
    - Smooth exponential decay in frequency domain
    - Material-dependent arrival time (refractive index delay)
    - Few-cycle temporal structure with absorption-induced asymmetry
    """

    if rng is None:
        rng = np.random.default_rng()

    if class_label not in material_physics:
        raise ValueError(f"Unknown class label '{class_label}' for synthetic generation")

    params = material_physics[class_label]
    n = params["n"]
    alpha_cm = params["alpha_cm-1"]
    thickness_mm = params["thickness_mm"]
    center_thz = params["central_freq_thz"]
    bandwidth_thz = params["bandwidth_thz"]

    thickness_m = thickness_mm * 1e-3
    thickness_cm = thickness_mm * 0.1

    # Calculate time delay from refractive index: Δt = (n-1)×d/c
    delay_s = ((n - 1.0) * thickness_m) / SPEED_OF_LIGHT
    
    # Temporal width from bandwidth: broader bandwidth → shorter pulse
    # σ_time ≈ 1/(2π×BW), for BW=0.35 THz → σ ≈ 0.45 ps
    sigma_time = 1.0 / (2.0 * np.pi * bandwidth_thz * 1e12)
    
    # Amplitude attenuation from absorption: A = exp(-α×d)
    amplitude = float(np.clip(np.exp(-alpha_cm * thickness_cm * 0.7), 0.4, 1.0))

    # Time axis
    t = np.arange(length) / fs
    t_centered = t - (t[-1] / 2)  # Center around 0
    shifted_time = t_centered - delay_s  # Apply refractive delay

    # Create THz pulse: Gaussian envelope × cosine carrier
    # This is a standard "Gabor atom" or Gaussian-modulated sinusoid
    envelope = np.exp(-0.5 * (shifted_time / sigma_time) ** 2)
    carrier = np.cos(2.0 * np.pi * center_thz * 1e12 * shifted_time)
    main_pulse = amplitude * envelope * carrier
    
    # Add asymmetric decay tail from absorption (makes pulse slightly asymmetric)
    # Higher absorption → faster decay
    tau_decay = 3.0e-12 / max(alpha_cm / 10.0, 1.0)
    decay_factor = np.where(
        shifted_time > 0,
        np.exp(-shifted_time / tau_decay),
        1.0  # No decay before pulse
    )
    waveform = main_pulse * decay_factor
    
    # Realistic experimental artifacts
    # 1. Low-frequency baseline drift (systematic error)
    drift_freq = rng.uniform(0.5e10, 2.0e10)  # 5-20 GHz
    drift_amp = rng.uniform(0.005, 0.015) * amplitude
    drift = drift_amp * np.sin(2.0 * np.pi * drift_freq * t)
    
    # 2. White noise (shot noise, electronics)
    noise_level = rng.uniform(0.008, 0.015) * amplitude
    noise = rng.normal(0.0, noise_level, size=length)
    
    # 3. Small pre-pulse (optical etalon, beam path reflections)
    # Appears ~1-2 ps before main pulse, much weaker
    prepulse_delay = rng.uniform(-2.0e-12, -1.0e-12)  # 1-2 ps before
    prepulse_time = shifted_time - prepulse_delay
    prepulse_amp = rng.uniform(0.03, 0.08) * amplitude
    prepulse = prepulse_amp * np.exp(-0.5 * (prepulse_time / sigma_time) ** 2) * np.cos(2.0 * np.pi * center_thz * 1e12 * prepulse_time)
    
    # Combine all components
    waveform = waveform + drift + noise + prepulse
    
    # Remove DC offset
    waveform -= waveform.mean()
    
    # Scale to experimental amplitude range (0.1-0.3 peak-to-peak typically)
    waveform_max = np.abs(waveform).max()
    if waveform_max > 1e-12:
        target_amp = rng.uniform(0.12, 0.28)
        waveform = waveform / waveform_max * target_amp

    return t, waveform


def load_synthetic(
    n_samples_per_class: int = 120,
    target_points: int = 1024,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic THz-like waveforms with class-specific characteristics.

    Parameters
    ----------
    n_samples_per_class : int
        Number of samples per class.
    target_points : int
        Number of points per waveform.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    times : np.ndarray (target_points,)
        Time axis in seconds.
    X_time : np.ndarray (n_samples, target_points)
        Time-domain waveforms.
    y_labels : np.ndarray (n_samples,)
        Class labels (str).
    freq_thz : np.ndarray (target_points//2 + 1,)
        Frequency axis in THz.
    spectra : np.ndarray (n_samples, target_points//2 + 1)
        Magnitude spectra.
    """
    rng = np.random.default_rng(seed)

    X_time_list: List[np.ndarray] = []
    y_labels_list: List[str] = []
    spectra_list: List[np.ndarray] = []
    times = None

    fs = 1e12  # 1 THz sampling rate

    for label in material_physics.keys():
        for _ in range(n_samples_per_class):
            t_axis, waveform = simulate_thz_waveform(
                length=target_points,
                fs=fs,
                class_label=label,
                rng=rng,
            )
            if times is None:
                times = t_axis
            X_time_list.append(waveform)
            y_labels_list.append(label)
            spectra_list.append(np.abs(np.fft.rfft(waveform)))

    X_time = np.vstack(X_time_list)
    y_labels = np.array(y_labels_list)
    spectra = np.vstack(spectra_list)

    if times is None:
        times = np.arange(target_points) / fs

    freq_thz = np.fft.rfftfreq(target_points, d=1.0 / fs) / 1e12

    print(
        f"Generated synthetic dataset: {X_time.shape[0]} samples, {X_time.shape[1]} points each."
    )
    print(f"Classes: {sorted(set(y_labels.tolist()))}")

    return times, X_time, y_labels, freq_thz, spectra


def safe_load(
    use_detris: bool = False,
    materials: int = 3,
    files_per_class: int = 24,
    n_samples_per_class: int = 120,
    target_points: int = 1024,
    seed: int = 42,
    fresh: bool = False,
    cache_dir: Path | str | None = None,
    strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Load physics-based synthetic THz data. DETRIS loading is disabled.

    Parameters
    ----------
    use_detris : bool
        Deprecated; retained for API compatibility.
    materials : int
        Deprecated; retained for API compatibility.
    files_per_class : int
        Deprecated; retained for API compatibility.
    n_samples_per_class : int
        Samples per class (synthetic).
    target_points : int
        Points per waveform.
    seed : int
        Random seed.
    fresh : bool
        Deprecated; retained for API compatibility.
    cache_dir : Path | str | None
        Deprecated; retained for API compatibility.
    strict : bool
        Deprecated; retained for API compatibility.

    Returns
    -------
    times, X_time, y_labels, freq_thz, spectra : np.ndarray
        Data arrays.
    used_detris : bool
        True if DETRIS data was successfully loaded, False if synthetic fallback.
    """
    if use_detris:
        warnings.warn(
            "DETRIS loading has been disabled; returning physics-based synthetic data instead.",
            stacklevel=2,
        )

    times, X_time, y_labels, freq_thz, spectra = load_synthetic(
        n_samples_per_class=n_samples_per_class, target_points=target_points, seed=seed
    )
    print("Using synthetic THz-like data.")
    return times, X_time, y_labels, freq_thz, spectra, False