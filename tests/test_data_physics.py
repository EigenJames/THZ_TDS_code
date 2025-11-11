import numpy as np

from src.data import SPEED_OF_LIGHT, material_physics, simulate_thz_waveform


def test_simulated_delay_matches_dielectric_properties() -> None:
    fs = 1e12
    length = 2048
    smoothing_window = np.ones(11) / 11.0

    for label, params in material_physics.items():
        rng = np.random.default_rng(1234)
        times, waveform = simulate_thz_waveform(length, fs, label, rng=rng)

        expected_delay_ps = (
            ((params["n"] - 1.0) * params["thickness_mm"] * 1e-3) / SPEED_OF_LIGHT
        ) * 1e12

        envelope = np.abs(waveform)
        smoothed = np.convolve(envelope, smoothing_window, mode="same")
        peak_time_ps = times[int(np.argmax(smoothed))] * 1e12

        assert (
            abs(peak_time_ps - expected_delay_ps) < 8.0
        ), f"Delay mismatch for {label}: expected {expected_delay_ps:.2f} ps, got {peak_time_ps:.2f} ps"
