import numpy as np


def find_stabilization_point(
    sequence,
    window=5,
    slope_threshold=0.005,
    curvature_threshold=0.05,
    patience=1,
    oscillation_tolerance=0.001,
    increasing_trend_threshold=0.01,
    flat_change_threshold=0.003
):
    if len(sequence) < window * 2:
        return float("inf")

    # Smooth the sequence using a moving average
    smoothed_sequence = np.convolve(sequence, np.ones(window) / window, mode='valid')

    # Calculate slopes and curvatures
    slopes = np.diff(smoothed_sequence) / smoothed_sequence[:-1]
    curvatures = np.diff(slopes)

    stabilization_count = 0
    for i in range(len(slopes) - window):
        recent_slopes = slopes[i: i + window]
        recent_curvatures = curvatures[i: i + window - 1]

        # Stabilization conditions
        is_stabilized = (
            np.all(np.abs(recent_slopes) < slope_threshold) and
            np.all(np.abs(recent_curvatures) < curvature_threshold)
        )

        # Oscillation detection
        recent_values = smoothed_sequence[i: i + window]
        oscillation_range = np.ptp(recent_values)
        is_oscillating = oscillation_range < oscillation_tolerance

        # Increasing trend detection
        has_increasing_trend = np.all(recent_slopes > increasing_trend_threshold)

        # Flat change detection
        flat_change = np.abs(smoothed_sequence[i + window - 1] - smoothed_sequence[i]) < flat_change_threshold

        if is_stabilized or is_oscillating or flat_change:
            stabilization_count += 1
            if stabilization_count >= patience:
                return i + window
        elif has_increasing_trend:
            return i + window
        else:
            stabilization_count = 0

    return float("inf")
