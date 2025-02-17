import numpy as np

def apply_quantile_mapping(control_values, historical_values):
    # Compute sorted arrays
    sorted_control = np.sort(control_values)
    sorted_historical = np.sort(historical_values)

    # Compute percentiles
    control_percentiles = np.linspace(0, 1, len(sorted_control))
    
    # Map each control value to its percentile
    corrected = []
    for val in control_values:
        # Find percentile in control distribution
        idx = np.searchsorted(sorted_control, val, side="left")
        frac = control_percentiles[idx-1] if idx > 0 else 0.0
        # Map to historical distribution at the same percentile
        mapped_idx = int(frac * (len(sorted_historical) - 1))
        corrected.append(sorted_historical[mapped_idx])
    return np.array(corrected)
