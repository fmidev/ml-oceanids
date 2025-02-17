from scipy.optimize import curve_fit

def fit_correction_function(control_values, historical_values, degree=3):
    """
    Fit a polynomial correction function using quantile mapping between 
    control forecasts and historical observations.
    
    Parameters:
    control_values: ECXSF control member (0) forecasts
    historical_values: Observed historical values
    degree: Degree of polynomial to fit (default=3)
    
    Returns:
    function that takes forecast value and returns corrected value
    """
    # Get sorted arrays for quantile mapping
    sorted_control = np.sort(control_values)
    sorted_historical = np.sort(historical_values)
    
    # Get equally spaced percentiles
    percentiles = np.linspace(0, 100, len(sorted_control))
    control_quantiles = np.percentile(sorted_control, percentiles)
    hist_quantiles = np.percentile(sorted_historical, percentiles)
    
    # Fit polynomial to the quantile pairs
    coeffs = np.polyfit(control_quantiles, hist_quantiles, degree)
    
    # Create correction function using the fitted polynomial
    def correct_forecast(x):
        return np.polyval(coeffs, x)
    
    return correct_forecast

def evaluate_correction(control_values, historical_values, correction_fn):
    """
    Evaluate the correction function performance
    """
    corrected = correction_fn(control_values)
    
    # Calculate error metrics
    mae_before = np.mean(np.abs(control_values - historical_values))
    mae_after = np.mean(np.abs(corrected - historical_values))
    
    print(f"Mean Absolute Error before correction: {mae_before:.2f}")
    print(f"Mean Absolute Error after correction: {mae_after:.2f}")
    
    return corrected
