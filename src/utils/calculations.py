"""
Skill Trend Analysis Module

This module provides functions for analyzing time series trends in skill data.
Features:
- Exponential smoothing with Numba acceleration
- Rolling metrics calculation (CAGR, momentum, etc.)
- Trend classification into 10 categories
- Robust growth rate calculations
- All functions are Numba-optimized for performance

Example:
--------
>>> data = np.random.rand(100, 24)  # 100 skills, 24 months
>>> results = process_skill_trends(data, alpha=0.3, window=6)
>>> print(results['categories_str'][:5])
"""

import numpy as np 
from numba import njit, prange

# -------------------------
# Input validation
# -------------------------
def validate_inputs(y: np.ndarray, alpha: float):
    """Validate input arrays and parameters"""
    if y.ndim != 2:
        raise ValueError(f"Expected 2D array, got {y.ndim}D")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if y.shape[1] < 2:
        raise ValueError(f"Need at least 2 time periods, got {y.shape[1]}")


# -------------------------
# 1. Exponential smoothing
# -------------------------
@njit(fastmath=True, parallel=True)
def exp_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential smoothing to each row of the input array.
    
    Parameters:
    -----------
    y : np.ndarray
        2D array of shape (n_skills, n_time)
    alpha : float
        Smoothing parameter (0-1), higher = more weight to recent values
    
    Returns:
    --------
    np.ndarray
        Smoothed time series with same shape as input
    """
    n_rows, n_cols = y.shape
    out = np.empty_like(y)
    for i in prange(n_rows):
        current = y[i, 0]
        out[i, 0] = current
        for j in range(1, n_cols):
            current = alpha * y[i, j] + (1 - alpha) * current
            out[i, j] = current
    return out


# -------------------------
# 2. Rolling metrics
# -------------------------
@njit(fastmath=True, parallel=True)
def calculate_rolling_metrics(
    y_smoothed: np.ndarray, 
    window: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rolling metrics for each skill time series.
    
    Parameters:
    -----------
    y_smoothed : np.ndarray
        Smoothed time series of shape (n_skills, n_time)
    window : int
        Window size for recent momentum calculation
    
    Returns:
    --------
    tuple of four np.ndarray:
        cagr : Annualized growth rate (%)
        recent_momentum : Recent momentum (% change)
        current_prevalence : Latest value
        peak_ratio : Current value relative to historical peak
    """
    n_skills, n_time = y_smoothed.shape
    cagr = np.zeros(n_skills, dtype=np.float64)
    recent_momentum = np.zeros(n_skills, dtype=np.float64)
    current_prevalence = np.zeros(n_skills, dtype=np.float64)
    peak_ratio = np.zeros(n_skills, dtype=np.float64)

    for i in prange(n_skills):
        start = y_smoothed[i, 0]
        end = y_smoothed[i, -1]
        n_periods = n_time - 1
        current_prevalence[i] = end
        
        # Calculate CAGR
        if start > 0 and n_periods > 0:
            cagr[i] = ((end / start) ** (12 / n_periods) - 1) * 100
        else:
            cagr[i] = 0.0

        # Calculate recent momentum
        if n_time >= 2 * window:
            recent_avg = np.mean(y_smoothed[i, -window:])
            prev_avg = np.mean(y_smoothed[i, -2 * window:-window])
            if prev_avg > 0:
                recent_momentum[i] = ((recent_avg - prev_avg) / prev_avg) * 100
            else:
                recent_momentum[i] = 0.0
        else:
            recent_momentum[i] = 0.0

        # Calculate peak ratio
        peak = np.max(y_smoothed[i, :])
        if peak > 0:
            peak_ratio[i] = end / peak
        else:
            peak_ratio[i] = 0.0

    return cagr, recent_momentum, current_prevalence, peak_ratio


# -------------------------
# 3. OLS slopes/intercepts (vectorized)
# -------------------------
@njit(fastmath=True, parallel=True)
def batch_ols(y_smoothed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate linear trend (slope and intercept) for each time series.
    Uses ordinary least squares regression.
    
    Parameters:
    -----------
    y_smoothed : np.ndarray
        Smoothed time series of shape (n_skills, n_time)
    
    Returns:
    --------
    tuple:
        slopes : Linear trend slopes
        intercepts : Intercept values
    """
    n_skills, n_time = y_smoothed.shape
    slopes = np.zeros(n_skills, dtype=np.float64)
    intercepts = np.zeros(n_skills, dtype=np.float64)
    
    x = np.arange(n_time, dtype=np.float64)
    x_mean = np.mean(x)
    
    for i in prange(n_skills):
        y_series = y_smoothed[i, :]
        y_mean = np.mean(y_series)
        
        numerator = 0.0
        denominator = 0.0
        
        for j in range(n_time):
            x_diff = x[j] - x_mean
            numerator += x_diff * (y_series[j] - y_mean)
            denominator += x_diff * x_diff
        
        if denominator != 0:
            slopes[i] = numerator / denominator
            intercepts[i] = y_mean - slopes[i] * x_mean
        else:
            slopes[i] = 0.0
            intercepts[i] = y_mean
    
    return slopes, intercepts


# -------------------------
# 4. Nonlinearity
# -------------------------
@njit(fastmath=True, parallel=True)
def calculate_nonlinearity(
    y_smoothed: np.ndarray, 
    slopes: np.ndarray, 
    intercepts: np.ndarray
) -> np.ndarray:
    """
    Calculate nonlinearity measure for each time series.
    
    Parameters:
    -----------
    y_smoothed : np.ndarray
        Smoothed time series
    slopes : np.ndarray
        Linear slopes from OLS
    intercepts : np.ndarray
        Intercepts from OLS
    
    Returns:
    --------
    np.ndarray
        Nonlinearity measure (std of residuals / mean absolute residuals)
    """
    n_skills, n_time = y_smoothed.shape
    nonlinearity = np.zeros(n_skills, dtype=np.float64)
    
    for i in prange(n_skills):
        linear_pred = intercepts[i] + slopes[i] * np.arange(n_time)
        residuals = y_smoothed[i, :] - linear_pred
        mean_abs = np.mean(np.abs(residuals))
        
        if mean_abs > 0:
            nonlinearity[i] = np.std(residuals) / mean_abs
        else:
            nonlinearity[i] = 0.0
    
    return nonlinearity


# -------------------------
# 5. Smoothed month-over-month growth
# -------------------------
def calculate_smoothed_growth(
    y: np.ndarray, 
    alpha: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute robust growth as average month-over-month % change using smoothed series.
    Ensures NaN-free results even with zeros in the series.
    
    Parameters:
    -----------
    y : np.ndarray
        2D array of shape (n_skills, n_time)
    alpha : float
        Smoothing parameter (0-1)
    
    Returns:
    --------
    tuple:
        growth_pct : Percentage growth for each skill
        y_smoothed : Smoothed time series
    """
    # Validate inputs
    validate_inputs(y, alpha)
    
    y_smoothed = exp_smooth(y, alpha)
    
    # Add epsilon relative to data scale to avoid zero division
    eps = 1e-6 * np.max(np.abs(y_smoothed)) + 1e-12
    prev = y_smoothed[:, :-1] + eps
    next_ = y_smoothed[:, 1:]
    
    mom_growth = (next_ - prev) / prev
    
    # Handle all-zero series
    all_zero = np.all(np.abs(y_smoothed) < eps, axis=1)
    growth_pct = np.mean(mom_growth, axis=1) * 100
    growth_pct = np.where(all_zero, 0.0, growth_pct)
    
    # Clip extreme outliers
    growth_pct = np.clip(growth_pct, -999, 999)
    
    return growth_pct, y_smoothed


# -------------------------
# 6. Percentile helpers
# -------------------------
@njit(fastmath=True)
def percentile_1d(arr: np.ndarray, q: float) -> float:
    """Compute percentile manually for 1D array in Numba"""
    if len(arr) == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    idx = int(q * (len(arr) - 1))
    return sorted_arr[idx]


@njit
def calculate_recent_percentile(y_smoothed: np.ndarray, window: int = 6) -> np.ndarray:
    """
    Calculate what percentile the latest month is within the recent window.
    
    Parameters:
    -----------
    y_smoothed : np.ndarray
        Smoothed time series
    window : int
        Window size for recent history
    
    Returns:
    --------
    np.ndarray
        Percentile values (0-1) for each skill
    """
    n_skills, n_time = y_smoothed.shape
    recent_percentiles = np.zeros(n_skills, dtype=np.float64)
    
    for i in range(n_skills):
        if n_time >= window:
            recent_data = y_smoothed[i, -window:]
            current_val = recent_data[-1]
            
            count_below = 0
            for j in range(window):
                if recent_data[j] <= current_val:
                    count_below += 1
            recent_percentiles[i] = count_below / window
        else:
            recent_percentiles[i] = 0.5  # Neutral baseline
    
    return recent_percentiles


# -------------------------
# 7. Trend classification
# -------------------------
TREND_MAP = [
    "Hot & Growing",      # 0 - High demand, increasing rapidly
    "Stable & In-Demand", # 1 - Established, steady demand  
    "Niche & Specialized", # 2 - Low but consistent demand
    "Declining & Risky",  # 3 - Decreasing demand
    "Emerging & New"      # 4 - Just appearing, potential future
]

@njit
def classify_trends_smart_recent(
    y_smoothed: np.ndarray,
    slopes: np.ndarray,
    recent_momentum: np.ndarray,
    current_prevalence: np.ndarray,
    p10: float, p25: float, p75: float, p90: float
) -> np.ndarray:
    """
    SIMPLIFIED 5-category classification for student decision-making.
    All function names maintained for pipeline compatibility.
    """
    n_skills, n_time = y_smoothed.shape
    categories = np.zeros(n_skills, dtype=np.int8)
    
    # Calculate annual growth from smoothed data
    annual_growth = np.zeros(n_skills, dtype=np.float64)
    for i in range(n_skills):
        start = y_smoothed[i, 0]
        end = y_smoothed[i, -1]
        n_periods = n_time - 1
        if start > 0 and n_periods > 0:
            annual_growth[i] = ((end / start) ** (12 / n_periods) - 1) * 100
    
    for i in range(n_skills):
        slope = slopes[i]
        momentum = recent_momentum[i]
        current_val = current_prevalence[i]
        growth = annual_growth[i]
        
        # 1. Emerging & New (4)
        # Very low current usage but strong growth
        if current_val < 2.0 and growth > 20.0:
            categories[i] = 4  # Emerging & New
            continue
            
        # 2. Hot & Growing (0)
        # Strong growth AND meaningful current usage
        if growth > 10.0 and current_val > 5.0:
            # Additional momentum check for reliability
            if momentum >= p75:  # Top quartile momentum
                categories[i] = 0  # Hot & Growing
                continue
                
        # 3. Declining & Risky (3)
        # Clear decline with conservative thresholds for high prevalence
        if growth < -5.0:  # Significant annual decline
            if current_val > 50.0:
                # High prevalence: require stronger evidence
                if slope < -0.1 or momentum <= p10:  # Steep decline or bottom decile
                    categories[i] = 3  # Declining & Risky
                else:
                    # Could be fluctuation, check next category
                    pass
            else:
                # Lower prevalence: standard threshold
                categories[i] = 3  # Declining & Risky
                continue
        # Additional check for steady decline not caught by annual growth
        elif slope < -0.08 and momentum <= p25:
            if current_val > 50.0 and slope < -0.12:  # Extra conservative for high prevalence
                categories[i] = 3  # Declining & Risky
            elif current_val <= 50.0:
                categories[i] = 3  # Declining & Risky
            continue
            
        # 4. Niche & Specialized (2)
        # Low prevalence but consistent (not declining rapidly)
        if current_val < 5.0 and abs(growth) < 8.0 and slope > -0.05:
            # Check if it's stable at low level (not emerging, not declining)
            categories[i] = 2  # Niche & Specialized
            continue
            
        # 5. Stable & In-Demand (1) - DEFAULT
        # Everything else: established skills with moderate/stable demand
        categories[i] = 1  # Stable & In-Demand
                
    return categories

def map_categories_to_strings(categories_int: np.ndarray) -> np.ndarray:
    """
    Maps the integer category indices back to their string names.
    
    Parameters:
    -----------
    categories_int : np.ndarray
        Integer category indices
    
    Returns:
    --------
    np.ndarray
        String category names
    """
    n = len(categories_int)
    categories_str = np.empty(n, dtype=object) 
    for i in range(n):
        if 0 <= categories_int[i] < len(TREND_MAP):
            categories_str[i] = TREND_MAP[categories_int[i]]
        else:
            categories_str[i] = "UNKNOWN"
    return categories_str


# -------------------------
# 8. Main processing function
# -------------------------
def process_skill_trends(
    y: np.ndarray,
    alpha: float = 0.3,
    window: int = 6
) -> dict:
    """
    Process skill trends with all metrics.
    
    Parameters:
    -----------
    y : np.ndarray
        2D array of shape (n_skills, n_time)
    alpha : float
        Smoothing parameter (0-1)
    window : int
        Window size for recent calculations
    
    Returns:
    --------
    dict
        Dictionary with all calculated metrics:
        - growth_pct: Percentage growth
        - y_smoothed: Smoothed time series
        - cagr: Annualized growth rate
        - recent_momentum: Recent momentum
        - current_prevalence: Latest values
        - peak_ratio: Current/peak ratio
        - slopes: Linear slopes
        - intercepts: Linear intercepts
        - nonlinearity: Nonlinearity measure
        - categories_int: Integer categories
        - categories_str: String categories
        - trend_categories: Alias for categories_str
    """
    # Calculate smoothed growth and get smoothed series
    growth_pct, y_smoothed = calculate_smoothed_growth(y, alpha)
    
    # Calculate all metrics
    cagr, recent_momentum, current_prevalence, peak_ratio = \
        calculate_rolling_metrics(y_smoothed, window)
    
    slopes, intercepts = batch_ols(y_smoothed)
    nonlinearity = calculate_nonlinearity(y_smoothed, slopes, intercepts)
    
    # Calculate percentiles for classification
    momentum_sorted = np.sort(recent_momentum)
    n = len(momentum_sorted)
    p10 = momentum_sorted[int(0.1 * n)] if n > 0 else 0
    p25 = momentum_sorted[int(0.25 * n)] if n > 0 else 0
    p75 = momentum_sorted[int(0.75 * n)] if n > 0 else 0
    p90 = momentum_sorted[int(0.9 * n)] if n > 0 else 0
    
    # Classify trends
    categories_int = classify_trends_smart_recent(
        y_smoothed, slopes, recent_momentum, 
        current_prevalence, p10, p25, p75, p90
    )
    categories_str = map_categories_to_strings(categories_int)
    
    return {
        'growth_pct': growth_pct,
        'y_smoothed': y_smoothed,
        'cagr': cagr,
        'recent_momentum': recent_momentum,
        'current_prevalence': current_prevalence,
        'peak_ratio': peak_ratio,
        'slopes': slopes,
        'intercepts': intercepts,
        'nonlinearity': nonlinearity,
        'categories_int': categories_int,
        'categories_str': categories_str,
        'trend_categories': categories_str  # alias for backward compatibility
    }


# -------------------------
# 9. Memory-efficient processing for large datasets
# -------------------------
def process_skill_trends_large(
    y: np.ndarray, 
    alpha: float = 0.3, 
    window: int = 6, 
    chunk_size: int = 1000
) -> dict:
    """
    Process in chunks for memory efficiency with large datasets.
    
    Parameters:
    -----------
    y : np.ndarray
        2D array of shape (n_skills, n_time)
    alpha : float
        Smoothing parameter
    window : int
        Window size
    chunk_size : int
        Number of skills to process at once
    
    Returns:
    --------
    dict
        Same structure as process_skill_trends
    """
    n_skills = y.shape[0]
    results_accumulated = {}
    
    # Process in chunks
    for start in range(0, n_skills, chunk_size):
        end = min(start + chunk_size, n_skills)
        chunk = y[start:end]
        chunk_results = process_skill_trends(chunk, alpha, window)
        
        # Accumulate results
        for key, value in chunk_results.items():
            if key not in results_accumulated:
                results_accumulated[key] = []
            results_accumulated[key].append(value)
    
    # Concatenate all chunks
    final_results = {}
    for key in results_accumulated:
        final_results[key] = np.concatenate(results_accumulated[key])
    
    return final_results

