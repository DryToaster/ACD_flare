import pickle
import numpy as np
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed

def _test_single_series(v_idx, series, alpha):
    """
    Worker function to test a single time series for stationarity.
    Returns the variable index and a 1 (failed) or 0 (passed).
    """
    # Edge case 1: Handle perfectly constant series (variance = 0)
    if np.var(series) == 0:
        return v_idx, 1
        
    try:
        # Perform the ADF test
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        
        # 1 means failed to reject null (non-stationary), 0 means stationary
        return v_idx, 1 if p_value > alpha else 0
        
    except ValueError:
        # Edge case 2: ADF test fails entirely (NaNs, infs)
        return v_idx, 1

def test_stationarity_from_pickle(file_path, alpha=0.10, n_jobs=-1):
    """
    Loads a .pkl file containing time series data and tests for stationarity
    in parallel using the Augmented Dickey-Fuller (ADF) test.
    
    Parameters:
    - file_path: Path to the pickle file.
    - alpha: Significance level for the ADF test.
    - n_jobs: Number of CPU cores to use. -1 uses all available cores.
    """
    # Load the data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    # Ensure it is a numpy array
    data = np.asarray(data)
    
    # Check assumptions about the data structure
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D array (instances, time_steps, variables), got shape {data.shape}")
        
    num_instances, time_steps, num_variables = data.shape
    total_tests = num_instances * num_variables
    
    print(f"Loaded data: {num_instances} instances | {time_steps} time steps | {num_variables} variables.")
    print(f"Starting {total_tests} parallel jobs across {n_jobs if n_jobs != -1 else 'all'} cores...\n")
    
    # Create the task generator
    # We pass the variable index (v) so we know which variable's failure count to increment later
    tasks = (
        delayed(_test_single_series)(v, data[i, :, v], alpha)
        for i in range(num_instances)
        for v in range(num_variables)
    )
    
    # Execute tasks in parallel
    # Note: verbose=5 provides a text-based progress bar in the console
    results = Parallel(n_jobs=n_jobs, verbose=5)(tasks)
    
    # Aggregate results
    variable_failures = np.zeros(num_variables)
    total_failures = 0
    
    for v_idx, failed in results:
        variable_failures[v_idx] += failed
        total_failures += failed

    # Print the statistical report
    print("\n--- Stationarity Test Failure Report ---")
    print(f"Significance level (alpha): {alpha}")
    print("Note: 'Failure' indicates the series is statistically non-stationary (failed to reject null hypothesis).\n")
    
    for v in range(num_variables):
        fail_pct = (variable_failures[v] / num_instances) * 100
        print(f"Variable {v+1:02d}: {fail_pct:>6.2f}% failure rate ({int(variable_failures[v])}/{num_instances})")
        
    overall_fail_pct = (total_failures / total_tests) * 100
    print("-" * 40)
    print(f"Overall Failure Percentage: {overall_fail_pct:.2f}% ({total_failures}/{total_tests} tests failed)")

# --- Example Usage ---
# test_stationarity_from_pickle('my_time_series_data.pkl', n_jobs=-1)
# Example execution:

test_stationarity_from_pickle('.\\acd\\Cleaned-SWANSF-Dataset\\train\\Partition5_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl')
test_stationarity_from_pickle('./SWANSF-diff/train/diff5.pkl')