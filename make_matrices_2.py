import numpy as np
import pickle as pk
import warnings
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import t as t_dist
from scipy.fft import fft, ifft
from dtaidistance import dtw

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. OPTIMIZED MATH FUNCTIONS (Fast Versions)
# ==========================================

def fast_granger_pvalues(data, lag=1):
    """
    Computes VAR(1) p-values using pure OLS (No statsmodels overhead).
    Returns matrix of 1 - p_value (Strength).
    """
    T, V = data.shape
    # Prepare Lagged Data (VAR(1))
    Y = data[lag:]
    X = data[:-lag]
    # Add Intercept
    X_const = np.column_stack([np.ones(T-lag), X])
    
    try:
        # Coefficients B: (X'X)^-1 X'Y
        B, residuals, rank, s = np.linalg.lstsq(X_const, Y, rcond=None)
        
        # Residuals & Sigma
        E = Y - np.dot(X_const, B)
        sigma_sq = np.sum(E**2, axis=0) / (T - lag - V - 1)
        
        # Standard Errors
        XTX_inv = np.linalg.inv(np.dot(X_const.T, X_const))
        var_scaling = np.diag(XTX_inv)[1:] 
        
        coeffs = B[1:, :]
        std_errs = np.sqrt(np.outer(var_scaling, sigma_sq))
        
        # t-stats and p-values
        t_stats = coeffs / std_errs
        df = T - lag - V - 1
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df))
        
        # Transpose: Source -> Target
        return 1 - p_values.T 
    except:
        return np.zeros((V, V))

def fast_xcorr_matrix(data):
    """
    Computes Max Absolute Cross-Correlation using FFT (Vectorized).
    Corrected to subtract mean (centering) so it behaves like Pearson with lags.
    """
    # 1. CENTER THE DATA (Crucial Correction)
    # Without this, DC offsets dominate the correlation.
    data_centered = data - np.mean(data, axis=0)
    
    # 2. Pad to next power of 2 for speed and to avoid circular wrapping artifacts
    # T=60 -> 2*60-1 = 119. Next power of 2 is 128.
    n_fft = 128 
    
    # 3. FFT
    f_data = fft(data_centered, n=n_fft, axis=0)
    
    # 4. Compute Cross-Power Spectrum (Broadcasting V x V)
    # (N, V, 1) * (N, 1, V) -> (N, V, V)
    spectrum = f_data[:, :, None] * np.conj(f_data[:, None, :])
    
    # 5. Inverse FFT to get Cross-Correlation in time domain
    corr_time = ifft(spectrum, axis=0).real
    
    # 6. Find Max Absolute Correlation across all lags
    max_corr = np.max(np.abs(corr_time), axis=0)
    
    # 7. Normalize (Covariance -> Correlation)
    # The diagonal of max_corr contains the auto-correlation at lag 0 (Energy)
    d = np.sqrt(np.diag(max_corr))
    
    # Avoid divide by zero if a variable is a flat line
    d[d == 0] = 1 
    
    # Outer product to get denominator matrix
    norm_corr = max_corr / np.outer(d, d)
    
    # Clip to [0, 1] range just in case of floating point epsilon errors
    norm_corr = np.clip(norm_corr, 0, 1)
    
    return norm_corr

# ==========================================
# 2. WORKER FUNCTION
# ==========================================

def process_sample(index, name):
    try:
        # Access data from Global Scope (Zero-Copy)
        sample = None
        if name == 'TRAIN':
            sample = GLOB_TRAIN_DATA[index]
        else:
            sample = GLOB_TEST_DATA[index]

        # --- A. Pearson ---
        pearson_mat = np.abs(np.corrcoef(sample, rowvar=False))

        # --- B. DTW ---
        # Fallback to Python-only if C-library fails
        try:
            dtw_mat = dtw.distance_matrix_fast(sample.T, compact=False)
        except:
            dtw_mat = np.zeros((sample.shape[1], sample.shape[1]))

        # --- C. Fast X-Corr (FFT) ---
        xcorr_mat = fast_xcorr_matrix(sample)
        
        # --- D. Fast Granger (OLS) ---
        granger_mat = fast_granger_pvalues(sample, lag=1)

        # --- CLEANUP (Crucial for NaNs) ---
        # Replace NaNs/Infs immediately
        pearson_mat = np.nan_to_num(pearson_mat, nan=0.0)
        dtw_mat     = np.nan_to_num(dtw_mat, nan=0.0)
        xcorr_mat   = np.nan_to_num(xcorr_mat, nan=0.0)
        granger_mat = np.nan_to_num(granger_mat, nan=0.0)

        return granger_mat, dtw_mat, pearson_mat, xcorr_mat

    except Exception as e:
        logging.error(f"Failed on {name} sample {index} with error: {e}")
        # Return zeros on failure
        V = 24
        z = np.zeros((V, V))
        return z, z, z, z

def generate_and_save(n_samples, name="TRAIN"):
    print(f"\nGenerating matrices for {name} ({n_samples} samples)...")
    
    # Use 'threading' backend to access GLOBALS without pickling (Fixes freezing)
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_sample)(i, name) for i in tqdm(range(n_samples))
    )
    
    print(f"Stacking arrays for {name}...")
    granger = np.array([r[0] for r in results])
    dtw_mat = np.array([r[1] for r in results])
    pearson = np.array([r[2] for r in results])
    xcorr   = np.array([r[3] for r in results])
    
    print(f"Saving {name} .npy files...")
    np.save(f"{name}_Granger.npy", granger)
    np.save(f"{name}_DTW.npy", dtw_mat)
    np.save(f"{name}_Pearson.npy", pearson)
    np.save(f"{name}_XCorr.npy", xcorr)
    
    print(f"Done saving {name}.")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        filename='feature_gen_errors.log', 
        level=logging.ERROR,
        format='%(asctime)s - %(message)s'
    )

    print("--- Loading Data ---")
    
    # LOAD TRAIN
    # Note: We don't need labels (y) just to generate features
    with open(".//test//Partition1_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as f:
        GLOB_TRAIN_DATA = pk.load(f)
        
    # LOAD TEST
    with open(".//test//Partition2_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as f:
        GLOB_TEST_DATA = pk.load(f)

    print(f"Train Data Shape: {GLOB_TRAIN_DATA.shape}")
    print(f"Test Data Shape:  {GLOB_TEST_DATA.shape}")

    # --- RUN GENERATION ---
    
    # 1. Generate TRAIN (and save to disk)
    generate_and_save(len(GLOB_TRAIN_DATA), name="TRAIN")
    
    # 2. Generate TEST (and save to disk)
    generate_and_save(len(GLOB_TEST_DATA), name="TEST")

    print("\nAll matrices generated and saved successfully.")