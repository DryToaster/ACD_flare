import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.linear_model import Ridge
from scipy.stats import f

n_instances = 1000
n_steps = 60
n_vars = 24

var_names = ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX',
 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX']

# 1. Load Data
# ---------------------------------------------------------
data = None
with open(".\\test\\Partition1_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as _part:
    data = pk.load(_part, ) 
data += np.random.normal(0, 1e-4, (73492, 60, 24))
print("Shape: ", data.shape)

fi = open(".//test//Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl", 'rb')
labs = pk.load(fi)
fi.close()

dat0 = np.ndarray((1000, 60, 24))
dat1 = np.ndarray((1000, 60, 24))
d0c = 0
d0i = 0
d1c = 0
d1i = 0
while d0c < 1000:
    if labs[d0i] == 0:
        dat0[d0c] = data[d0i]
        d0c += 1
    d0i += 1

while d1c < 1000:
    if labs[d1i] == 1:
        dat1[d1c] = data[d1i]
        d1c += 1
    d1i += 1






# 2. Reshape Data with NaN Buffers
# ---------------------------------------------------------
# We create a buffer of NaNs to insert between instances
# Shape: (1000, 1, 24)
nan_buffer = np.full((n_instances, 1, n_vars), np.nan)

# Concatenate the buffer to the end of each instance
# New shape per instance: (61, 24) -> The last row is NaN
dat1b = np.concatenate([dat1, nan_buffer], axis=1)
dat0b = np.concatenate([dat0, nan_buffer], axis=1)
# Flatten into 2D: (1000 * 61, 24)
# The NaNs effectively "reset" the lag memory between instances
dat1_2d = dat1b.reshape(-1, n_vars)
dat0_2d = dat0b.reshape(-1, n_vars)

# Convert to DataFrame
df_0 = pd.DataFrame(dat0_2d, columns=var_names)
df_1 = pd.DataFrame(dat1_2d, columns=var_names)

print(f"Pooled Data Shape: {df_1}")
# You will see rows with NaNs every 61st row
print("First separation check (rows 58-62):")
print(df_0.iloc[58:63])




# 3. Train One Global VAR Model
# ---------------------------------------------------------
# We use missing='drop' (default) so it skips the NaNs during training

def calculate_granger_ridge(data_3d, lag=1, alpha=1.0):
    n_inst, n_steps, n_vars = data_3d.shape
    
    # 1. Prepare global X (lags) and Y (current)
    X_list, Y_list = [], []
    for i in range(n_inst):
        X_list.append(data_3d[i, :-lag, :])
        Y_list.append(data_3d[i, lag:, :])
    
    X_full = np.vstack(X_list)  # Shape: (N, n_vars * lag)
    Y_full = np.vstack(Y_list)  # Shape: (N, n_vars)
    
    n_samples = X_full.shape[0]
    adj_matrix = np.zeros((n_vars, n_vars)) # To store p-values

    print(f"Fitting models for {n_vars} variables...")

    for target_idx in range(n_vars):
        y = Y_full[:, target_idx]
        
        # --- FULL MODEL ---
        # Predict target using ALL variables
        model_full = Ridge(alpha=alpha)
        model_full.fit(X_full, y)
        rss_full = np.sum((y - model_full.predict(X_full)) ** 2)
        
        for source_idx in range(n_vars):
            if target_idx == source_idx:
                adj_matrix[source_idx, target_idx] = 1.0
                continue
            
            # --- RESTRICTED MODEL ---
            # Remove the columns associated with the 'source' variable
            # (In lag 1, this is just one column. In lag p, it's p columns)
            cols_to_keep = [c for c in range(n_vars) if c != source_idx]
            X_restricted = X_full[:, cols_to_keep]
            
            model_restricted = Ridge(alpha=alpha)
            model_restricted.fit(X_restricted, y)
            rss_restricted = np.sum((y - model_restricted.predict(X_restricted)) ** 2)
            
            # --- F-TEST ---
            # Number of restrictions (m) is the lag order
            m = lag 
            # Degrees of freedom (n - k)
            df = n_samples - (n_vars * lag)
            
            f_stat = ((rss_restricted - rss_full) / m) / (rss_full / df)
            p_value = f.sf(f_stat, m, df) # Survival function (1-CDF)
            
            adj_matrix[source_idx, target_idx] = p_value

    return adj_matrix



results0 = calculate_granger_ridge(dat0)
results1 = calculate_granger_ridge(dat1)

# 5. Visualize
# ---------------------------------------------------------
plt.figure(figsize=(12, 10))
# Plotting 1 - p_value so that High Value (Red) = High Confidence of Causality
# If p=0.01, value is 0.99 (Strong link)
sns.heatmap(1 - results1, cmap="OrRd", vmin=0.0, vmax=1.0,
            xticklabels=var_names, yticklabels=var_names)
plt.title(f"Global Causal Links (1-p), Flare > C | Lag 1")
plt.xlabel("Target")
plt.ylabel("Source")
plt.show()

plt.figure(figsize=(12, 10))
# Plotting 1 - p_value so that High Value (Red) = High Confidence of Causality
# If p=0.01, value is 0.99 (Strong link)
sns.heatmap(1 - results0, cmap="OrRd", vmin=0.0, vmax=1.0,
            xticklabels=var_names, yticklabels=var_names)
plt.title(f"Global Causal Links (1-p), Non-Flare | Lag 1")
plt.xlabel("Target")
plt.ylabel("Source")
plt.show()