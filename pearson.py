import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle as pk

# 1. Load Data
# Shape: (1000 instances, 60 time steps, 24 variables)
# Replace with your actual data
N_INSTANCES = 1000
N_STEPS = 60
N_VARS = 24

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

# 3. Stack (Pool) the Data
# Concatenate all instances vertically into one giant matrix
# New shape: (60,000, 24)
data_pooled = np.vstack(dat0)

# Convert to DataFrame for easy correlation calculation
df_pooled = pd.DataFrame(data_pooled, columns=var_names)

# 4. Calculate Pearson Correlation
# This computes the standard R value (-1 to 1)
corr_matrix = np.abs(df_pooled.corr(method='pearson'))

# 5. Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="OrRd", center=0, vmin=0, vmax=1,
            xticklabels=var_names, yticklabels=var_names, annot=False)
plt.title("Pearson Correlation (Global Synchronized Similarity), Absolute Value, Non-Flare")
plt.show()

data_pooled = np.vstack(dat1)

# Convert to DataFrame for easy correlation calculation
df_pooled = pd.DataFrame(data_pooled, columns=var_names)

# 4. Calculate Pearson Correlation
# This computes the standard R value (-1 to 1)
corr_matrix = np.abs(df_pooled.corr(method='pearson'))

# 5. Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="OrRd", center=0, vmin=0, vmax=1,
            xticklabels=var_names, yticklabels=var_names, annot=False)
plt.title("Pearson Correlation (Global Synchronized Similarity), Absolute Value, Flare > C")
plt.show()