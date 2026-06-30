import pickle
import numpy as np

def apply_differencing_and_save(input_filepath, output_filepath):
    """
    Loads a .pkl time series array, applies first-order differencing 
    along the time axis, and saves the new array.
    """
    # 1. Load the original data
    with open(input_filepath, 'rb') as f:
        data = pickle.load(f)
        
    data = np.asarray(data)
    print(f"Original shape: {data.shape} (Instances, Time Steps, Variables)")
    
    # 2. Apply first-order differencing along the time axis (axis=1)
    # This computes: new_data[i, t, v] = data[i, t+1, v] - data[i, t, v]
    differenced_data = np.diff(data, n=1, axis=1)
    
    # Note: Differencing reduces the time step count by exactly 1.
    print(f"Differenced shape: {differenced_data.shape}")
    
    # 3. Save the differenced dataset
    with open(output_filepath, 'wb') as f:
        pickle.dump(differenced_data, f)
        
    print(f"Differenced dataset successfully saved to: {output_filepath}")
    
    return differenced_data

# Example execution:
apply_differencing_and_save('./acd/Cleaned-SWANSF-Dataset/test/Partition5_LSBZM-Norm_FPCKNN-impute.pkl', 'diff5.pkl')

# You can then run your previous stationarity test on the new file:
#test_stationarity_from_pickle('differenced_p1.pkl')