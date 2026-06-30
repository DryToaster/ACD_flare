import pickle
import numpy as np

def get_label_percentages(pickle_file_path):
    """
    Loads a pickled NumPy array of labels (1 by n) and calculates the 
    percentage representation of each unique label.
    """
    # Open the pickle file in binary read mode
    with open(pickle_file_path, 'rb') as file:
        arr = pickle.load(file)
        
    # Ensure the loaded object is a NumPy array
    if not isinstance(arr, np.ndarray):
        raise TypeError("The pickled object is not a NumPy ndarray.")
        
    # Flatten ensures that a shape of (1, n) is treated strictly as a 1D sequence of n elements
    arr_flat = arr.flatten()
    
    # Handle the edge case of an empty array
    total_elements = arr_flat.size
    if total_elements == 0:
        return {} 
        
    # np.unique with return_counts=True finds all distinct labels and how often they appear
    unique_labels, counts = np.unique(arr_flat, return_counts=True)
    
    # Calculate the percentage for each count
    percentages = (counts / total_elements) * 100
    
    # Zip the labels and percentages together into a dictionary for easy reading
    label_distribution = dict(zip(unique_labels, percentages))
    
    return label_distribution

# --- Example Usage ---
percentages = get_label_percentages('acd/Cleaned-SWANSF-Dataset/test/Partition5_Labels_LSBZM-Norm_FPCKNN-impute.pkl')
for label, percent in percentages.items():
    print(f"Label {label}: {percent:.2f}%")