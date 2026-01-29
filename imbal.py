import pickle as pk
import numpy as np

def count_labels(file_path, label_name):
    try:
        with open(file_path, 'rb') as f:
            labels = np.array(pk.load(f))
        
        # Calculate counts
        total = len(labels)
        positives = np.sum(labels == 1)
        negatives = np.sum(labels == 0)
        
        print(f"--- Analysis for {label_name} ---")
        print(f"File: {file_path}")
        print(f"Total Samples: {total}")
        print(f"Positive (Flares): {positives} ({(positives/total)*100:.4f}%)")
        print(f"Negative (Quiet):  {negatives} ({(negatives/total)*100:.4f}%)")
        
        # Check for any unexpected values (like 2, 3, 4)
        unique_vals = np.unique(labels)
        if len(unique_vals) > 2:
            print(f"Warning: Found non-binary labels: {unique_vals}")
        print("-" * 30)
        
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Update these paths to match your folder structure
    train_path = ".//test//Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
    test_path  = ".//test//Partition2_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
    
    count_labels(train_path, "TRAINING SET")
    count_labels(test_path, "TESTING SET")