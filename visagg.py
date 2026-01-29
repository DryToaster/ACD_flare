import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os

def visualize_average_matrices_v2():
    # Paths based on your directory structure
    label_path = ".//test//Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
    # Mapping display names: we use the filename for loading but display "VAR" for Granger
    methods = {
        "Granger": "VAR", 
        "DTW": "DTW", 
        "Pearson": "Pearson", 
        "XCorr": "XCorr", 
        "ACD": "ACD"
    }
    
    # Load Labels
    print(f"Loading labels from {label_path}...")
    with open(label_path, 'rb') as f:
        labels = np.array(pk.load(f)).astype(int)

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # Set up the visualization grid
    # Increased figsize height (from 4 * len to 5 * len) to accommodate more space
    fig, axes = plt.subplots(len(methods), 2, figsize=(12, 5 * len(methods)))

    for i, (file_key, display_name) in enumerate(methods.items()):
        file_name = f"TRAIN_{file_key}.npy"
        
        if os.path.exists(file_name):
            print(f"Processing {display_name}...")
            data = np.load(file_name, mmap_mode='r')
            
            avg_pos = np.mean(data[pos_indices], axis=0)
            avg_neg = np.mean(data[neg_indices], axis=0)
            
            if avg_pos.ndim == 1:
                side = int(np.sqrt(avg_pos.shape[0]))
                avg_pos = avg_pos.reshape(side, side)
                avg_neg = avg_neg.reshape(side, side)
                
            # Plot Positive Heatmap
            im_pos = axes[i, 0].imshow(avg_pos, cmap='magma', interpolation='nearest')
            axes[i, 0].set_title(f"{display_name} - Average Positive (Flare)")
            plt.colorbar(im_pos, ax=axes[i, 0])
            
            # Plot Negative Heatmap
            im_neg = axes[i, 1].imshow(avg_neg, cmap='magma', interpolation='nearest')
            axes[i, 1].set_title(f"{display_name} - Average Negative (Quiet)")
            plt.colorbar(im_neg, ax=axes[i, 1])
        else:
            for col in range(2):
                axes[i, col].text(0.5, 0.5, f"File Not Found:\n{file_name}", 
                                 ha='center', va='center', color='red')
                axes[i, col].set_title(f"{display_name} - MISSING")

    # Add vertical spacing (hspace) between rows
    plt.subplots_adjust(hspace=0.4) 
    
    plt.suptitle("Class-Specific Average Causal Matrices (Partition 1)", fontsize=16, y=0.98)
    plt.savefig("average_causal_matrices_v2.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_average_matrices_v2()