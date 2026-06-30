import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your utils imported from the codebase
from model.utils import create_rel_rec_send
from model.model_loader import load_encoder

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

class PickledTimeSeriesDataset(Dataset):
    def __init__(self, pickle_path, num_atoms):
        """
        Args:
            pickle_path (str): Path to your .pkl file containing the ndarray.
            num_atoms (int): The number of variables/particles in your dataset (e.g., 24).
        """
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            raw_data = pickle.load(f)
            
        # Ensure it is a numpy array first, then convert to a PyTorch float32 tensor
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
            
        self.data = torch.tensor(raw_data, dtype=torch.float32)

        if len(self.data.shape) == 4:
            self.data = self.data.permute(0, 2, 1, 3)
            
        # If your data is missing the 'dims' dimension entirely: [samples, timesteps, atoms] (3D)
        elif len(self.data.shape) == 3:
            # Swap atoms/timesteps and add a dummy dimension of 1 at the end
            self.data = self.data.permute(0, 2, 1).unsqueeze(-1)

        self.num_samples = self.data.shape[0]
        self.num_atoms = num_atoms
        
        print(f"Successfully loaded dataset! Shape: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a single sample's time-series and a dummy target graph.
        """
        # 1. Get the time-series data for this specific sample
        time_series = self.data[idx]
        
        # 2. Generate a dummy target graph (fully connected)
        # The model extracts off-diagonal edges later, so a matrix of 1s works perfectly.
        # Shape: [num_atoms, num_atoms]
        dummy_edges = torch.ones((self.num_atoms, self.num_atoms), dtype=torch.long)
        
        # Set the diagonal (self-loops) to 0, just to be mathematically safe
        dummy_edges.fill_diagonal_(0)

        return time_series, dummy_edges

def create_dataloaders(args, train_path, valid_path, test_path):
    # 1. Instantiate the Datasets
    train_dataset = PickledTimeSeriesDataset(train_path, args.num_atoms)
    valid_dataset = PickledTimeSeriesDataset(valid_path, args.num_atoms)
    test_dataset  = PickledTimeSeriesDataset(test_path, args.num_atoms)

    # 2. Wrap them in PyTorch DataLoaders
    # Note: num_workers=0 ensures we don't hit the multiprocessing crash on HPC clusters
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Always shuffle training data
        num_workers=0,
        drop_last=True # Drops the last incomplete batch to prevent shape mismatch errors
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False
    )

    return train_loader, valid_loader, test_loader

def pad_dataloader_for_25_nodes(dataloader):
    """Yields batches with a dummy 25th particle appended."""
    for data, target in dataloader:
        batch_size, _, timesteps, dims = data.shape
        # Create a dummy variable (all zeros) for the 25th node
        dummy_node = torch.zeros((batch_size, 1, timesteps, dims), dtype=data.dtype)
        if data.is_cuda:
            dummy_node = dummy_node.cuda()
        # Concatenate to make the shape [batch, 25, timesteps, dims]
        padded_data = torch.cat([data, dummy_node], dim=1)
        yield padded_data, target

def save_discretized_graphs(encoder, dataloader, args, rel_rec, rel_send, num_nodes, save_path, mask_idx=None):
    """Extracts, discretizes, reshapes, and saves the predicted graphs."""
    encoder.eval()
    all_graphs = []
    
    # Create the off-diagonal indices to rebuild the NxN matrix
    off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rows, cols = np.where(off_diag)

    print(f"Starting inference for {save_path}...")
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(args.device)

            if mask_idx is not None:
                # NO manual slicing! Just pass the raw data and let the encoder handle it.
                logits, _, _ = encoder(data, rel_rec, rel_send, mask_idx=mask_idx)
            else:
                logits = encoder(data, rel_rec, rel_send)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()

            batch_size = preds.shape[0]
            adj_matrix = np.zeros((batch_size, num_nodes, num_nodes), dtype=np.int8)
            adj_matrix[:, rows, cols] = preds 

            all_graphs.append(adj_matrix)

    # Concatenate and save
    final_graphs = np.concatenate(all_graphs, axis=0)
    np.save(save_path, final_graphs)
    print(f"Saved successfully! Shape: {final_graphs.shape}\n")
    
    return final_graphs

def plot_graph_heatmap(graph_matrix, title, save_name):
    """Generates a heatmap visualization of a single adjacency matrix."""
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to create a clean heatmap
    sns.heatmap(
        graph_matrix, 
        cmap="Blues", 
        cbar=False, 
        linewidths=0.5, 
        linecolor='lightgray',
        square=True
    )
    
    plt.title(title, pad=20, fontsize=16)
    plt.xlabel("Receiver Node", labelpad=15, fontsize=14)
    plt.ylabel("Sender Node", labelpad=15, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved heatmap visualization to {save_name}\n")

if __name__ == "__main__":
    import os
    import argparse
    
    # ==========================================
    # 1. SETUP ARGUMENTS & PATHS
    # ==========================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='SWANSF-diff')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cuda', action='store_true', default=True)
    # Add any other required args your model_loader expects (like hidden units)
    parser.add_argument('--encoder_hidden', type=int, default=256)
    parser.add_argument('--edge_types', type=int, default=2)
    args, _ = parser.parse_known_args() # Use known_args to avoid crashing on missing flags

    args.global_temp = False
    args.encoder = "cnn"          # Or "cnn", depending on what architecture you trained
    args.dims = 1                 # Features per timestep per particle
    args.timesteps = 59           # You mentioned 59 timesteps earlier!
    args.encoder_dropout = 0.0
    args.factor = True            # Typically True in this architecture
    args.num_GPU = 1 if args.cuda else 0
    
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Point directly to your 73,492-sample test file
    test_path = os.path.join(args.datadir, "test", "diff1.pkl")

    # ==========================================
    # 2. CREATE TEST DATALOADER
    # ==========================================
    print("Initializing Test DataLoader...")
    test_dataset = PickledTimeSeriesDataset(test_path, num_atoms=24)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False
    )

    # Dummy normalization bounds (Encoder doesn't strictly need these unless it normalizes inputs)
    loc_max, loc_min, vel_max, vel_min = 1.0, -1.0, 1.0, -1.0

    # ==========================================
    # 3. LOAD MODEL 1: Standard Model
    # ==========================================
    print("\nLoading Standard Model...")
    args.num_atoms = 24
    args.unobserved = 0
    args.model_unobserved = 0
    args.use_encoder = True # Ensure this is true so it knows to build it
    args.load_folder = "cnn-rnn-EVAL-1" # Set standard checkpoint path
    
    # Load ONLY the encoder!
    encoder_standard = load_encoder(args)
    encoder_standard.eval() # Freeze dropout/batchnorm for inference

    rel_rec_24, rel_send_24 = create_rel_rec_send(args, args.num_atoms)
    
    graphs_standard = save_discretized_graphs(
        encoder=encoder_standard,
        dataloader=test_loader,
        args=args,
        rel_rec=rel_rec_24,
        rel_send=rel_send_24,
        num_nodes=24,
        save_path="test_graphs_cnn_rnn_24x24.npy",
        mask_idx=None
    )
    
    plot_graph_heatmap(
        graph_matrix=graphs_standard[0], 
        title="Predicted Causal Graph (Standard Model)", 
        save_name="heatmap_standard.png"
    )

# ==========================================
    # 4. LOAD MODEL 2: Hidden Variable Model
    # ==========================================
    print("\nLoading Hidden Variable Model...")
    args.num_atoms = 24  
    args.unobserved = 1
    args.model_unobserved = 0
    args.use_encoder = True
    
    # --- THIS WAS THE MISSING LINE! ---
    args.load_folder = "mlph-rnn" 
    # ----------------------------------
    
    encoder_hidden = load_encoder(args)
    encoder_hidden.eval()

    rel_rec_24, rel_send_24 = create_rel_rec_send(args, args.num_atoms)
    
    # Pass the standard 'test_loader' here
    graphs_hidden = save_discretized_graphs(
        encoder=encoder_hidden,
        dataloader=test_loader, 
        args=args,
        rel_rec=rel_rec_24,
        rel_send=rel_send_24,
        num_nodes=24,
        save_path="test_graphs_hidden_rnn_24x24.npy",
        mask_idx=23 # Tells the encoder to internally slice out particle 23
    )
    
    plot_graph_heatmap(
        graph_matrix=graphs_hidden[0], 
        title="Predicted Causal Graph (Hidden Variable Model)", 
        save_name="heatmap_hidden.png"
    )