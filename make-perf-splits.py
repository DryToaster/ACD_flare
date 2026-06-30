import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from torch.utils.data import Dataset, DataLoader

# Assuming you have your utils imported from the codebase
from model.utils import create_rel_rec_send
from model.model_loader import load_encoder

class PickledTimeSeriesDataset(Dataset):
    def __init__(self, pickle_path, num_atoms):
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            raw_data = pickle.load(f)
            
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
            
        self.data = torch.tensor(raw_data, dtype=torch.float32)

        if len(self.data.shape) == 4:
            self.data = self.data.permute(0, 2, 1, 3)
        elif len(self.data.shape) == 3:
            self.data = self.data.permute(0, 2, 1).unsqueeze(-1)

        self.num_samples = self.data.shape[0]
        self.num_atoms = num_atoms
        print(f"Successfully loaded dataset! Shape: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        time_series = self.data[idx]
        dummy_edges = torch.ones((self.num_atoms, self.num_atoms), dtype=torch.long)
        dummy_edges.fill_diagonal_(0)
        return time_series, dummy_edges


def process_and_save_graphs(encoder, dataloader, args, rel_rec, rel_send, num_nodes, discrete_save_path, continuous_save_path, mask_idx=None):
    """Extracts both discrete (argmax) and continuous (raw activations) graphs and saves them."""
    encoder.eval()
    all_discrete = []
    all_continuous = []
    
    # Create the off-diagonal indices to rebuild the NxN matrix
    off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rows, cols = np.where(off_diag)

    print(f"Starting inference...")
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(args.device)

            if mask_idx is not None:
                logits, _, _ = encoder(data, rel_rec, rel_send, mask_idx=mask_idx)
            else:
                logits = encoder(data, rel_rec, rel_send)

            # Get raw activations (probabilities)
            probs = F.softmax(logits, dim=-1)
            
            # 1. Discrete: Argmax to get 0 or 1
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            
            # 2. Continuous: The raw activation/probability of the edge existing (class index 1)
            # Note: If you specifically want pre-softmax logits instead of 0-1 probabilities, 
            # you can change this to: acts = logits[:, :, 1].cpu().numpy()
            acts = probs[:, :, 1].cpu().numpy()

            batch_size = preds.shape[0]
            
            # Rebuild discrete NxN matrix
            adj_discrete = np.zeros((batch_size, num_nodes, num_nodes), dtype=np.int8)
            adj_discrete[:, rows, cols] = preds 
            all_discrete.append(adj_discrete)
            
            # Rebuild continuous NxN matrix
            adj_continuous = np.zeros((batch_size, num_nodes, num_nodes), dtype=np.float32)
            adj_continuous[:, rows, cols] = acts
            all_continuous.append(adj_continuous)

    # Concatenate batches
    final_discrete = np.concatenate(all_discrete, axis=0)
    final_continuous = np.concatenate(all_continuous, axis=0)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(discrete_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(continuous_save_path), exist_ok=True)

    # Save to disk
    np.save(discrete_save_path, final_discrete)
    np.save(continuous_save_path, final_continuous)
    
    print(f"Saved Discrete -> {discrete_save_path} (Shape: {final_discrete.shape})")
    print(f"Saved Continuous -> {continuous_save_path} (Shape: {final_continuous.shape})\n")
    
    return final_discrete, final_continuous


def plot_graph_heatmap(graph_matrix, title, save_name):
    """Generates a heatmap visualization of a single adjacency matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(graph_matrix, cmap="Blues", cbar=True, linewidths=0.5, linecolor='lightgray', square=True)
    plt.title(title, pad=20, fontsize=16)
    plt.xlabel("Receiver Node", labelpad=15, fontsize=14)
    plt.ylabel("Sender Node", labelpad=15, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()


if __name__ == "__main__":
    
    # ==========================================
    # 1. RUN CONFIGURATION
    # ==========================================
    # Set the encoder directory ("cnn-rnn", "mlph-rnn", etc.)
    ENCODER_FOLDER = "cnn-rnn-EVAL-3" 
    
    # Set to 23 if using a Hidden Variable model, otherwise set to None
    MASK_IDX = None 
    NUM_ATOMS = 24
    
    # Define the list of files to process and their respective output locations
    FILES_TO_PROCESS = [
        {
            "input_path": "SWANSF-diff/test/diff5.pkl",
            "out_discrete": "output_graphs/run-3/test_discrete.npy",
            "out_continuous": "output_graphs/run-3/test_continuous.npy",
            "generate_heatmap": False
        },
        {
            "input_path": "SWANSF-diff/train/diff4.pkl",
            "out_discrete": "output_graphs/run-3/train_discrete.npy",
            "out_continuous": "output_graphs/run-3/train_continuous.npy",
            "generate_heatmap": False
        }
        # Add as many files as you want here!
        # {
        #     "input_path": "SWANSF-diff/train/diff1.pkl",
        #     "out_discrete": "output_graphs/discrete/train_diff1_discrete.npy",
        #     "out_continuous": "output_graphs/continuous/train_diff1_continuous.npy",
        #     "generate_heatmap": False
        # }
    ]

    # ==========================================
    # 2. SETUP ARGUMENTS & LOAD ENCODER
    # ==========================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--encoder_hidden', type=int, default=256)
    parser.add_argument('--edge_types', type=int, default=2)
    args, _ = parser.parse_known_args()

    args.global_temp = False
    args.encoder = "cnn"
    args.dims = 1
    args.timesteps = 59
    args.encoder_dropout = 0.0
    args.factor = True
    args.num_GPU = 1 if args.cuda else 0
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Model Loading Args
    args.num_atoms = NUM_ATOMS
    args.unobserved = 1 if MASK_IDX is not None else 0
    args.model_unobserved = 0
    args.use_encoder = True
    args.load_folder = ENCODER_FOLDER

    print(f"Loading Encoder from: {args.load_folder}...")
    encoder = load_encoder(args)
    encoder.eval()
    
    rel_rec, rel_send = create_rel_rec_send(args, args.num_atoms)

    # ==========================================
    # 3. PROCESS ALL FILES
    # ==========================================
    for file_config in FILES_TO_PROCESS:
        input_file = file_config["input_path"]
        
        print(f"\n{'='*50}")
        print(f"PROCESSING: {input_file}")
        print(f"{'='*50}")
        
        # Initialize DataLoader for the current file
        dataset = PickledTimeSeriesDataset(input_file, num_atoms=NUM_ATOMS)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Process and extract graphs
        discrete_graphs, continuous_graphs = process_and_save_graphs(
            encoder=encoder,
            dataloader=dataloader,
            args=args,
            rel_rec=rel_rec,
            rel_send=rel_send,
            num_nodes=NUM_ATOMS,
            discrete_save_path=file_config["out_discrete"],
            continuous_save_path=file_config["out_continuous"],
            mask_idx=MASK_IDX
        )
        
        # Optional: Generate quick heatmaps to verify the output visually
        if file_config.get("generate_heatmap", False):
            base_name = os.path.basename(file_config["input_path"]).split('.')[0]
            heatmap_dir = os.path.dirname(file_config["out_discrete"])
            
            plot_graph_heatmap(
                graph_matrix=discrete_graphs[0], 
                title=f"Discrete Causal Graph ({base_name})", 
                save_name=os.path.join(heatmap_dir, f"heatmap_discrete_{base_name}.png")
            )
            plot_graph_heatmap(
                graph_matrix=continuous_graphs[0], 
                title=f"Continuous Causal Graph ({base_name})", 
                save_name=os.path.join(heatmap_dir, f"heatmap_continuous_{base_name}.png")
            )