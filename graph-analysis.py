import os
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# SWANSF Variable Names
VAR_NAMES = [
    'R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 
    'MEANALP', 'TOTFX'
]

def load_and_split_data(graphs_path, labels_path):
    print(f"Loading graphs from {graphs_path}...")
    graphs = np.load(graphs_path)
    
    print(f"Loading labels from {labels_path}...")
    if labels_path.endswith('.pkl'):
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
    else:
        labels = np.load(labels_path)
    
    assert graphs.shape[0] == labels.shape[0], "Mismatch between number of graphs and labels!"
    
    pos_graphs = graphs[labels == 1]
    neg_graphs = graphs[labels == 0]
    
    print(f"Total Graphs: {graphs.shape[0]}")
    print(f"Positive Set (Class 1): {pos_graphs.shape[0]}")
    print(f"Negative Set (Class 0): {neg_graphs.shape[0]}\n")
    
    return pos_graphs, neg_graphs, graphs.shape[1]

def get_labels(num_nodes):
    """Safely handles the 24-node standard or 25-node hidden models."""
    labels = VAR_NAMES.copy()
    if num_nodes > len(labels):
        labels.append('HIDDEN_VAR')
    return labels

def plot_averaged_heatmaps(pos_graphs, neg_graphs, num_nodes, save_dir):
    print("Generating averaged heatmaps...")
    pos_avg = pos_graphs.mean(axis=0)
    neg_avg = neg_graphs.mean(axis=0)
    labels = get_labels(num_nodes)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(pos_avg, ax=axes[0], cmap="Reds", square=True, xticklabels=labels, yticklabels=labels)
    axes[0].set_title("Average Causal Graph (Positive Set)", pad=15, fontsize=16)
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)
    
    sns.heatmap(neg_avg, ax=axes[1], cmap="Blues", square=True, xticklabels=labels, yticklabels=labels)
    axes[1].set_title("Average Causal Graph (Negative Set)", pad=15, fontsize=16)
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "averaged_heatmaps.png"), dpi=300)
    plt.close()

def plot_degree_distributions(pos_graphs, neg_graphs, num_nodes, scale_factor, save_dir):
    print(f"Generating per-variable degree distributions (scaled by {scale_factor})...")
    
    pos_in_degrees = pos_graphs.sum(axis=1)
    neg_in_degrees = neg_graphs.sum(axis=1)
    pos_out_degrees = pos_graphs.sum(axis=2)
    neg_out_degrees = neg_graphs.sum(axis=2)
    
    bins = np.arange(num_nodes + 2) - 0.5 
    cols = 5
    rows = math.ceil(num_nodes / cols)
    labels = get_labels(num_nodes)
    
    # --- Helper function to plot grids ---
    def plot_grid(pos_data, neg_data, title, filename):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), sharex=True)
        fig.suptitle(title, fontsize=24, y=0.98)
        axes_flat = axes.flatten()
        
        for i in range(num_nodes):
            ax = axes_flat[i]
            pos_weights = np.ones_like(pos_data[:, i]) / scale_factor
            
            ax.hist(neg_data[:, i], bins=bins, color='royalblue', alpha=0.6, label='Negative Set')
            ax.hist(pos_data[:, i], bins=bins, weights=pos_weights, color='firebrick', alpha=0.6, label='Positive Set (Scaled)')
            
            ax.set_title(labels[i], fontsize=12, fontweight='bold')
            ax.set_xlim(-0.5, num_nodes + 0.5)
            
            if i == 0:
                ax.legend()
                
        for j in range(num_nodes, len(axes_flat)):
            axes_flat[j].set_visible(False)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

    plot_grid(pos_out_degrees, neg_out_degrees, "Out-Degree Distribution per Variable", "out_degree_grid.png")
    plot_grid(pos_in_degrees, neg_in_degrees, "In-Degree Distribution per Variable", "in_degree_grid.png")

def plot_net_flow(pos_graphs, neg_graphs, num_nodes, scale_factor, save_dir):
    print("Generating Net Causal Flow metric...")
    labels = get_labels(num_nodes)
    
    # Net Flow = Out - In. Summed across the dataset, scaled for the positive class.
    pos_net_flow = (pos_graphs.sum(axis=(0, 2)) - pos_graphs.sum(axis=(0, 1))) / scale_factor
    neg_net_flow = neg_graphs.sum(axis=(0, 2)) - neg_graphs.sum(axis=(0, 1))
    
    x = np.arange(num_nodes)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width/2, neg_net_flow, width, label='Negative Set', color='royalblue')
    ax.bar(x + width/2, pos_net_flow, width, label='Positive Set (Scaled)', color='firebrick')
    
    ax.set_title("Net Causal Flow (Total Out-Degree minus Total In-Degree)", fontsize=16)
    ax.set_ylabel("Net Flow Volume (Positive = Driver, Negative = Receiver)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontweight='bold')
    
    # Add a zero line for visual clarity
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "net_causal_flow.png"), dpi=300)
    plt.close()

def plot_graph_statistics(pos_graphs, neg_graphs, num_nodes, save_dir):
    print("Generating whole-graph density and reciprocity statistics...")
    max_possible_edges = num_nodes * (num_nodes - 1)
    
    # 1. Density
    pos_density = pos_graphs.sum(axis=(1, 2)) / max_possible_edges
    neg_density = neg_graphs.sum(axis=(1, 2)) / max_possible_edges
    
    # 2. Reciprocity (Fraction of edges that are bidirectional)
    # A * A^T gives a 1 only if both (i,j) and (j,i) exist.
    pos_recip_edges = (pos_graphs * pos_graphs.transpose(0, 2, 1)).sum(axis=(1, 2))
    neg_recip_edges = (neg_graphs * neg_graphs.transpose(0, 2, 1)).sum(axis=(1, 2))
    
    # Add 1e-9 to prevent division by zero for completely empty graphs
    pos_reciprocity = pos_recip_edges / (pos_graphs.sum(axis=(1, 2)) + 1e-9)
    neg_reciprocity = neg_recip_edges / (neg_graphs.sum(axis=(1, 2)) + 1e-9)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Density
    sns.histplot(neg_density, bins=30, ax=axes[0], color='royalblue', label='Negative Set', stat='density', alpha=0.5, kde=True)
    sns.histplot(pos_density, bins=30, ax=axes[0], color='firebrick', label='Positive Set', stat='density', alpha=0.5, kde=True)
    axes[0].set_title("Graph Density Distribution")
    axes[0].set_xlabel("Density (Fraction of possible edges)")
    axes[0].legend()
    
    # Plot Reciprocity
    sns.histplot(neg_reciprocity, bins=30, ax=axes[1], color='royalblue', label='Negative Set', stat='density', alpha=0.5, kde=True)
    sns.histplot(pos_reciprocity, bins=30, ax=axes[1], color='firebrick', label='Positive Set', stat='density', alpha=0.5, kde=True)
    axes[1].set_title("Network Reciprocity (Feedback Loops)")
    axes[1].set_xlabel("Reciprocity (Fraction of edges that are bidirectional)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "whole_graph_statistics.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION
    # ==========================================
    GRAPHS_FILE = "test_graphs_cnn_rnn_24x24.npy" 
    LABELS_FILE = "test/Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl"  
    OUTPUT_DIR = "graph_analysis_results_cnn_rnn"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pos_graphs, neg_graphs, num_nodes = load_and_split_data(GRAPHS_FILE, LABELS_FILE)
    
    # Run visualizations
    plot_averaged_heatmaps(pos_graphs, neg_graphs, num_nodes=num_nodes, save_dir=OUTPUT_DIR)
    
    plot_degree_distributions(pos_graphs, neg_graphs, num_nodes=num_nodes, scale_factor=0.0171, save_dir=OUTPUT_DIR)
    
    plot_net_flow(pos_graphs, neg_graphs, num_nodes=num_nodes, scale_factor=0.0171, save_dir=OUTPUT_DIR)
    
    plot_graph_statistics(pos_graphs, neg_graphs, num_nodes=num_nodes, save_dir=OUTPUT_DIR)
    
    print(f"Analysis complete! All plots saved to ./{OUTPUT_DIR}/")