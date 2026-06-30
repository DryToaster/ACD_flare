import os
import numpy as np
import pandas as pd
from scipy import stats
import pickle

# SWANSF Variable Names
VAR_NAMES = [
    'R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX', 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 
    'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 
    'MEANALP', 'TOTFX'
]

def load_data(graphs_path, labels_path):
    graphs = np.load(graphs_path)
    if labels_path.endswith('.pkl'):
        with open(labels_path, 'rb') as f:
            labels = np.array(pickle.load(f))
    else:
        labels = np.load(labels_path)
        
    return graphs[labels == 1], graphs[labels == 0], graphs.shape[1]

def get_labels(num_nodes):
    labels = VAR_NAMES.copy()
    if num_nodes > len(labels):
        labels.append('HIDDEN_VAR')
    return labels

def calc_rank_biserial(u_stat, n1, n2):
    """Calculates the Rank-Biserial Correlation effect size from the MWU U-statistic."""
    # Using absolute value to represent magnitude of difference
    return abs(1 - (2 * u_stat) / (n1 * n2))

def run_graph_level_tests(pos_graphs, neg_graphs, num_nodes):
    print("\n" + "="*50)
    print(" GRAPH-LEVEL METRICS (Mann-Whitney U Test)")
    print("="*50)
    
    n_pos, n_neg = pos_graphs.shape[0], neg_graphs.shape[0]
    max_edges = num_nodes * (num_nodes - 1)
    
    # 1. Density
    pos_density = pos_graphs.sum(axis=(1, 2)) / max_edges
    neg_density = neg_graphs.sum(axis=(1, 2)) / max_edges
    
    u_dens, p_dens = stats.mannwhitneyu(pos_density, neg_density, alternative='two-sided')
    r_dens = calc_rank_biserial(u_dens, n_pos, n_neg)
    
    # 2. Reciprocity
    pos_recip = (pos_graphs * pos_graphs.transpose(0, 2, 1)).sum(axis=(1, 2)) / (pos_graphs.sum(axis=(1, 2)) + 1e-9)
    neg_recip = (neg_graphs * neg_graphs.transpose(0, 2, 1)).sum(axis=(1, 2)) / (neg_graphs.sum(axis=(1, 2)) + 1e-9)
    
    u_recip, p_recip = stats.mannwhitneyu(pos_recip, neg_recip, alternative='two-sided')
    r_recip = calc_rank_biserial(u_recip, n_pos, n_neg)
    
    results = pd.DataFrame({
        'Metric': ['Density', 'Reciprocity'],
        'P-Value': [p_dens, p_recip],
        'Effect Size (r)': [r_dens, r_recip]
    })
    
    # Format p-values for readability
    results['P-Value'] = results['P-Value'].apply(lambda x: f"{x:.2e}" if x > 0 else "< 1e-300")
    print(results.to_string(index=False))

def run_variable_level_tests(pos_graphs, neg_graphs, num_nodes):
    print("\n" + "="*60)
    print(" PER-VARIABLE METRICS (Kolmogorov-Smirnov Test)")
    print("="*60)
    
    labels = get_labels(num_nodes)
    
    # Calculate Out-Degree
    pos_out = pos_graphs.sum(axis=2)
    neg_out = neg_graphs.sum(axis=2)
    
    # Calculate Net Flow (Out - In)
    pos_net = pos_graphs.sum(axis=2) - pos_graphs.sum(axis=1)
    neg_net = neg_graphs.sum(axis=2) - neg_graphs.sum(axis=1)
    
    results_list = []
    
    for i in range(num_nodes):
        var_name = labels[i]
        
        # K-S Test for Out-Degree
        ks_out_stat, p_out = stats.ks_2samp(pos_out[:, i], neg_out[:, i])
        
        # K-S Test for Net Flow
        ks_net_stat, p_net = stats.ks_2samp(pos_net[:, i], neg_net[:, i])
        
        results_list.append({
            'Variable': var_name,
            'Out-Degree P-Val': p_out,
            'Out-Degree Effect (D)': ks_out_stat,
            'Net Flow P-Val': p_net,
            'Net Flow Effect (D)': ks_net_stat
        })
        
    df = pd.DataFrame(results_list)
    
    # Format for readability
    df['Out-Degree P-Val'] = df['Out-Degree P-Val'].apply(lambda x: f"{x:.2e}" if x > 0 else "< 1e-300")
    df['Net Flow P-Val'] = df['Net Flow P-Val'].apply(lambda x: f"{x:.2e}" if x > 0 else "< 1e-300")
    
    # Sort by Net Flow Effect Size to see the most predictive variables at the top
    df = df.sort_values(by='Net Flow Effect (D)', ascending=False).reset_index(drop=True)
    
    print(df.to_string())
    
    # Save to CSV for reporting
    df.to_csv("variable_ks_tests.csv", index=False)
    print("\nSaved detailed variable results to 'variable_ks_tests.csv'")

if __name__ == "__main__":
    # Ensure these paths point to your actual arrays
    GRAPHS_FILE = "test_graphs_cnn_rnn_24x24.npy" 
    LABELS_FILE = "test/Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl" 
    
    pos_graphs, neg_graphs, num_nodes = load_data(GRAPHS_FILE, LABELS_FILE)
    
    run_graph_level_tests(pos_graphs, neg_graphs, num_nodes)
    run_variable_level_tests(pos_graphs, neg_graphs, num_nodes)