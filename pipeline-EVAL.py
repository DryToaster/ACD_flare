import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data_file(filepath):
    """Smart loader that handles both .npy and .pkl files."""
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data
    else:
        return np.load(filepath)

# ==========================================
# 1. GC-xLSTM CLASSIFIER (Standalone)
# ==========================================
class GC_xLSTM_Classifier(nn.Module):
    def __init__(self, num_vars=24, hidden_size=32, num_classes=2):
        super().__init__()
        self.num_vars = num_vars
        self.W = nn.Parameter(torch.Tensor(num_vars, num_vars, hidden_size))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.lstms = nn.ModuleList([nn.LSTM(hidden_size, hidden_size, batch_first=True) for _ in range(num_vars)])
        self.fc = nn.Linear(num_vars * hidden_size, num_classes)
        
    def forward(self, x):
        B, T, V = x.shape
        var_outputs = []
        for v in range(self.num_vars):
            x_emb = torch.matmul(x, self.W[v]) 
            out, _ = self.lstms[v](x_emb)
            var_outputs.append(out[:, -1, :]) 
        fused = torch.cat(var_outputs, dim=-1)
        return self.fc(fused)

# ==========================================
# 2. CAIFORMER (Handles ACD & Standalone)
# ==========================================
class CAIformer(nn.Module):
    def __init__(self, num_vars=24, d_model=64, nhead=4, num_classes=2):
        super().__init__()
        self.num_vars = num_vars
        self.internal_graph = nn.Parameter(torch.randn(num_vars, num_vars))
        self.var_embedding = nn.Linear(1, d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model * num_vars, num_classes)

    def forward(self, x_mvts, x_graph=None):
        B, T, V = x_mvts.shape
        if x_graph is not None:
            adj_matrix = x_graph
        else:
            adj_matrix = torch.sigmoid(self.internal_graph).unsqueeze(0).expand(B, -1, -1)
            
        x_spatial = torch.bmm(adj_matrix, x_mvts.transpose(1, 2)).transpose(1, 2)
        x_spatial = x_spatial.unsqueeze(-1)
        out_features = []
        
        for v in range(V):
            var_seq = self.var_embedding(x_spatial[:, :, v, :]) 
            transformed = self.temporal_transformer(var_seq)    
            out_features.append(transformed.mean(dim=1))        
            
        fused = torch.cat(out_features, dim=1)
        return self.fc(fused)

# ==========================================
# 3. BASELINE MODELS
# ==========================================
class BaseRNN(nn.Module):
    def __init__(self, model_type, input_size=24, hidden_size=64, num_classes=2):
        super().__init__()
        self.rnn = getattr(nn, model_type)(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=24, num_classes=2, nhead=4, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        out = self.transformer(x)
        return self.fc(out.mean(dim=1))

class TabularMLP(nn.Module):
    def __init__(self, input_size=576, num_classes=2): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================
def train_classifier(model, X_mvts_tr, y_train, X_mvts_te, y_test, device, X_tab_tr=None, X_tab_te=None, epochs=30, batch_size=32):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if X_tab_tr is not None:
        train_data = TensorDataset(torch.FloatTensor(X_mvts_tr), torch.FloatTensor(X_tab_tr), torch.LongTensor(y_train))
    else:
        train_data = TensorDataset(torch.FloatTensor(X_mvts_tr), torch.LongTensor(y_train))
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 3:
                outputs = model(batch[0].to(device), batch[1].to(device))
                loss = criterion(outputs, batch[2].to(device))
            else:
                outputs = model(batch[0].to(device))
                loss = criterion(outputs, batch[1].to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds = []
    all_probs = []
    
    # Wrap test data in a DataLoader so we don't blow up the GPU RAM
    if X_tab_te is not None:
        test_data = TensorDataset(torch.FloatTensor(X_mvts_te), torch.FloatTensor(X_tab_te))
    else:
        test_data = TensorDataset(torch.FloatTensor(X_mvts_te))
        
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2: # MVTS + Graph
                logits = model(batch[0].to(device), batch[1].to(device))
            else:               # MVTS only
                logits = model(batch[0].to(device))
                
            # Get raw probabilities for the positive class (solar flare)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            # Get the hard 0/1 predictions
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            
    torch.cuda.empty_cache()
    
    # Return as flat numpy arrays
    return np.array(all_preds), np.array(all_probs)

def evaluate_binary_models(run_configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models_config = {
        "MVTS_RNN": lambda: BaseRNN('RNN', input_size=24),
        "MVTS_LSTM": lambda: BaseRNN('LSTM', input_size=24),
        "MVTS_Transformer": lambda: TimeSeriesTransformer(input_size=24),
        "MVTS_MiniROCKET": "minirocket",
        "Tabular_RF_Discrete": lambda: RandomForestClassifier(n_estimators=100),
        "Tabular_RF_Continuous": lambda: RandomForestClassifier(n_estimators=100),
        "Tabular_MLP_Discrete": lambda: TabularMLP(input_size=576),
        "Tabular_MLP_Continuous": lambda: TabularMLP(input_size=576),
        "Transformer_with_ACD_Discrete": "caiformer_acd_disc",
        "Transformer_with_ACD_Continuous": "caiformer_acd_cont",
        "CAIformer_Standalone": "caiformer_standalone",
        "GC_xLSTM_Classifier": "gc_xlstm"
    }
    
    results = {name: {'acc': [], 'f1': [], 'prec': [], 'rec': [], 'tss': [], 'hss2': []} for name in models_config.keys()}
    
    for config in run_configs:
        run_name = config.get("run_name", "Unnamed Run")
        print(f"\n{'='*55}\n--- Running {run_name.upper()} ---\n{'='*55}")
        
        # Load MVTS and Labels (with flatten to prevent dimension errors)
        X_mvts_tr = load_data_file(config["X_mvts_train"])
        X_mvts_te = load_data_file(config["X_mvts_test"])
        y_tr = load_data_file(config["y_train"]).flatten()
        y_te = load_data_file(config["y_test"]).flatten()
        
        # Load Discrete and Continuous Graphs
        X_acd_disc_tr = load_data_file(config["X_graph_discrete_train"])
        X_acd_disc_te = load_data_file(config["X_graph_discrete_test"])
        X_acd_cont_tr = load_data_file(config["X_graph_continuous_train"])
        X_acd_cont_te = load_data_file(config["X_graph_continuous_test"])
        
        # Flattened variants for Tabular models
        X_acd_disc_tr_flat = X_acd_disc_tr.reshape(X_acd_disc_tr.shape[0], -1)
        X_acd_disc_te_flat = X_acd_disc_te.reshape(X_acd_disc_te.shape[0], -1)
        X_acd_cont_tr_flat = X_acd_cont_tr.reshape(X_acd_cont_tr.shape[0], -1)
        X_acd_cont_te_flat = X_acd_cont_te.reshape(X_acd_cont_te.shape[0], -1)
        
        # Initialize a plot for this specific split
        plt.figure(figsize=(10, 8))
        
        for name, model_init in models_config.items():
            preds = None
            probs = None # Track probabilities for the ROC curve
            
            if name == "Transformer_with_ACD_Discrete":
                model = CAIformer(num_vars=24)
                preds, probs = train_classifier(model, X_mvts_tr, y_tr, X_mvts_te, y_te, device, X_tab_tr=X_acd_disc_tr, X_tab_te=X_acd_disc_te)
                
            elif name == "Transformer_with_ACD_Continuous":
                model = CAIformer(num_vars=24)
                preds, probs = train_classifier(model, X_mvts_tr, y_tr, X_mvts_te, y_te, device, X_tab_tr=X_acd_cont_tr, X_tab_te=X_acd_cont_te)
                
            elif name == "CAIformer_Standalone":
                model = CAIformer(num_vars=24)
                preds, probs = train_classifier(model, X_mvts_tr, y_tr, X_mvts_te, y_te, device, X_tab_tr=None, X_tab_te=None)
                
            elif name == "GC_xLSTM_Classifier":
                model = GC_xLSTM_Classifier(num_vars=24)
                preds, probs = train_classifier(model, X_mvts_tr, y_tr, X_mvts_te, y_te, device, X_tab_tr=None, X_tab_te=None)
                
            elif name.startswith("MVTS_"):
                if name == "MVTS_MiniROCKET":
                    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                    
                    # Fix MiniROCKET transform so it fits on train and transforms on test
                    minirocket = MiniRocketMultivariate()
                    X_tr_tf = minirocket.fit_transform(X_mvts_tr.transpose(0, 2, 1))
                    X_te_tf = minirocket.transform(X_mvts_te.transpose(0, 2, 1))
                    
                    clf.fit(X_tr_tf, y_tr)
                    preds = clf.predict(X_te_tf)
                    # Ridge doesn't output true probabilities, but decision_function gives bounded scores perfect for ROC
                    probs = clf.decision_function(X_te_tf) 
                else:
                    preds, probs = train_classifier(model_init(), X_mvts_tr, y_tr, X_mvts_te, y_te, device)
            
            elif name.startswith("Tabular_"):
                cur_tr_flat = X_acd_cont_tr_flat if "Continuous" in name else X_acd_disc_tr_flat
                cur_te_flat = X_acd_cont_te_flat if "Continuous" in name else X_acd_disc_te_flat
                
                if "MLP" in name:
                    preds, probs = train_classifier(model_init(), cur_tr_flat, y_tr, cur_te_flat, y_te, device)
                else:
                    rf_model = model_init()
                    rf_model.fit(cur_tr_flat, y_tr)
                    preds = rf_model.predict(cur_te_flat)
                    probs = rf_model.predict_proba(cur_te_flat)[:, 1] # Grab the probability of class 1
            
            # --- METRICS ---
            acc = accuracy_score(y_te, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average='binary')
            tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
            
            tss = (tp / (tp + fn) if (tp + fn) > 0 else 0.0) + (tn / (tn + fp) if (tn + fp) > 0 else 0.0) - 1.0
            den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
            hss2 = (2 * (tp * tn - fp * fn)) / den if den > 0 else 0.0
            
            # --- AUC & ROC PLOTTING ---
            fpr, tpr, _ = roc_curve(y_te, probs)
            roc_auc = auc(fpr, tpr)
            
            # Make sure 'auc' is tracked in your initial `results` dictionary!
            if 'auc' not in results[name]: results[name]['auc'] = []
            
            results[name]['acc'].append(acc); results[name]['f1'].append(f1); results[name]['prec'].append(prec)
            results[name]['rec'].append(rec); results[name]['tss'].append(tss); results[name]['hss2'].append(hss2)
            results[name]['auc'].append(roc_auc)
            
            # Add this model's curve to the plot
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
            print(f"  {name} -> F1: {f1:.4f} | TSS: {tss:.4f} | HSS2: {hss2:.4f} | AUC: {roc_auc:.4f}")

        # Finish rendering the plot for this split
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve - {run_name}', fontsize=16)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        safe_name = run_name.replace(" ", "_").lower()
        plt.savefig(f"roc_curve_{safe_name}.png", dpi=300)
        plt.close()
        print(f"\n  --> Saved ROC Graph to roc_curve_{safe_name}.png")

    print("\n" + "="*55 + f"\nFINAL COMPARISON: {len(run_configs)}-SPLIT AVERAGE\n" + "="*55)
    for name in results:
        print(f"\n{name.upper()}:")
        for metric in ['acc', 'prec', 'rec', 'f1', 'tss', 'hss2', 'auc']:
            print(f"  {metric.upper()}: {np.mean(results[name][metric]):.4f} +/- {np.std(results[name][metric]):.4f}")

if __name__ == "__main__":
    set_seed()
    
    # Define individual file paths for each run here
    run_configurations = [
        # {
        #     "run_name": "Split 1",
        #     "X_mvts_train": "SWANSF-diff/train/diff2.pkl",
        #     "X_mvts_test":  "SWANSF-diff/test/diff3.pkl",
        #     "X_graph_discrete_train": "output_graphs/run-1/train_discrete.npy",
        #     "X_graph_discrete_test":  "output_graphs/run-1/test_discrete.npy",
        #     "X_graph_continuous_train": "output_graphs/run-1/train_continuous.npy",
        #     "X_graph_continuous_test":  "output_graphs/run-1/test_continuous.npy",
        #     "y_train":      "train/Partition2_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl",
        #     "y_test":       "test/Partition3_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
        # },
        # {
        #     "run_name": "Split 2",
        #     "X_mvts_train": "SWANSF-diff/train/diff3.pkl",
        #     "X_mvts_test":  "SWANSF-diff/test/diff4.pkl",
        #     "X_graph_discrete_train": "output_graphs/run-2/train_discrete.npy",
        #     "X_graph_discrete_test":  "output_graphs/run-2/test_discrete.npy",
        #     "X_graph_continuous_train": "output_graphs/run-2/train_continuous.npy",
        #     "X_graph_continuous_test":  "output_graphs/run-2/test_continuous.npy",
        #     "y_train":      "train/Partition3_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl",
        #     "y_test":       "test/Partition4_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
        # },
        {
            "run_name": "Split 3",
            "X_mvts_train": "SWANSF-diff/train/diff4.pkl",
            "X_mvts_test":  "SWANSF-diff/test/diff5.pkl",
            "X_graph_discrete_train": "output_graphs/run-3/train_discrete.npy",
            "X_graph_discrete_test":  "output_graphs/run-3/test_discrete.npy",
            "X_graph_continuous_train": "output_graphs/run-3/train_continuous.npy",
            "X_graph_continuous_test":  "output_graphs/run-3/test_continuous.npy",
            "y_train":      "train/Partition4_Labels_RUS-Tomek-TimeGAN_LSBZM-Norm_WithoutC_FPCKNN-impute.pkl",
            "y_test":       "test/Partition5_Labels_LSBZM-Norm_FPCKNN-impute.pkl"
        },
        # You can add Split 2 and Split 3 here following the exact same key structure!
    ]
    
    evaluate_binary_models(run_configurations)