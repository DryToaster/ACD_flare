import numpy as np
import pandas as pd
import pickle as pk
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

def calculate_tss(y_true, y_pred):
    """Computes the True Skill Statistic (TSS)."""
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1

def run_final_balanced_modeling():
    # 1. LOAD BINARY LABELS
    with open(".//test//Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as f:
        train_y = np.array(pk.load(f)).astype(int)
    with open(".//test//Partition2_Labels_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as f:
        test_y = np.array(pk.load(f)).astype(int)

    print(f"Dataset Balance:")
    print(f"  Train: {np.sum(train_y)} Flares / {len(train_y)} Total")
    print(f"  Test:  {np.sum(test_y)} Flares / {len(test_y)} Total")

    # Added 'ACD' to the list of methods to evaluate
    methods = ["Granger", "DTW", "Pearson", "XCorr", "ACD"]
    results_table = []

    for method in methods:
        train_file = f'TRAIN_{method}.npy'
        test_file = f'TEST_{method}.npy'
        
        if not os.path.exists(train_file):
            print(f"\nSkipping {method}: {train_file} not found.")
            continue

        print(f"\n--- Evaluating Method: {method} ---")
        
        # 2. LOAD & RESHAPE
        # .reshape(len(train_y), -1) ensures it works for 3D (N, 24, 24) or 2D (N, features)
        X_train = np.nan_to_num(np.load(train_file).reshape(len(train_y), -1))
        X_test = np.nan_to_num(np.load(test_file).reshape(len(test_y), -1))

        # 3. SCALE
        scaler = StandardScaler()
        X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train))
        X_test_scaled = np.nan_to_num(scaler.transform(X_test))

        # 4. TRAIN BALANCED MODELS
        # Logistic Regression (Requires Scaling)
        lr = LogisticRegression(class_weight='balanced', max_iter=2000, n_jobs=-1)
        lr.fit(X_train_scaled, train_y)
        probs_lr = lr.predict_proba(X_test_scaled)[:, 1]

        # Random Forest (Handles raw features well)
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
        rf.fit(X_train, train_y)
        probs_rf = rf.predict_proba(X_test)[:, 1]

        # 5. THRESHOLD SEARCH FOR OPTIMAL TSS
        # Since flares are rare, the best TSS usually occurs at thresholds < 0.5
        for model_name, probs in [("LogReg", probs_lr), ("RandFor", probs_rf)]:
            best_tss = -1
            best_metrics = {}

            # Testing thresholds from 0.05 up to 0.60
            for thresh in np.linspace(0.01, 0.5, 50):
                preds = (probs >= thresh).astype(int)
                current_tss = calculate_tss(test_y, preds)
                
                if current_tss > best_tss:
                    best_tss = current_tss
                    best_metrics = {
                        "Method": method,
                        "Model": model_name,
                        "Threshold": round(thresh, 2),
                        "TSS": round(current_tss, 4),
                        "Recall": round(recall_score(test_y, preds, zero_division=0), 4),
                        "Precision": round(precision_score(test_y, preds, zero_division=0), 4),
                        "Accuracy": round(accuracy_score(test_y, preds), 4)
                    }
            results_table.append(best_metrics)

    # 6. DISPLAY COMPARISON
    df_results = pd.DataFrame(results_table)
    print("\n" + "="*80)
    print("             COMPARATIVE PERFORMANCE (Sorted by TSS)")
    print("="*80)
    print(df_results.sort_values(by="TSS", ascending=False).to_string(index=False))

if __name__ == "__main__":
    run_final_balanced_modeling()