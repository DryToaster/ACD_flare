# Amortized Causal Discovery

This repo contains a modification of the [original ACD repository](https://github.com/loeweX/AmortizedCausalDiscovery) from the official PyTorch implementation of:

Sindy Löwe*, David Madras*, Richard Zemel, Max Welling - [Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data](https://arxiv.org/abs/2006.10833)


## How to run the code

### Dependencies

- [Python and Conda](https://www.anaconda.com/)
- Setup the conda environment `ACD` by running:

    ```bash
    bash setup_dependencies.sh
    ```
  If you want to make use of your GPU, you might have to install a cuda-enabled pytorch version manually. Use the appropriate command provided [here](https://pytorch.org/) to achieve this.
- Don't forget to activate the environment and cd into the codebase directory when playing with the code later on

    ```bash
    source activate ACD
    cd codebase
    ```

## Causality-Aware Multivariate Time Series Classification for Solar Flare Prediction

This repository contains the official code implementation for predicting solar flares using Amortized Causal Discovery (ACD) and the CAIformer/GC-xLSTM architectures on the SWAN-SF dataset.

By decoupling causal graph generation from temporal sequence modeling, we extract the underlying magnetic field interactions (causal graphs) and utilize them to improve downstream binary classification performance (Flare vs. No-Flare).

### Repository Structure

stationarity.py: Analyzes the original time series data for stationarity using parallelized Augmented Dickey-Fuller (ADF) tests.

diff.py: Applies first-order differencing to correct non-stationary datasets.

perc.py: Utility script to load label files and calculate the percentage distribution of class representations.

make-discrete.py: Extracts and evaluates discrete causal graphs from single data partitions, generating visual heatmaps for standard and hidden-variable models.

graph-analysis.py: Generates visual charts (heatmaps, degree distributions, net causal flow, etc.) comparing positive and negative causal graphs.

graph-statistics.py: Performs rigorous statistical testing (Mann-Whitney U, Kolmogorov-Smirnov) to quantify graph differences.

make-perf-splits.py: Extracts discrete and continuous causal graphs from the trained encoder to construct the main experimental splits.

pipeline-EVAL.py: The main evaluation pipeline. Trains and evaluates baseline MVTS models, tabular models, and our proposed architectures (CAIformer, GC-xLSTM).

train.py: Trains a model using the ACD framework

### Step 1: Data Sourcing & Preprocessing

This project uses the SWAN-SF (Space Weather Analytics for Solar Flares) dataset. We specifically utilize the preprocessed and optimized version available at the Cleaned-SWANSF-Dataset repository.

#### 1. Acquire the Base Data:
The dataset from the repository above comes fully prepared for binary classification tasks (Flare vs. No-Flare). It features several robust preprocessing steps:

Imputation: FPCKNN technique.

Class Overlap Handling: Elimination of Class C samples.

Sampling: TimeGAN, Tomek Links, and Random Under Sampling (RUS).

Normalization: LSBZM normalization.

Download the 3D .pkl files from their repository. The data features 24 attributes and is shaped as (num_samples, num_timestamps, num_attributes). Tip: You can use perc.py to quickly verify the label balance of the partitions you download.

#### 2. Test for Stationarity:
Before feeding data into the causal discovery models, we must ensure it meets the stationarity assumptions required by autoregressive models. Use stationarity.py to run Augmented Dickey-Fuller (ADF) tests across all variables in your partitions.

python stationarity.py



Note: If the script reports high failure rates (indicating non-stationarity), proceed to the differencing step.

3. Apply Differencing:
Use diff.py to apply first-order differencing along the time axis, which stabilizes the mean of the time series. This script loads the original partitions and outputs them as diff[N].pkl. You can check the results with stationarity.py.

python diff.py



### Step 2: Training Amortized Causal Discovery (ACD)

With the stationary (differenced) dataset ready, train the ACD model to learn the causal interactions between the 24 magnetic field variables. We train this using the --flare_diff suffix to point to the correct dataset, define model structure, and set default hyperparameters.

Run the ACD training script on your target training partition:

python -m train \
    --suffix flare_diff \
    --epochs 300 \
    --encoder your_choice \
    --decoder you_choose \



(See the paper/appendix for the full list of ACD hyperparameters).

### Step 3: Graph Analysis & Visualization

Before running downstream classifiers, we analyze the causal structures learned by the ACD model to ensure they capture meaningful physical interactions.

#### 1. Generate Visual Charts:
Run graph-analysis.py to produce visualizations of the graphs extracted from the test set.

python graph-analysis.py



This generates:

Averaged causal heatmaps for Flare (Positive) vs. No-Flare (Negative) populations.

In-degree and Out-degree distributions per variable.

Net Causal Flow charts (Out-Degree minus In-Degree).

Whole-graph density and reciprocity distributions.

(Note: You can also use make-discrete.py for a quick standalone extraction and heatmap visualization of a single partition using standard or hidden-variable models).

#### 2. Execute Statistical Tests:
Run graph-statistics.py to quantify the statistical significance of the visual differences.

python graph-statistics.py



This script computes:

Graph-Level Metrics: Mann-Whitney U Tests for Density and Reciprocity.

Variable-Level Metrics: Kolmogorov-Smirnov (K-S) Tests for node-specific Out-Degree and Net Flow.

Outputs are printed to the console and saved to variable_ks_tests.csv.

### Step 4: Creating Experimental Splits

To prepare for downstream classification, we must generate the static graph datasets (both discrete and continuous) for every data split.

Configure the FILES_TO_PROCESS dictionary inside make-perf-splits.py to map your input differenced files (diff[N].pkl) to their respective output directories (e.g., output_graphs/run-3/).

Then execute the split generator:

python make-perf-splits.py



This script utilizes the frozen, trained ACD encoder to perform batched inference, yielding two files per partition:

train_discrete.npy / test_discrete.npy (Hard edge classifications, 0 or 1)

train_continuous.npy / test_continuous.npy (Soft edge probabilities, 0.0 to 1.0)

### Step 5: Downstream Classification & Evaluation

With the MVTS data and the ACD-generated graphs ready, you can execute the final experiment.

The pipeline-EVAL.py script orchestrates the training, evaluation, and metric calculation for 12 different model configurations, including:

CAIformer (with Discrete ACD, Continuous ACD, and Standalone)

GC-xLSTM (Standalone)

MVTS Baselines (MiniROCKET, Transformer, LSTM, RNN)

Tabular Baselines (Random Forest, MLP on flattened graphs)

Configure your target runs in the run_configurations dictionary at the bottom of the script, then execute:

python pipeline-EVAL.py



Expected Outputs:

Console Logs: Real-time printing of F1, TSS, HSS2, and AUC for each model on the current split. This will take a LONG time.

ROC Curves: High-resolution .png files (e.g., roc_curve_split_1.png) visualizing the True Positive Rate vs. False Positive Rate.

Final Averaged Metrics: The script concludes by printing the mean and standard deviation for all metrics across all evaluated splits (e.g., 3-Split Average).
