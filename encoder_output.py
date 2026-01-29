import torch
import numpy as np
import pandas as pd
import pickle as pk
import os
from utils import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset

print(torch.cuda.is_available())
bigone = None
#print(os.cwd())
with open(".\\test\\Partition2_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as _part:
    bigone = pk.load(_part, )

print("Shape: ", bigone.shape)


sample = torch.tensor(bigone[0])
#print(sample.shape)

with torch.no_grad():
    out = '1'

args = arg_parser.parse_args()
#print(args.save_folder)
#logs = logger.Logger(args)

#if args.GPU_to_use is not None:
    #logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))
print(torch.cuda.is_available())
(
    train_loader,
    valid_loader,
    test_loader,
    loc_max,
    loc_min,
    vel_max,
    vel_min,
) = data_loader.load_data(args)

rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)

encoder, decoder, optimizer, scheduler, edge_probs = model_loader.load_model(
    args, loc_max, loc_min, vel_max, vel_min
)
print(type(train_loader))
train_iter = iter(train_loader)
feats = ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSX',
 'EPSY', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX']
sd = torch.load("encoder.pt")
encoder.load_state_dict(sd)
if isinstance(encoder, torch.nn.DataParallel):
    encoder = encoder.module
print(encoder)
encoder.eval()
encoder = encoder.to('cpu')

bigone = np.transpose(bigone, (0,2,1))
f = open(".//test//Partition1_Labels_LSBZM-Norm_FPCKNN-impute.pkl", 'rb')
dat = pk.load(f)
f.close()

all_out = np.zeros((88557, 24, 24))

# agg0 = np.zeros((24,24))
# agg0c = 0

# agg1 = np.zeros((24,24))
# agg1c = 0
# t_total = 0
BATCH_SIZE = 128
DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
static_mask = ~torch.eye(24, dtype=torch.bool, device=DEVICE)
dataset = TensorDataset(torch.from_numpy(bigone).float()) 
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. PREPARE STATIC INPUTS ---
# If rel_rec/rel_send are static (the same for every item), move them to GPU once.
# Note: Some encoders require these to be expanded to match batch size.
# If your encoder fails on shape mismatch, uncomment the .expand lines below.
rel_rec = rel_rec.to(DEVICE)
rel_send = rel_send.to(DEVICE)
# rel_rec = rel_rec.unsqueeze(0).expand(BATCH_SIZE, -1, -1) 
# rel_send = rel_send.unsqueeze(0).expand(BATCH_SIZE, -1, -1)

# Pre-allocate output container on CPU to hold results
# 73492 items, 24x24 matrix
all_out = np.zeros((88557, 24, 24), dtype=np.float32)

print(f"Processing {88557} items with Batch Size {BATCH_SIZE}...")
start_total = time.perf_counter()

# --- 4. THE OPTIMIZED LOOP ---
current_idx = 0

# Disable gradient calculation to save VRAM and speed up inference
with torch.no_grad(): 
    for batch in loader:
        # batch is a list [data], so take batch[0]
        inputs = batch[0].to(DEVICE)
        
        # Determine current batch size (last batch might be smaller than 64)
        curr_batch_size = inputs.size(0)

        # A. Run Encoder on the whole batch
        # res shape example: [64, N, M]
        res = encoder(inputs, rel_rec, rel_send)

        # B. Vectorized Slicing & Reshaping
        # We process 64 items at once. 
        # Target shape for resctangle: [64, 24, 23]
        res2 = res[:, :, 1:2] 
        resctangle = res2.reshape(curr_batch_size, 24, 23)

        # C. The Logic Fix (Matrix reconstruction on GPU)
        # Create an empty container: [64, 24, 24]
        matrix_batch = torch.zeros((curr_batch_size, 24, 24), device=DEVICE)
        
        # Expand the static mask to match the batch: [64, 24, 24]
        batch_mask = static_mask.unsqueeze(0).expand(curr_batch_size, -1, -1)
        
        # MAGIC STEP: Fill non-diagonal spots using the mask.
        # This replaces the nested "for i, for j" loops entirely.
        # .flatten() ensures the data flows into the True spots of the mask sequentially
        matrix_batch[batch_mask] = resctangle.flatten()

        # D. Move to CPU and store
        # We transfer 64 items at once, reducing synchronization overhead by 64x
        all_out[current_idx : current_idx + curr_batch_size] = matrix_batch.cpu().numpy()
        
        current_idx += curr_batch_size

t_total = time.perf_counter() - start_total
avg = t_total / 88557
print(f"{"ACD":<10} | Total CPU Time: {t_total/3600:.2f} hrs | Avg/Sample: {avg*1000:.2f} ms")

np.save('output_test.npy', all_out)

agg0 = agg0 / agg0c
agg0 = (agg0-agg0.min())/(agg0.max()-agg0.min())
agg1 = agg1 / agg1c
agg0 = (agg1-agg1.min())/(agg1.max()-agg1.min())
agg = agg1 - agg0

import seaborn as sns

heatmap_data = np.clip(agg0, -5, 10)
plt.figure(figsize=(10,8))
sns.heatmap(heatmap_data, 
            xticklabels=feats, 
            yticklabels=feats,
            annot=False,  # Show values in cells
            cmap='OrRd',  # Color scheme
            fmt='g')  # Format for annotations

plt.title('Average Predicted Edge Probability - Non-Flares (72238 samples)')
plt.xlabel('Target')
plt.ylabel('Predictor')
plt.show()

heatmap_data = np.clip(agg1, -5, 10)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, 
            xticklabels=feats, 
            yticklabels=feats,
            annot=False,  # Show values in cells
            cmap='OrRd',  # Color scheme
            fmt='g')  # Format for annotations

plt.title('Average Predicted Edge Probability - Flares > C (1254 samples)')
plt.xlabel('Target')
plt.ylabel('Predictor')
plt.show()

print("Negative: ",agg0c)
print("Positive: ",agg1c)

heatmap_data = agg
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, 
            xticklabels=feats, 
            yticklabels=feats,
            annot=False,  # Show values in cells
            cmap='OrRd',  # Color scheme
            fmt='g')  # Format for annotations

plt.title('Difference in Average Predicted Edge Probability, non-flare minus positive-flare')
plt.xlabel('Target')
plt.ylabel('Predictor')
plt.show()

print("Done")