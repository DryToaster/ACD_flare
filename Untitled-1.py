import torch
import numpy as np
import pandas as pd
import pickle as pk
import os
from utils import arg_parser, logger, data_loader, forward_pass_and_eval
from model import utils, model_loader

bigone = None
#print(os.cwd())
with open(".\\test\\Partition1_LSBZM-Norm_FPCKNN-impute.pkl", 'rb') as _part:
    bigone = pk.load(_part, )

print("Shape: ", bigone.shape)


sample = torch.tensor(bigone[0])
print(sample.shape)

with torch.no_grad():
    out = '1'

args = arg_parser.parse_args()
print(args.save_folder)
#logs = logger.Logger(args)

#if args.GPU_to_use is not None:
    #logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

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

train_iter = iter(train_loader)
first = next(train_iter)
fourth = next(train_iter)

sd = torch.load("encoder.pt")
encoder.load_state_dict(sd)
encoder.eval()

#print(first)

res = encoder(fourth[0], rel_rec, rel_send)
res2 = res[:, :, 1:2]
res1 = res[:, :, 0:1]
#print(res[:, :, 1:2])
resctangle = res2.reshape(23,24)

import matplotlib.pyplot as plt
import torch

# Convert tensor to numpy (matplotlib works with numpy arrays)
heatmap_data = resctangle.detach().cpu().numpy()
print(heatmap_data)
# Create the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Heatmap Visualization')
plt.xlabel('Column')
plt.ylabel('Row')
plt.show()

print("Done")