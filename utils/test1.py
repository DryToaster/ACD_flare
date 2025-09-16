import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def load_sep_data(batch_size=1, datadir="data"):
    from glob import glob
    import pandas as pd
    print("Loading data from {}".format(datadir))
    g = os.path.join(datadir, "stox", "*.csv")
    
    l = glob(g)
    loc_train = torch.zeros((len(l), 5, 3019))
    
    for i in range(len(l)):
        df = pd.read_csv(l[i])
        df = df.drop(columns=["Name", "Date"])
        df = (df-df.min())/(df.max()-df.min())
        mat = df.to_numpy()
        tens = torch.Tensor(mat)
        tens = tens.reshape((5, 3019))
        loc_train[i] = tens
    print(loc_train.shape)
    edges_train = torch.ones((len(l), 5, 5))
    for i in range(edges_train.shape[1]):
        for j in range(edges_train.shape[0]):
            edges_train[j][i][i] = 0
    
    loc_max = loc_train.max()
    loc_min = loc_train.min()

    train_data = TensorDataset(loc_train, edges_train)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    return (train_data_loader, loc_max, loc_min)
load_sep_data()