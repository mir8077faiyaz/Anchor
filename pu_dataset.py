import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "data_provider"))

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
from data_gen import kvs
import info

# -------------------------------------------------
# SEED FUNCTION (NOW VARIABLE)
# -------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

def load_pu(data_root):
    X = sio.loadmat(f"{data_root}/PaviaU.mat")["paviaU"]
    y = sio.loadmat(f"{data_root}/PaviaU_gt.mat")["paviaU_gt"]
    return X, y


def load_houston(data_root):
    X = sio.loadmat(f"{data_root}/Houston.mat")["Houston"]
    y = sio.loadmat(f"{data_root}/Houston_gt.mat")["Houston_gt"]
    return X, y


# def load_pines(data_root):
#     X = sio.loadmat(f"{data_root}/Indian_pines.mat")["Pines"]
#     y = sio.loadmat(f"{data_root}/Indian_pines_gt.mat")["Pines_gt"]
#     return X, y

def load_pines(data_root):
    X = sio.loadmat(f"{data_root}/Indian_pines_corrected.mat")["indian_pines_corrected"]
    y = sio.loadmat(f"{data_root}/Indian_pines_gt.mat")["indian_pines_gt"]
    return X, y

def load_whlk(data_root):
    X = sio.loadmat(f"{data_root}/WHU_Hi_LongKou.mat")["WHU_Hi_LongKou"]
    y = sio.loadmat(f"{data_root}/WHU_Hi_LongKou_gt.mat")["WHU_Hi_LongKou_gt"]
    return X, y
# -------------------------------------------------
# NORMALIZATION
# -------------------------------------------------

def normalize_per_band(X):
    H, W, C = X.shape
    X = X.reshape(-1, C)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X.reshape(H, W, C)


# -------------------------------------------------
# GET LABELED PIXELS
# -------------------------------------------------

def get_labeled_indices(y):
    indices = []
    H, W = y.shape

    for i in range(H):
        for j in range(W):
            if y[i, j] > 0:
                indices.append((i, j))

    return indices


# -------------------------------------------------
# SPLIT (NOW SEED-CONTROLLED)
# -------------------------------------------------

def split_indices(indices, labels, train_ratio=0.2, seed=42):

    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels
    )

    return train_idx, test_idx



class OnTheFlyDataset(torch.utils.data.Dataset):

    def __init__(self, cube, labels, indices, patch_size):

        self.cube = cube
        self.labels = labels
        self.indices = indices
        self.patch_size = patch_size
        self.margin = patch_size // 2

        # Pad once
        self.cube = np.pad(
            self.cube,
            ((self.margin, self.margin),
             (self.margin, self.margin),
             (0, 0)),
            mode="constant"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        i, j = self.indices[idx]

        i_pad = i + self.margin
        j_pad = j + self.margin

        patch = self.cube[
            i_pad - self.margin : i_pad + self.margin + 1,
            j_pad - self.margin : j_pad + self.margin + 1
        ]

        label = self.labels[i, j] - 1

        # patch = torch.FloatTensor(patch).permute(2, 0, 1).unsqueeze(0)
        patch = torch.FloatTensor(patch).permute(2, 0, 1)

        return patch, torch.tensor(label, dtype=torch.long)
    

def get_train_test_split(X, y, train_ratio=0.2, seed=42):

    indices = get_labeled_indices(y)
    labels_for_split = [y[i, j] - 1 for i, j in indices]
    constant_seed = 42
    train_idx, test_idx = split_indices(
        indices,
        labels_for_split,
        train_ratio=train_ratio,
        seed=constant_seed
    )

    return train_idx, test_idx
