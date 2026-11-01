# metrics.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proto_etf_modules import ProtoEncoder
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    oa = np.mean(y_true == y_pred)

    class_acc = np.nan_to_num(np.diag(cm) / cm.sum(axis=1))
    aa = np.mean(class_acc)

    kappa = cohen_kappa_score(y_true, y_pred)

    return oa*100, aa*100, kappa*100


def cosine_logits(proto: torch.Tensor, etf_vertices: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """proto: (B, dim) unnormalized OK (we normalize here); etf_vertices: (num_classes, dim), unit-norm."""
    proto_n = F.normalize(proto, dim=-1)
    sims = proto_n @ etf_vertices.t()  # (B, num_classes), cosine similarity
    return sims / temperature