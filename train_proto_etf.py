"""
train_proto_etf.py

Main training script for the ETF-prototype denoising architecture.

- Trains ONLY on clean data + generic simulated spectral/spatial/shot
  perturbations (spectral_perturb, spatial_perturb, shot_noise_perturb).
- NEVER uses HyperTTA corruptions -- those are reserved exclusively for
  test.py, preserving a strict train/test corruption separation.
- Saves one checkpoint per seed: proto_etf_pu_seed{seed}.pth

pu_dataset / data_gen / info are treated as black boxes, imported the same
way as in the existing test_ssftt_diffusion_tent_multirun.py pipeline.
"""

import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "data_provider"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from pu_dataset import *  # noqa: F401,F403  (black box: load_pu, normalize_per_band,
                           # get_labeled_indices, split_indices, set_seed, OnTheFlyDataset)

from proto_etf_modules import ProtoEncoder, generate_etf_simplex
from training_pipeline import training_step, apply_training_perturbations

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_ROOT = r"/home/shafin/ECCV_Hyperspectral_TTA/data"

PATCH_SIZE = 15
BATCH_SIZE = 64

PROTO_DIM = 128
DIM = 64
EPOCHS = 60
LR = 1e-3
LAMBDA_CE = 1.0
CE_TEMPERATURE = 0.1
CD_WARMUP_EPOCHS = 10
# Phase-A: cross-degradation supervised contrastive alignment.
#
# Start conservatively. The ETF CE remains the primary classification
# objective; this term encourages clean/degraded representations
# of the same class to remain close.
LAMBDA_CD = 0.20

##################################
RD_WARMUP_EPOCHS = 10
##################################

CD_TEMPERATURE = 0.1
LAMBDA_RD = 0.15

# Perturbation family activation probabilities (per sample, independent)
P_SPECTRAL = 0.5
P_SPATIAL = 0.7
P_SHOT = 0.7

# P_SPECTRAL = 0.0
# P_SPATIAL = 0.0
# P_SHOT = 0.0

ETF_SEED = 2024  # fixed and independent of SEEDS -- ETF geometry must be
                  # identical across seeds and must match test_proto_etf.py

SEEDS = [42,43,44,45,46]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset name -> (loader function name, checkpoint prefix)
DATASETS = {
    "Pines": ("load_pines", "pines"),
    "Pavia": ("load_pu", "pu"),
    "WHLK": ("load_whlk", "whlk"),
}

# -------------------------------------------------
# Balanced-class batch sampler
# -------------------------------------------------
class BalancedClassBatchSampler(Sampler):
    """Every yielded batch has ~samples_per_class samples of every
    class (with-replacement resampling for classes with too few examples)."""

    def __init__(self, labels, samples_per_class, num_batches):
        self.samples_per_class = samples_per_class
        self.num_batches = num_batches
        labels = np.asarray(labels)
        self.class_to_indices = {
            int(c): np.where(labels == c)[0] for c in np.unique(labels)
        }
        self.classes = sorted(self.class_to_indices.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in self.classes:
                pool = self.class_to_indices[c]
                replace = len(pool) < self.samples_per_class
                chosen = np.random.choice(pool, self.samples_per_class, replace=replace)
                batch.extend(chosen.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def build_train_loader(X_raw, y, train_idx, train_labels):
    dataset = OnTheFlyDataset(X_raw, y, train_idx, PATCH_SIZE)
    num_classes = int(max(train_labels)) + 1
    samples_per_class = max(1, BATCH_SIZE // num_classes)  # dynamic, from num_classes
    batches_per_epoch = max(1, len(train_idx) // BATCH_SIZE)
    sampler = BalancedClassBatchSampler(
        train_labels, samples_per_class, batches_per_epoch
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    return loader, num_classes


def make_clean_noisy_pair(clean_norm_cube: torch.Tensor):
    """clean_norm_cube is ALREADY globally normalized (via normalize_per_band
    on the full training cube, before OnTheFlyDataset extracts patches --
    same convention as the existing tent_local_scg.py pipeline). Perturbing
    here operates entirely in that normalized space; shot_noise_perturb no
    longer requires raw non-negative data (see training_pipeline.py)."""
    noisy_norm_cube = apply_training_perturbations(
        clean_norm_cube, p_spectral=P_SPECTRAL, p_spatial=P_SPATIAL, p_shot=P_SHOT
    )
    return clean_norm_cube, noisy_norm_cube


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="Pines")
    args = parser.parse_args()
    loader_name, ckpt_prefix = DATASETS[args.dataset]

    X, y = globals()[loader_name](DATA_ROOT)
    X_raw = X.astype(np.float32)  # cast off uint16 before any processing
                                   # (torch.FloatTensor inside OnTheFlyDataset
                                   # can't convert uint16 directly)

    # Global per-band normalization on the WHOLE cube, matching the exact
    # convention used in the existing tent_local_scg.py pipeline
    # (X = normalize_per_band(X)). This is far more robust than per-patch
    # normalization: a single corrupted/extreme patch can't blow up its own
    # local statistics, since stats are computed once over the full image.
    X_norm = normalize_per_band(X_raw)

    indices = get_labeled_indices(y)
    labels_for_split = [y[i, j] - 1 for i, j in indices]

    train_idx, test_idx = split_indices(
        indices, labels_for_split, train_ratio=0.2, seed=42
    )
    train_labels = [y[i, j] - 1 for i, j in train_idx]

    SPECTRAL_DIM = X.shape[2]

    for seed in SEEDS:
        print(f"\n========== TRAIN ETF-Prototype Model (seed={seed}) ==========")
        set_seed(seed)

        loader, num_classes = build_train_loader(X_norm, y, train_idx, train_labels)

        model = ProtoEncoder(
            in_channels=SPECTRAL_DIM, patch_size=PATCH_SIZE, dim=DIM, proto_dim=PROTO_DIM
        ).to(device)

        from training_pipeline import Decoder
        # Decoder consumes invariant + private features (2 * PROTO_DIM).
        decoder = Decoder(
            latent_dim=PROTO_DIM * 2, out_channels=SPECTRAL_DIM, patch_size=PATCH_SIZE
        ).to(device)

        etf_vertices = generate_etf_simplex(
            num_classes=num_classes, feat_dim=PROTO_DIM, seed=ETF_SEED
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):

            epoch_stats = {
                "loss_total": 0.0,
                "loss_ce": 0.0,
                "loss_cd": 0.0,
                "loss_rd": 0.0,
            }

            cd_ramp = min(
                1.0,
                float(epoch + 1) / float(CD_WARMUP_EPOCHS),
            )

            rd_ramp = min(
                1.0,
                float(epoch + 1) / float(RD_WARMUP_EPOCHS),
            )

            current_lambda_cd = LAMBDA_CD * cd_ramp
            current_lambda_rd = LAMBDA_RD * rd_ramp

            n_batches = 0

            for cube, label in loader:
                # OnTheFlyDataset is assumed to yield (B, C, H, W) patches,
                # matching the pre-unsqueeze shape used elsewhere in the
                # existing pipeline (SSFTTnet does unsqueeze(1) after this
                # same dataset). Patches here are ALREADY globally normalized
                # (X_norm was built via normalize_per_band before the loader).
                cube = cube.to(device).float()
                label = label.to(device).long()

                clean_norm, noisy_norm = make_clean_noisy_pair(cube)

                stats = training_step(
                    model, optimizer, etf_vertices,
                    clean_norm, noisy_norm, label,
                    lambda_ce=LAMBDA_CE, ce_temperature=CE_TEMPERATURE,
                    lambda_cd=current_lambda_cd, cd_temperature=CD_TEMPERATURE,
                    lambda_rd=current_lambda_rd,
                )

                for k in epoch_stats:
                    epoch_stats[k] += stats[k]
                n_batches += 1

            for k in epoch_stats:
                epoch_stats[k] /= max(1, n_batches)

            print(
                f"[seed {seed}] epoch {epoch + 1}/{EPOCHS} "
                f"loss={epoch_stats['loss_total']:.4f} "
                f"ce={epoch_stats['loss_ce']:.4f} "
                f"cd={epoch_stats['loss_cd']:.4f} "
                f"rd={epoch_stats['loss_rd']:.4f}"
            )

        ckpt_path = f"proto_etf_{ckpt_prefix}_seed{seed}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "etf_vertices": etf_vertices.cpu(),
                "num_classes": num_classes,
                "proto_dim": PROTO_DIM,
                "dim": DIM,
                "patch_size": PATCH_SIZE,
                "spectral_dim": SPECTRAL_DIM,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")