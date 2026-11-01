"""
test_proto_etf.py

Main test script for the ETF-prototype denoising architecture.

- Loads each seed's checkpoint from train_proto_etf.py.
- Applies REAL HyperTTA corruptions (JPEG, Additive, Poisson, Salt&Pepper,
  Stripe, Deadline, Blur, Fog, Zero-Mean) -- these are NEVER seen during
  training, preserving strict train/test corruption separation.
- For each (seed, corruption): evaluates
    1) No-adaptation: student forward -> cosine-sim to frozen ETF vertices -> argmax
    2) Entropy-TTA: EntropyTTA (norm-layer-only adaptation, bank frozen)
- Records OA/AA/Kappa per seed per corruption, then reports mean +/- std
  across the 5 seeds and a corruption-average summary, mirroring the
  existing test_ssftt_diffusion_tent_multirun.py reporting format.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "data_provider"))

import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import copy
from pu_dataset import *  # noqa: F401,F403  (black box: load_pu, normalize_per_band,
                           # get_labeled_indices, split_indices, set_seed, OnTheFlyDataset)

from metrics import compute_metrics, cosine_logits
from data_gen import kvs
import info

from proto_etf_modules import ProtoEncoder
from etf_tta import ETFTTA

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_ROOT = r"/home/shafin/ECCV_Hyperspectral_TTA/data"

PATCH_SIZE = 15
BATCH_SIZE = 64

TTA_TEMPERATURE = 0.1

# -------------------------------------------------
# ETF-aware TTA configuration
# -------------------------------------------------
ETF_TTA_LR = 1e-3
ETF_TTA_STEPS = 1
ETF_TTA_CONFIDENCE = 0.90 # CONDUCTING ABLATIONS ON THIS. ORIGINAL IS 0.9

SEEDS = [42,43,44,45,46]

# dataset name -> (HyperTTA data sign, checkpoint prefix, loader function name)
DATASETS = {
    "Pines": ("Indian", "pines", "load_pines"),
    "Pavia": ("Pavia", "pu", "load_pu"),
    "WHLK": ("WHLK", "whlk", "load_whlk"),
}
DATA_SIGN = "Indian"
CKPT_PREFIX = "pines"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# HyperTTA degradation wrapper (unchanged from the existing pipeline)
# -------------------------------------------------
def apply_hypertta_degradation(cube, noise_type, data_sign="Indian"):
    cls = kvs[noise_type]
    generator = cls(data_sign)
    params_dict = getattr(info, f"{data_sign}_kvs_params")
    params = params_dict[noise_type]
    generator.set_params(noise_type, params)
    cube = cube.astype(np.float32)
    return generator.gen(cube)


degradations = {
    "JPEG": lambda cube: apply_hypertta_degradation(cube, "jpeg", DATA_SIGN),
    "Zero-Mean": lambda cube: apply_hypertta_degradation(cube, "zmguass", DATA_SIGN),
    "Additive": lambda cube: apply_hypertta_degradation(cube, "additive", DATA_SIGN),
    "Poisson": lambda cube: apply_hypertta_degradation(cube, "poisson", DATA_SIGN),
    "Salt&Pepper": lambda cube: apply_hypertta_degradation(cube, "salt_pepper", DATA_SIGN),
    "Stripe": lambda cube: apply_hypertta_degradation(cube, "stripes", DATA_SIGN),
    "Deadline": lambda cube: apply_hypertta_degradation(cube, "deadlines", DATA_SIGN),
    "Blur": lambda cube: apply_hypertta_degradation(cube, "kernel", DATA_SIGN),
    "Fog": lambda cube: apply_hypertta_degradation(cube, "thin_fog", DATA_SIGN),
}

results_no_tta = {name: [] for name in degradations.keys()}
results_etf_tta = {name: [] for name in degradations.keys()}

def load_checkpoint(seed):
    ckpt = torch.load(f"proto_etf_{CKPT_PREFIX}_seed{seed}.pth", map_location=device)

    model = ProtoEncoder(
        in_channels=ckpt["spectral_dim"],
        patch_size=ckpt["patch_size"],
        dim=ckpt["dim"],
        proto_dim=ckpt["proto_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    etf_vertices = ckpt["etf_vertices"].to(device)
    return model, etf_vertices


@torch.no_grad()
def evaluate_no_adapt(model, etf_vertices, loader):
    model.eval()
    preds, labels = [], []
    for cube, label in loader:
        cube = cube.to(device).float()  # already globally normalized upstream
        proto = model(cube)
        logits = cosine_logits(proto, etf_vertices, temperature=TTA_TEMPERATURE)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(label.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return compute_metrics(labels, preds)


def evaluate_with_etf_tta(model, etf_vertices, loader):
    """
    ETF-aware TTA with reference-prediction stability filtering.

    - ETF vertices remain frozen.
    - Only BN/LN affine parameters are adapted.
    - Adaptation is non-episodic within a corruption.
    - Pseudo-labels come from the pre-adaptation ETF prediction.
    - Only confident samples whose prediction agrees with
      the original frozen model contribute to the ETF loss.
    """

    # ---------------------------------------------------------
    # Frozen reference model: original checkpoint
    # ---------------------------------------------------------
    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()

    for param in reference_model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # ETF-TTA model
    # ---------------------------------------------------------
    etf_tta = ETFTTA(
        model=model,
        etf_vertices=etf_vertices,
        reference_model=reference_model,
        lr=ETF_TTA_LR,
        steps=ETF_TTA_STEPS,
        temperature=TTA_TEMPERATURE,
        confidence_threshold=ETF_TTA_CONFIDENCE
    )


    preds = []
    labels = []
    diagnostics = []
    reference_agreements = []

    for cube, label in loader:
        cube = cube.to(device).float()

        # logits = etf_tta(cube)

        # preds.append(
        #     logits.argmax(dim=-1).cpu().numpy()
        # )
        # ---------------------------------------------------------
        # ETF-TTA prediction
        # ---------------------------------------------------------
        logits = etf_tta(cube)

        adapted_pred = logits.argmax(dim=-1)

        # ---------------------------------------------------------
        # Frozen reference prediction
        # ---------------------------------------------------------
        with torch.no_grad():
            reference_proto = reference_model(cube)

            reference_logits = cosine_logits(
                reference_proto,
                etf_vertices,
                temperature=TTA_TEMPERATURE,
            )

            reference_pred = reference_logits.argmax(dim=-1)

        # Agreement with the original frozen model.
        # This measures cumulative drift, not just within-batch change.
        reference_agreement = (
            adapted_pred == reference_pred
        ).float().mean().item()

        reference_agreements.append(reference_agreement)

        preds.append(
            adapted_pred.cpu().numpy()
        )



        labels.append(
            label.numpy()
        )

        if etf_tta.last_diagnostics is not None:
            diagnostics.append(
                etf_tta.last_diagnostics
            )

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    metrics = compute_metrics(labels, preds)

    if diagnostics:
        selected_fraction = np.mean([
            d["selected_fraction"]
            for d in diagnostics
        ])

        alignment_before = np.mean([
            d["alignment_before"]
            for d in diagnostics
        ])

        alignment_after = np.mean([
            d["alignment_after"]
            for d in diagnostics
        ])

        prediction_change_rate = np.mean([
            d["prediction_change_rate"]
            for d in diagnostics
        ])

        entropy_before = np.mean([
            d["entropy_before"]
            for d in diagnostics
        ])

        entropy_after = np.mean([
            d["entropy_after"]
            for d in diagnostics
        ])

        confidence_before = np.mean([
            d["confidence_before"]
            for d in diagnostics
        ])

        confidence_after = np.mean([
            d["confidence_after"]
            for d in diagnostics
        ])

        stable_fraction = np.mean([
            d["stable_fraction"]
            for d in diagnostics
        ])

        selected_stable_fraction = np.mean([
            d["selected_stable_fraction"]
            for d in diagnostics
        ])

    else:
        selected_fraction = 0.0
        alignment_before = 0.0
        alignment_after = 0.0
        prediction_change_rate = 0.0
        entropy_before = 0.0
        entropy_after = 0.0
        confidence_before = 0.0
        confidence_after = 0.0
        stable_fraction = 0.0
        selected_stable_fraction = 0.0

    if reference_agreements:
        reference_agreement = float(np.mean(reference_agreements))
    else:
        reference_agreement = 1.0

    diag = {
        "selected_fraction": float(selected_fraction),
        "stable_fraction": float(stable_fraction),
        "selected_stable_fraction": float(selected_stable_fraction),
        "alignment_before": float(alignment_before),
        "alignment_after": float(alignment_after),
        "prediction_change_rate": float(prediction_change_rate),
        "entropy_before": float(entropy_before),
        "entropy_after": float(entropy_after),
        "confidence_before": float(confidence_before),
        "confidence_after": float(confidence_after),
    }

    return (*metrics, diag)

def print_corruption_diagnostics(name, X_raw, X_corr):
    """Quick sanity check on the corrupted cube's shape/value range before
    normalization -- flags the kind of blowup or shape mismatch that
    silently wrecks downstream normalization/alignment and tanks accuracy
    for specific corruptions (Fog has shown a debug print from inside the
    black-box generator suggesting it may handle the cube differently --
    the shape check below confirms whether it's still (H, W, C))."""
    if X_corr.shape != X_raw.shape:
        print(
            f"[diag] {name}: SHAPE MISMATCH -- X_raw={X_raw.shape} "
            f"X_corr={X_corr.shape}. This will misalign pixels against the "
            f"label map (i, j) indices used by OnTheFlyDataset."
        )

    finite = np.isfinite(X_corr)
    n_bad = X_corr.size - finite.sum()
    vals = X_corr[finite]
    print(
        f"[diag] {name}: shape={X_corr.shape} min={vals.min():.3f} max={vals.max():.3f} "
        f"mean={vals.mean():.3f} std={vals.std():.3f} non-finite={n_bad}"
    )


def print_summary(results, title):
    print(f"\n================ {title} ================")
    for name, vals in results.items():
        vals = np.array(vals)
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        print(f"\n{name}")
        print(f"OA: {mean[0]:.2f} +/- {std[0]:.2f}")
        print(f"AA: {mean[1]:.2f} +/- {std[1]:.2f}")
        print(f"Kappa: {mean[2]:.2f} +/- {std[2]:.2f}")

    oa_list = [np.array(v).mean(axis=0)[0] for v in results.values()]
    aa_list = [np.array(v).mean(axis=0)[1] for v in results.values()]
    kappa_list = [np.array(v).mean(axis=0)[2] for v in results.values()]
    print(f"\n---- Corruption average ({title}) ----")
    print(f"mOA: {np.mean(oa_list):.2f}")
    print(f"mAA: {np.mean(aa_list):.2f}")
    print(f"mKappa: {np.mean(kappa_list):.2f}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="Pines")
    args = parser.parse_args()
    DATA_SIGN, CKPT_PREFIX, loader_name = DATASETS[args.dataset]

    X, y = globals()[loader_name](DATA_ROOT)
    X_raw = X.copy()  # apply_hypertta_degradation() already casts to float32
                       # internally before generating, so X_corr comes out
                       # float32 regardless of X's original dtype (e.g. uint16)

    indices = get_labeled_indices(y)
    labels_for_split = [y[i, j] - 1 for i, j in indices]

    train_idx, test_idx = split_indices(
        indices, labels_for_split, train_ratio=0.2, seed=42
    )

    for seed in SEEDS:
        print(f"\n========== TEST ETF-Prototype Model (seed={seed}) ==========")
        set_seed(seed)

        model, etf_vertices = load_checkpoint(seed)

        for name, degrade_fn in degradations.items():
            print(f"\n--- {name} ---")

            X_corr = degrade_fn(X_raw.copy())  # raw-scale corruption, as in existing pipeline
            X_corr = X_corr.astype(np.float32)  # some HyperTTA generators cast back to
                                                 # the input dtype (e.g. uint16) internally;
                                                 # force float32 before normalizing
            print_corruption_diagnostics(name, X_raw, X_corr)

            # Global per-band normalization on the WHOLE corrupted cube --
            # matches the exact convention used in tent_local_scg.py
            # (X_corr = normalize_per_band(X_corr)) and in train_proto_etf.py.
            # This is what was missing before: per-patch normalization let a
            # single extreme/corrupted patch blow up its own local stats,
            # which is the most likely cause of the near-chance Poisson and
            # Zero-Mean results.
            X_corr_norm = normalize_per_band(X_corr)

            dataset = OnTheFlyDataset(X_corr_norm, y, test_idx, PATCH_SIZE)

            # shuffle=True is important here, not cosmetic: test_idx pixels
            # are spatially contiguous (neighboring HSI pixels are usually
            # the same class), so shuffle=False would give the diversity
            # regularizer in EntropyTTA batches that are legitimately
            # single-class-dominated -- it would then be actively wrong to
            # force those batches toward a spread-out marginal prediction.
            # Shuffling makes each batch closer to the i.i.d./class-diverse
            # assumption the diversity term relies on. Order doesn't affect
            # metrics either way since predictions+labels are pooled before
            # compute_metrics.
            g = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                num_workers=4, pin_memory=True,
            )

            oa, aa, kappa = evaluate_no_adapt(model, etf_vertices, loader)
            print(f"[NO ADAPT] OA={oa:.2f} AA={aa:.2f} Kappa={kappa:.2f}")
            results_no_tta[name].append([oa, aa, kappa])

            # ---------------------------------------------------------
            # ETF-AWARE TTA
            # ---------------------------------------------------------
            model_for_etf_tta, _ = load_checkpoint(seed)

            etf_oa, etf_aa, etf_kappa, etf_diag = evaluate_with_etf_tta(
                model_for_etf_tta,
                etf_vertices,
                loader,
            )

            # print(
            #     f"[ETF TTA] OA={etf_oa:.2f} "
            #     f"AA={etf_aa:.2f} "
            #     f"Kappa={etf_kappa:.2f} "
            #     f"selected={etf_diag['selected_fraction']:.3f} "
            #     f"align={etf_diag['alignment_before']:.4f}"
            #     f"->{etf_diag['alignment_after']:.4f} "
            #     f"change={etf_diag['prediction_change_rate']:.3f}"
            # )
            print(
                f"ETF-TTA: OA={etf_oa:.2f}, AA={etf_aa:.2f}, "
                f"Kappa={etf_kappa:.4f} | "
                f"Selected={etf_diag['selected_fraction']:.3f} | "
                f"Stable={etf_diag['stable_fraction']:.3f} | "
                f"StableSelected={etf_diag['selected_stable_fraction']:.3f} | "
                f"Conf={etf_diag['confidence_before']:.3f}"
                f"->{etf_diag['confidence_after']:.3f} | "
                f"Entropy={etf_diag['entropy_before']:.3f}"
                f"->{etf_diag['entropy_after']:.3f} | "
                f"Align={etf_diag['alignment_before']:.4f}"
                f"->{etf_diag['alignment_after']:.4f} | "
                f"Change={etf_diag['prediction_change_rate']:.3f}"
            )

            results_etf_tta[name].append(
                [etf_oa, etf_aa, etf_kappa]
            )

    print_summary(results_no_tta, "FINAL V2 ETF NO-TTA RESULTS")
    print_summary(results_etf_tta, "ETF-AWARE TTA")