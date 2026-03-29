# train_surrogate.py
"""
Surrogate POD + régression polynomiale pour la cavité de Stokes.

Usage:
    python train_surrogate.py [--data data/dataset.npz] [--k 5] [--deg 3]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from pathlib import Path


def compute_pod(
    snapshots: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Décomposition POD par SVD tronquée.

    Args:
        snapshots : (N, Ngrid) matrice de snapshots (N simulations, Ngrid points)
        k         : nombre de modes à retenir

    Returns:
        mean   : (Ngrid,)  moyenne des snapshots
        modes  : (k, Ngrid) modes POD (vecteurs propres spatiaux)
        coeffs : (N, k)    coordonnées des snapshots dans la base POD
        energy : float     fraction d'énergie capturée par les k modes (∈ ]0, 1])
    """
    mean = snapshots.mean(axis=0)
    U_c = snapshots - mean
    _, s, Vt = np.linalg.svd(U_c, full_matrices=False)
    rank = Vt.shape[0]
    if k > rank:
        import warnings
        warnings.warn(f"k={k} exceeds matrix rank {rank}; clamped to {rank}", stacklevel=2)
        k = rank
    modes = Vt[:k]
    coeffs = U_c @ modes.T
    energy = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return mean, modes, coeffs, energy


def make_regression_pipe(degree: int = 3, alpha: float = 1e-3):
    """Pipeline sklearn : PolynomialFeatures(degree) + Ridge(alpha).

    Note: Designed to regress on 1/ν features (pass 1/nu, not nu).
    """
    return make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))


def fit_surrogate(
    nu_train: np.ndarray,
    snapshots_train: np.ndarray,
    k: int = 5,
    degree: int = 3,
) -> tuple:
    """
    Entraîne un surrogate POD + régression pour un champ donné.

    Note: La régression est effectuée sur 1/ν (physique de Stokes : u ∝ 1/ν).

    Returns:
        mean   : (Ngrid,)
        modes  : (k, Ngrid)
        pipe   : pipeline sklearn ajusté (1/ν → coefficients POD)
        energy : fraction d'énergie POD
    """
    mean, modes, coeffs, energy = compute_pod(snapshots_train, k)
    pipe = make_regression_pipe(degree)
    pipe.fit((1.0 / nu_train).reshape(-1, 1), coeffs)
    return mean, modes, pipe, energy


def predict_field(
    nu_query: np.ndarray,
    mean: np.ndarray,
    modes: np.ndarray,
    pipe,
) -> np.ndarray:
    """
    Prédit le champ pour les valeurs de ν données.

    Note: La régression est effectuée sur 1/ν (physique de Stokes : u ∝ 1/ν).

    Args:
        nu_query : (M,) ou scalaire
        mean     : (Ngrid,)
        modes    : (k, Ngrid)
        pipe     : pipeline ajusté

    Returns:
        (M, Ngrid) champs prédits
    """
    nu_inv = 1.0 / np.atleast_1d(nu_query)
    coeffs_pred = pipe.predict(nu_inv.reshape(-1, 1))
    return coeffs_pred @ modes + mean


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Erreur L2 relative moyennée sur les échantillons.

    Args:
        pred : (M, Ngrid)
        true : (M, Ngrid)

    Returns:
        float — moyenne sur les M échantillons de ||pred_i - true_i|| / ||true_i||
    """
    norms_true = np.linalg.norm(true, axis=1)
    norms_err = np.linalg.norm(pred - true, axis=1)
    return float((norms_err / (norms_true + 1e-15)).mean())
