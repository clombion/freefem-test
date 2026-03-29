# generate_dataset.py
"""
Génère le dataset de simulations Stokes en faisant varier ν.
Produit: data/dataset.npz

Usage:
    python generate_dataset.py
"""
import subprocess
import sys
import numpy as np
from pathlib import Path

N_SIM = 40
NU_MIN = 0.005
NU_MAX = 2.0
DATA_DIR = Path("data/snapshots")
FREEFEM_CMD = "FreeFem++"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge un CSV exporté par FreeFEM.

    Format attendu : header "x,y,ux,uy,p" puis Ngrid lignes de données.

    Returns:
        X, Y, UX, UY, P : np.ndarray de forme (Ngrid,)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
