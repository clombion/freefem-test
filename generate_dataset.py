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
FREEFEM_CMD = "/Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge un CSV exporté par FreeFEM.

    Format attendu : header "x,y,ux,uy,p" puis Ngrid lignes de données.

    Returns:
        X, Y, UX, UY, P : np.ndarray de forme (Ngrid,)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def run_simulation(nu: float) -> Path:
    """
    Lance FreeFEM pour la viscosité nu.

    Produit: DATA_DIR/fields_nu_{nu}.csv  (nom exact tel que FreeFEM le génère)

    Returns:
        Path vers le CSV produit.

    Raises:
        RuntimeError si FreeFEM retourne un code non nul.
        FileNotFoundError si FreeFEM n'a pas loggé de chemin exporté,
                          ou si le fichier créé a moins de 2602 lignes.
    """
    result = subprocess.run(
        [FREEFEM_CMD, "-nw", "stokes_cavity.edp", "-nu", str(nu)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[FreeFEM stderr] {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"FreeFEM a échoué pour nu={nu} (code {result.returncode})")

    # Récupère le chemin exact depuis la sortie FreeFEM (évite les problèmes
    # de conversion float→string entre Python et FreeFEM, ex: 2.0 vs "2").
    for line in result.stdout.splitlines():
        if line.startswith("Exported:"):
            csv_path = Path(line.split("Exported:", 1)[1].strip())
            # Vérifier que le fichier existe et a le bon nombre de lignes
            # (FreeFEM échoue silencieusement si le répertoire est absent)
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV annoncé mais absent: {csv_path}\n"
                    "Vérifier que data/snapshots/ existe avant de lancer FreeFEM."
                )
            with csv_path.open() as fh:
                line_count = sum(1 for _ in fh)
            if line_count != 2602:
                raise FileNotFoundError(
                    f"CSV tronqué: {csv_path} a {line_count} lignes, attendu 2602."
                )
            return csv_path

    raise FileNotFoundError(f"FreeFEM n'a pas loggé de fichier exporté pour nu={nu}")


def generate_dataset(nu_values: np.ndarray) -> dict[str, np.ndarray]:
    """
    Lance N simulations FreeFEM et consolide les résultats.

    Returns:
        dict avec clés: nu_values, X, Y, UX, UY, P
        Shapes : (N,), (Ngrid,), (Ngrid,), (N, Ngrid), (N, Ngrid), (N, Ngrid)
    """
    if len(nu_values) == 0:
        raise ValueError("nu_values must be non-empty")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = len(nu_values)

    # Première simulation pour déterminer Ngrid
    print(f"[1/{n}] nu = {nu_values[0]:.5f}", flush=True)
    csv0 = run_simulation(nu_values[0])
    X, Y, ux0, uy0, p0 = load_csv(csv0)
    ngrid = len(X)

    UX = np.zeros((n, ngrid))
    UY = np.zeros((n, ngrid))
    P  = np.zeros((n, ngrid))
    UX[0], UY[0], P[0] = ux0, uy0, p0

    for i in range(1, n):
        print(f"[{i+1}/{n}] nu = {nu_values[i]:.5f}", flush=True)
        csv = run_simulation(nu_values[i])
        _, _, UX[i], UY[i], P[i] = load_csv(csv)

    return {"nu_values": nu_values, "X": X, "Y": Y, "UX": UX, "UY": UY, "P": P}


if __name__ == "__main__":
    nu_values = np.logspace(np.log10(NU_MIN), np.log10(NU_MAX), N_SIM)
    dataset = generate_dataset(nu_values)

    out = Path("data/dataset.npz")
    tmp = out.with_suffix(".npz.tmp")
    np.savez(tmp, **dataset)
    tmp.rename(out)

    print(f"\nDataset sauvé : {out}")
    print(f"  nu_values shape : {dataset['nu_values'].shape}")
    print(f"  UX shape        : {dataset['UX'].shape}")
    print(f"  UY shape        : {dataset['UY'].shape}")
    print(f"  P  shape        : {dataset['P'].shape}")
