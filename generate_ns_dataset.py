# generate_ns_dataset.py
"""
Génère le dataset de simulations Navier-Stokes en faisant varier ν.
Produit: data/dataset_ns.npz

Usage:
    python generate_ns_dataset.py
"""
import subprocess
import sys
import numpy as np
from pathlib import Path

N_SIM = 30
NU_MIN = 0.01   # Re_max = 100
NU_MAX = 1.0    # Re_min = 1
DATA_DIR = Path("data/snapshots_ns")
FREEFEM_CMD = "/Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++"


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def run_simulation(nu: float) -> Path:
    result = subprocess.run(
        [FREEFEM_CMD, "-nw", "navier_stokes_cavity.edp", "-nu", str(nu)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[FreeFEM stderr] {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"FreeFEM failed for nu={nu} (code {result.returncode})")

    for line in result.stdout.splitlines():
        if line.startswith("Exported:"):
            csv_path = Path(line.split("Exported:", 1)[1].strip())
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV announced but missing: {csv_path}")
            with csv_path.open() as fh:
                line_count = sum(1 for _ in fh)
            if line_count != 2602:
                raise FileNotFoundError(
                    f"Truncated CSV: {csv_path} has {line_count} lines, expected 2602."
                )
            return csv_path

    raise FileNotFoundError(f"FreeFEM did not log exported file for nu={nu}")


def generate_dataset(nu_values: np.ndarray) -> dict[str, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = len(nu_values)

    print(f"[1/{n}] nu = {nu_values[0]:.5f} (Re = {1/nu_values[0]:.1f})", flush=True)
    csv0 = run_simulation(nu_values[0])
    X, Y, ux0, uy0, p0 = load_csv(csv0)
    ngrid = len(X)

    UX = np.zeros((n, ngrid))
    UY = np.zeros((n, ngrid))
    P  = np.zeros((n, ngrid))
    UX[0], UY[0], P[0] = ux0, uy0, p0

    for i in range(1, n):
        print(f"[{i+1}/{n}] nu = {nu_values[i]:.5f} (Re = {1/nu_values[i]:.1f})", flush=True)
        csv = run_simulation(nu_values[i])
        _, _, UX[i], UY[i], P[i] = load_csv(csv)

    return {"nu_values": nu_values, "X": X, "Y": Y, "UX": UX, "UY": UY, "P": P}


if __name__ == "__main__":
    nu_values = np.logspace(np.log10(NU_MAX), np.log10(NU_MIN), N_SIM)
    dataset = generate_dataset(nu_values)

    out = Path("data/dataset_ns.npz")
    tmp = out.with_suffix(".npz.tmp")
    np.savez(tmp, **dataset)
    tmp.rename(out)

    print(f"\nDataset saved: {out}")
    print(f"  nu_values shape : {dataset['nu_values'].shape}")
    print(f"  Re range        : {1/dataset['nu_values'].max():.1f} - {1/dataset['nu_values'].min():.1f}")
    print(f"  UX shape        : {dataset['UX'].shape}")
