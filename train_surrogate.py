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


def plot_field_comparison(
    X: np.ndarray,
    Y: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    nu_val: float,
    field_name: str,
    axes: tuple,
) -> None:
    """
    Affiche côte à côte : champ prédit, champ simulé, erreur absolue.

    Args:
        X, Y       : (Ngrid,) coordonnées de la grille (doit être un carré parfait)
        pred       : (Ngrid,) champ prédit
        true       : (Ngrid,) champ simulé
        nu_val     : valeur de ν (pour le titre)
        field_name : "ux", "uy" ou "p"
        axes       : tuple de 3 Axes matplotlib (prédit, simulé, erreur)
    """
    n = int(round(len(X) ** 0.5))
    xi = X.reshape(n, n)
    yi = Y.reshape(n, n)

    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    ax_pred, ax_true, ax_err = axes
    im_pred = ax_pred.contourf(xi, yi, pred.reshape(n, n), levels=30, vmin=vmin, vmax=vmax)
    plt.colorbar(im_pred, ax=ax_pred, shrink=0.8)
    ax_pred.set_title(f"{field_name} prédit (ν={nu_val:.4f})")
    ax_pred.set_aspect("equal")

    im_true = ax_true.contourf(xi, yi, true.reshape(n, n), levels=30, vmin=vmin, vmax=vmax)
    plt.colorbar(im_true, ax=ax_true, shrink=0.8)
    ax_true.set_title(f"{field_name} simulé")
    ax_true.set_aspect("equal")

    err = np.abs(pred - true)
    im_err = ax_err.contourf(xi, yi, err.reshape(n, n), levels=30, cmap="hot_r")
    plt.colorbar(im_err, ax=ax_err, shrink=0.8)
    ax_err.set_title(f"|erreur| max={err.max():.2e}")
    ax_err.set_aspect("equal")


def main(data_path: str = "data/dataset.npz", k: int = 5, degree: int = 3) -> None:
    # --- Chargement ---
    data = np.load(data_path)
    nu_all = data["nu_values"]
    X, Y = data["X"], data["Y"]
    UX, UY, P = data["UX"], data["UY"], data["P"]
    print(f"Dataset : {len(nu_all)} simulations, {UX.shape[1]} points/grille")

    # --- Split train / test (5 points de test, interpolation) ---
    rng = np.random.default_rng(42)
    test_idx = rng.choice(len(nu_all), size=5, replace=False)
    train_mask = np.ones(len(nu_all), dtype=bool)
    train_mask[test_idx] = False

    nu_train, nu_test = nu_all[train_mask], nu_all[test_idx]
    UX_train, UX_test = UX[train_mask], UX[test_idx]
    UY_train, UY_test = UY[train_mask], UY[test_idx]
    P_train,  P_test  = P[train_mask],  P[test_idx]

    print(f"Train : {len(nu_train)} | Test : {len(nu_test)}")

    # --- Entraînement (un surrogate par champ) ---
    mean_ux, modes_ux, pipe_ux, e_ux = fit_surrogate(nu_train, UX_train, k, degree)
    mean_uy, modes_uy, pipe_uy, e_uy = fit_surrogate(nu_train, UY_train, k, degree)
    mean_p,  modes_p,  pipe_p,  e_p  = fit_surrogate(nu_train, P_train,  k, degree)

    print(f"\nÉnergie POD capturée (k={k} modes):")
    print(f"  ux : {e_ux * 100:.4f}%")
    print(f"  uy : {e_uy * 100:.4f}%")
    print(f"  p  : {e_p  * 100:.4f}%")

    # --- Prédiction sur le set de test ---
    UX_pred = predict_field(nu_test, mean_ux, modes_ux, pipe_ux)
    UY_pred = predict_field(nu_test, mean_uy, modes_uy, pipe_uy)
    P_pred  = predict_field(nu_test, mean_p,  modes_p,  pipe_p)

    err_ux = relative_l2_error(UX_pred, UX_test)
    err_uy = relative_l2_error(UY_pred, UY_test)
    err_p  = relative_l2_error(P_pred,  P_test)

    print(f"\nErreur L2 relative sur test:")
    print(f"  ux : {err_ux * 100:.4f}%")
    print(f"  uy : {err_uy * 100:.4f}%")
    print(f"  p  : {err_p  * 100:.4f}%")

    # --- Scalaires : max|u| vs ν (dataset complet + surrogate sweep) ---
    nu_sweep = np.logspace(np.log10(nu_all.min()), np.log10(nu_all.max()), 200)
    UX_sw = predict_field(nu_sweep, mean_ux, modes_ux, pipe_ux)
    UY_sw = predict_field(nu_sweep, mean_uy, modes_uy, pipe_uy)
    max_u_sweep = np.max(np.sqrt(UX_sw ** 2 + UY_sw ** 2), axis=1)
    max_u_data  = np.max(np.sqrt(UX ** 2 + UY ** 2), axis=1)
    p_mean_data = P.mean(axis=1)

    # --- Figure 1 : comparaison champs pour le premier ν de test ---
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.suptitle(f"Surrogate POD (k={k}, deg={degree}) — ν_test = {nu_test[0]:.4f}")
    for row, (pred_row, true_row, name) in enumerate([
        (UX_pred[0], UX_test[0], "ux"),
        (UY_pred[0], UY_test[0], "uy"),
        (P_pred[0],  P_test[0],  "p"),
    ]):
        plot_field_comparison(X, Y, pred_row, true_row, nu_test[0], name,
                              (axes[row, 0], axes[row, 1], axes[row, 2]))
    plt.tight_layout()
    out1 = Path("data/comparison_fields.png")
    plt.savefig(out1, dpi=150)
    print(f"\nFigure sauvée : {out1}")

    # --- Figure 2 : scalaires vs ν ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Scalaires : surrogate vs simulations")

    ax1.loglog(nu_all, max_u_data, "o", markersize=4, label="Simulé")
    ax1.loglog(nu_sweep, max_u_sweep, "-", linewidth=2, label="Surrogate")
    ax1.set_xlabel("ν")
    ax1.set_ylabel("max |u|")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("max |u| vs ν")

    ax2.semilogx(nu_all, p_mean_data, "o", markersize=4, label="Simulé")
    ax2.set_xlabel("ν")
    ax2.set_ylabel("pression moyenne")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Pression moyenne vs ν")

    plt.tight_layout()
    out2 = Path("data/scalar_comparison.png")
    plt.savefig(out2, dpi=150)
    print(f"Figure sauvée : {out2}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surrogate POD pour Stokes cavité")
    parser.add_argument("--data", default="data/dataset.npz", help="Chemin du dataset .npz")
    parser.add_argument("--k",   type=int, default=5, help="Nombre de modes POD")
    parser.add_argument("--deg", type=int, default=3, help="Degré polynomial")
    args = parser.parse_args()
    main(args.data, args.k, args.deg)
