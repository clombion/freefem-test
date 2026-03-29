# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "numpy>=1.26",
#     "scikit-learn>=1.4",
#     "matplotlib>=3.8",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    return mo, np, plt, Path


@app.cell
def _(mo):
    mo.md("""
# Surrogate POD — Stokes en cavité entraînée

Ce notebook charge un modèle surrogate (POD + régression polynomiale)
entraîné sur des simulations FreeFEM de l'écoulement de Stokes
dans une cavité carrée avec couvercle entraîné.

**Déplacez le slider ν** pour prédire instantanément les champs de vitesse
et de pression sans relancer FreeFEM.
""")
    return


@app.cell
def _(np, Path):
    from train_surrogate import fit_surrogate, predict_field

    data = np.load(Path("data/dataset.npz"))
    nu_all = data["nu_values"]
    X, Y = data["X"], data["Y"]
    UX, UY, P = data["UX"], data["UY"], data["P"]
    n_grid = int(round(len(X) ** 0.5))
    return fit_surrogate, predict_field, nu_all, X, Y, UX, UY, P, n_grid


@app.cell
def _(fit_surrogate, nu_all, UX, UY, P):
    k, degree = 5, 3
    mean_ux, modes_ux, pipe_ux, e_ux = fit_surrogate(nu_all, UX, k, degree)
    mean_uy, modes_uy, pipe_uy, e_uy = fit_surrogate(nu_all, UY, k, degree)
    mean_p, modes_p, pipe_p, e_p = fit_surrogate(nu_all, P, k, degree)
    return (
        k, degree,
        mean_ux, modes_ux, pipe_ux, e_ux,
        mean_uy, modes_uy, pipe_uy, e_uy,
        mean_p, modes_p, pipe_p, e_p,
    )


@app.cell
def _(mo, e_ux, e_uy, e_p, k):
    mo.md(
        f"""
        **Modèle entraîné** — POD avec k={k} modes

        | Champ | Énergie capturée |
        |-------|-----------------|
        | ux    | {e_ux * 100:.2f}% |
        | uy    | {e_uy * 100:.2f}% |
        | p     | {e_p * 100:.2f}% |
        """
    )
    return


@app.cell
def _(mo, nu_all):
    nu_slider = mo.ui.slider(
        start=float(nu_all.min()),
        stop=float(nu_all.max()),
        value=0.1,
        step=0.001,
        label="ν (viscosité cinématique)",
        full_width=True,
    )
    nu_slider
    return (nu_slider,)


@app.cell
def _(
    predict_field, np, nu_slider,
    mean_ux, modes_ux, pipe_ux,
    mean_uy, modes_uy, pipe_uy,
    mean_p, modes_p, pipe_p,
):
    nu = nu_slider.value
    ux_pred = predict_field(np.array([nu]), mean_ux, modes_ux, pipe_ux)[0]
    uy_pred = predict_field(np.array([nu]), mean_uy, modes_uy, pipe_uy)[0]
    p_pred = predict_field(np.array([nu]), mean_p, modes_p, pipe_p)[0]
    return nu, ux_pred, uy_pred, p_pred


@app.cell
def _(mo, nu, np, ux_pred, uy_pred):
    speed = float(np.max(np.sqrt(ux_pred**2 + uy_pred**2)))
    mo.md(
        f"""
        ### ν = {nu:.4f} — max |u| = {speed:.4f}
        """
    )
    return


@app.cell
def _(plt, np, X, Y, n_grid, ux_pred, uy_pred, p_pred, nu):
    xi = X.reshape(n_grid, n_grid)
    yi = Y.reshape(n_grid, n_grid)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Prédiction surrogate POD — ν = {nu:.4f}", fontsize=14)

    im1 = ax1.contourf(xi, yi, ux_pred.reshape(n_grid, n_grid), levels=30)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    ax1.set_title("ux")
    ax1.set_aspect("equal")

    im2 = ax2.contourf(xi, yi, uy_pred.reshape(n_grid, n_grid), levels=30)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title("uy")
    ax2.set_aspect("equal")

    im3 = ax3.contourf(xi, yi, p_pred.reshape(n_grid, n_grid), levels=30)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.set_title("p")
    ax3.set_aspect("equal")

    plt.tight_layout()
    fig
    return


@app.cell
def _(plt, np, X, Y, n_grid, ux_pred, uy_pred, nu):
    xi_v = X.reshape(n_grid, n_grid)
    yi_v = Y.reshape(n_grid, n_grid)
    ux_g = ux_pred.reshape(n_grid, n_grid)
    uy_g = uy_pred.reshape(n_grid, n_grid)
    speed_g = np.sqrt(ux_g**2 + uy_g**2)

    fig_v, ax_v = plt.subplots(figsize=(6, 5))
    ax_v.contourf(xi_v, yi_v, speed_g, levels=30, cmap="viridis")

    skip = 3
    ax_v.quiver(
        xi_v[::skip, ::skip], yi_v[::skip, ::skip],
        ux_g[::skip, ::skip], uy_g[::skip, ::skip],
        color="white", alpha=0.7, scale=20,
    )
    ax_v.set_title(f"Champ de vitesse — ν = {nu:.4f}")
    ax_v.set_aspect("equal")
    plt.tight_layout()
    fig_v
    return


@app.cell
def _(
    plt, np, predict_field, nu_all, UX, UY,
    mean_ux, modes_ux, pipe_ux,
    mean_uy, modes_uy, pipe_uy,
):
    nu_sweep = np.logspace(np.log10(nu_all.min()), np.log10(nu_all.max()), 200)
    ux_sw = predict_field(nu_sweep, mean_ux, modes_ux, pipe_ux)
    uy_sw = predict_field(nu_sweep, mean_uy, modes_uy, pipe_uy)
    max_u_sweep = np.max(np.sqrt(ux_sw**2 + uy_sw**2), axis=1)
    max_u_data = np.max(np.sqrt(UX**2 + UY**2), axis=1)

    fig_s, ax_s = plt.subplots(figsize=(8, 4))
    ax_s.loglog(nu_all, max_u_data, "o", markersize=5, label="Simulations FreeFEM")
    ax_s.loglog(nu_sweep, max_u_sweep, "-", linewidth=2, label="Surrogate POD")
    ax_s.set_xlabel("ν")
    ax_s.set_ylabel("max |u|")
    ax_s.set_title("max |u| vs ν — surrogate vs simulations")
    ax_s.legend()
    ax_s.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig_s
    return


if __name__ == "__main__":
    app.run()
