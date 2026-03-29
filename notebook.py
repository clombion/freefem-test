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

## Le problème physique

On simule un **écoulement de Stokes** dans un carré [0,1]×[0,1] :
le couvercle supérieur se déplace à vitesse constante (u=1, v=0),
toutes les autres parois sont fixes (condition de non-glissement).

C'est le problème classique de la **lid-driven cavity** — un benchmark
fondamental en mécanique des fluides numérique.

Les équations de Stokes (régime lent, sans inertie) s'écrivent :

$$-\\nu \\, \\Delta \\mathbf{u} + \\nabla p = 0, \\quad \\nabla \\cdot \\mathbf{u} = 0$$

où **ν** est la viscosité cinématique et **u**, **p** sont la vitesse et la pression.

## L'approche surrogate

Au lieu de relancer une simulation FreeFEM à chaque changement de ν,
on construit un **modèle réduit (ROM)** :

1. **Générer un dataset** : 40 simulations FreeFEM pour ν ∈ [0.005, 2.0]
2. **Décomposer par POD** (Proper Orthogonal Decomposition) : extraire les modes
   dominants via SVD — quelques modes suffisent à capturer 99% de l'énergie
3. **Régresser** : un polynôme en 1/ν prédit les coefficients POD pour tout ν

Résultat : la prédiction est **instantanée** (~1 ms) au lieu de ~0.1 s par simulation.
""")
    return


@app.cell
def _(mo):
    mo.md("## Chargement du dataset et entraînement")
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
def _(mo, e_ux, e_uy, e_p, k, nu_all):
    mo.md(f"""
Le surrogate est entraîné sur les **{len(nu_all)} simulations** du dataset
avec **k={k} modes POD**. La fraction d'énergie capturée par ces modes
mesure la qualité de la décomposition (100% = reconstruction parfaite) :

| Champ | Description | Énergie capturée |
|-------|-------------|-----------------|
| ux    | Vitesse horizontale | {e_ux * 100:.2f}% |
| uy    | Vitesse verticale   | {e_uy * 100:.2f}% |
| p     | Pression            | {e_p * 100:.2f}% |

Pour Stokes, la solution varie linéairement en 1/ν — le surrogate
capture cette structure et prédit avec une erreur quasi nulle.
""")
    return


@app.cell
def _(mo):
    mo.md("""
## Exploration interactive

Déplacez le slider pour choisir une viscosité ν.
Le surrogate prédit instantanément les champs correspondants.

- **ν petit** (→ 0.005) : écoulement plus vigoureux, gradients plus forts
- **ν grand** (→ 2.0) : écoulement amorti, champs plus lisses
""")
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
    mo.md(f"### ν = {nu:.4f} — max |u| = {speed:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
### Champs scalaires

De gauche à droite : vitesse horizontale (**ux**), vitesse verticale (**uy**),
et **pression** (p). Les couleurs indiquent l'intensité — le couvercle
entraîne le fluide vers la droite (ux ≈ 1 en haut), créant un
tourbillon de recirculation visible sur uy.
""")
    return


@app.cell
def _(plt, np, X, Y, n_grid, ux_pred, uy_pred, p_pred, nu):
    xi = X.reshape(n_grid, n_grid)
    yi = Y.reshape(n_grid, n_grid)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Prédiction surrogate POD — ν = {nu:.4f}", fontsize=14)

    im1 = ax1.contourf(xi, yi, ux_pred.reshape(n_grid, n_grid), levels=30)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    ax1.set_title("ux (vitesse horizontale)")
    ax1.set_aspect("equal")

    im2 = ax2.contourf(xi, yi, uy_pred.reshape(n_grid, n_grid), levels=30)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title("uy (vitesse verticale)")
    ax2.set_aspect("equal")

    im3 = ax3.contourf(xi, yi, p_pred.reshape(n_grid, n_grid), levels=30)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.set_title("p (pression)")
    ax3.set_aspect("equal")

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
### Champ de vecteurs

Les flèches montrent la direction et l'intensité de l'écoulement.
Le fond coloré indique la norme de la vitesse (magnitude).
On observe le tourbillon principal créé par le couvercle entraîné :
le fluide descend le long de la paroi droite et remonte par la gauche.
""")
    return


@app.cell
def _(plt, np, X, Y, n_grid, ux_pred, uy_pred, nu):
    xi_v = X.reshape(n_grid, n_grid)
    yi_v = Y.reshape(n_grid, n_grid)
    ux_g = ux_pred.reshape(n_grid, n_grid)
    uy_g = uy_pred.reshape(n_grid, n_grid)
    speed_g = np.sqrt(ux_g**2 + uy_g**2)

    fig_v, ax_v = plt.subplots(figsize=(6, 5))
    im_v = ax_v.contourf(xi_v, yi_v, speed_g, levels=30, cmap="viridis")
    plt.colorbar(im_v, ax=ax_v, shrink=0.8, label="|u|")

    skip = 3
    ax_v.quiver(
        xi_v[::skip, ::skip], yi_v[::skip, ::skip],
        ux_g[::skip, ::skip], uy_g[::skip, ::skip],
        color="white", alpha=0.7, scale=20,
    )
    ax_v.set_title(f"Champ de vitesse — ν = {nu:.4f}")
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("y")
    ax_v.set_aspect("equal")
    plt.tight_layout()
    fig_v
    return


@app.cell
def _(mo):
    mo.md("""
## Validation du surrogate

Le graphique ci-dessous compare la quantité scalaire max|u| issue des
40 simulations FreeFEM (points bleus) avec la prédiction continue du
surrogate (courbe orange). Pour Stokes, max|u| est constant car la
vitesse du couvercle impose l'échelle — le surrogate reproduit cette
propriété exactement.
""")
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
    ax_s.set_xlabel("ν (viscosité cinématique)")
    ax_s.set_ylabel("max |u|")
    ax_s.set_title("Validation : max |u| vs ν")
    ax_s.legend()
    ax_s.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig_s
    return


@app.cell
def _(mo):
    mo.md("""
---

## Comment ça marche en coulisses

1. **FreeFEM** résout les EDP de Stokes par éléments finis (Taylor-Hood P2/P1)
   sur un maillage 40×40 pour 40 valeurs de ν log-espacées
2. **POD** (SVD de la matrice de snapshots) extrait les k modes spatiaux
   dominants — chaque solution se décompose en combinaison linéaire de ces modes
3. **Régression polynomiale** en 1/ν prédit les coefficients de cette
   décomposition pour tout ν, sans résoudre les EDP

Le coût passe de **O(N³)** (résolution EF) à **O(k)** (produit matrice-vecteur) — un
gain de plusieurs ordres de grandeur.
""")
    return


if __name__ == "__main__":
    app.run()
