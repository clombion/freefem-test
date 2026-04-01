# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "numpy>=1.26",
#     "scikit-learn>=1.4",
#     "scipy>=1.10",
#     "matplotlib>=3.8",
# ]
#
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline, Pipeline
    from scipy.linalg import LinAlgWarning
    return (
        LinAlgWarning, Pipeline, PolynomialFeatures, Ridge,
        make_pipeline, mo, np, plt, warnings,
    )


@app.cell(hide_code=True)
def _(mo, plt):
    _theme = mo.app_meta().theme
    if _theme == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    return


@app.cell(hide_code=True)
def _(LinAlgWarning, Pipeline, PolynomialFeatures, Ridge, make_pipeline, np, warnings):
    def compute_pod(snapshots, k=5):
        mean = snapshots.mean(axis=0)
        U_c = snapshots - mean
        _, s, Vt = np.linalg.svd(U_c, full_matrices=False)
        rank = Vt.shape[0]
        if k > rank:
            k = rank
        modes = Vt[:k]
        coeffs = U_c @ modes.T
        denom = float((s ** 2).sum())
        energy = float((s[:k] ** 2).sum() / denom) if denom > 0.0 else 0.0
        return mean, modes, coeffs, energy

    def fit_surrogate(nu_train, snapshots_train, k=5, degree=3):
        mean, modes, coeffs, energy = compute_pod(snapshots_train, k)
        pipe = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=0.01))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="An ill-conditioned matrix", category=LinAlgWarning)
            pipe.fit((1.0 / nu_train).reshape(-1, 1), coeffs)
        return mean, modes, pipe, energy

    def predict_field(nu_query, mean, modes, pipe):
        nu_arr = np.atleast_1d(nu_query)
        coeffs_pred = pipe.predict((1.0 / nu_arr).reshape(-1, 1))
        return coeffs_pred @ modes + mean
    return compute_pod, fit_surrogate, predict_field


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Surrogate POD — Stokes en cavité entraînée

    ## Le problème physique

    On simule un **écoulement de Stokes** dans un carré [0,1]×[0,1] :
    le couvercle supérieur se déplace à vitesse constante (u=1, v=0),
    toutes les autres parois sont fixes (condition de non-glissement).

    Les équations de Stokes (régime lent, sans inertie) s'écrivent :

    $$-\nu \, \Delta \mathbf{u} + \nabla p = 0, \quad \nabla \cdot \mathbf{u} = 0$$

    ## L'approche surrogate

    1. **Générer un dataset** : 40 simulations FreeFEM pour ν ∈ [0.005, 2.0]
    2. **Décomposer par POD** (SVD) — quelques modes capturent >99% de l'énergie
    3. **Régresser** : un polynôme en 1/ν prédit les coefficients POD

    Résultat : prédiction **instantanée** (~1 ms) au lieu de ~100 ms par simulation.
    """)
    return


@app.cell
async def _(mo, np):
    import sys

    if "pyodide" in sys.modules:
        from pyodide.http import pyfetch
        _url = str(mo.notebook_location()).rstrip("/") + "/public/dataset.npz"
        _resp = await pyfetch(_url)
        if _resp.status != 200:
            raise RuntimeError(f"Failed to fetch dataset: HTTP {_resp.status}")
        with open("/tmp/dataset.npz", "wb") as _f:
            _f.write(await _resp.bytes())
        data = np.load("/tmp/dataset.npz")
    else:
        data = np.load("public/dataset.npz")

    nu_all = data["nu_values"]
    X, Y = data["X"], data["Y"]
    UX, UY, P = data["UX"], data["UY"], data["P"]
    n_grid = int(round(len(X) ** 0.5))
    return nu_all, X, Y, UX, UY, P, n_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Paramètres du modèle

    Ajustez les paramètres du surrogate. **k** contrôle le nombre de modes POD
    retenus (plus = plus précis mais plus lent à entraîner). **Degré** contrôle
    la complexité du polynôme de régression.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(start=1, stop=20, value=5, step=1, label="k (modes POD)")
    deg_slider = mo.ui.slider(start=1, stop=6, value=3, step=1, label="Degré polynomial")
    mo.hstack([k_slider, deg_slider], gap=2)
    return k_slider, deg_slider


@app.cell
def _(P, UX, UY, deg_slider, fit_surrogate, k_slider, nu_all):
    k = k_slider.value
    degree = deg_slider.value
    mean_ux, modes_ux, pipe_ux, e_ux = fit_surrogate(nu_all, UX, k, degree)
    mean_uy, modes_uy, pipe_uy, e_uy = fit_surrogate(nu_all, UY, k, degree)
    mean_p, modes_p, pipe_p, e_p = fit_surrogate(nu_all, P, k, degree)
    return (
        k, degree,
        mean_ux, modes_ux, pipe_ux, e_ux,
        mean_uy, modes_uy, pipe_uy, e_uy,
        mean_p, modes_p, pipe_p, e_p,
    )


@app.cell(hide_code=True)
def _(e_p, e_ux, e_uy, k, degree, mo):
    mo.md(f"""
    **Modèle entraîné** — k={k} modes, degré {degree}

    | Champ | Énergie POD |
    |-------|------------|
    | ux | {e_ux * 100:.2f}% |
    | uy | {e_uy * 100:.2f}% |
    | p  | {e_p * 100:.2f}% |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    field_dropdown = mo.ui.dropdown(
        options={"Vitesse horizontale (ux)": "ux", "Vitesse verticale (uy)": "uy",
                 "Pression (p)": "p", "Magnitude vitesse (|u|)": "speed"},
        value="Magnitude vitesse (|u|)",
        label="Champ à afficher",
    )
    cmap_dropdown = mo.ui.dropdown(
        options=["viridis", "coolwarm", "plasma", "RdBu_r", "inferno", "cividis"],
        value="viridis",
        label="Colormap",
    )
    mo.hstack([field_dropdown, cmap_dropdown], gap=2)
    return field_dropdown, cmap_dropdown


@app.cell
def _(
    predict_field, np,
    mean_ux, modes_ux, pipe_ux,
    mean_uy, modes_uy, pipe_uy,
    mean_p, modes_p, pipe_p,
):
    nu = 0.1
    ux_pred = predict_field(np.array([nu]), mean_ux, modes_ux, pipe_ux)[0]
    uy_pred = predict_field(np.array([nu]), mean_uy, modes_uy, pipe_uy)[0]
    p_pred = predict_field(np.array([nu]), mean_p, modes_p, pipe_p)[0]
    speed_pred = np.sqrt(ux_pred**2 + uy_pred**2)
    return nu, ux_pred, uy_pred, p_pred, speed_pred


@app.cell(hide_code=True)
def _(X, Y, n_grid, nu, ux_pred, uy_pred, p_pred, speed_pred,
      field_dropdown, cmap_dropdown, np, plt, mo):
    xi = X.reshape(n_grid, n_grid)
    yi = Y.reshape(n_grid, n_grid)

    fields = {"ux": ux_pred, "uy": uy_pred, "p": p_pred, "speed": speed_pred}
    labels = {"ux": "ux (vitesse horizontale)", "uy": "uy (vitesse verticale)",
              "p": "p (pression)", "speed": "|u| (magnitude vitesse)"}
    field_key = field_dropdown.value
    field_data = fields[field_key].reshape(n_grid, n_grid)
    cmap = cmap_dropdown.value

    fig_main, axes_main = plt.subplots(1, 2, figsize=(13, 5))

    # Left: selected field contour
    im1 = axes_main[0].contourf(xi, yi, field_data, levels=30, cmap=cmap)
    plt.colorbar(im1, ax=axes_main[0], shrink=0.8)
    axes_main[0].set_title(f"{labels[field_key]} — ν={nu:.4f}")
    axes_main[0].set_xlabel("x")
    axes_main[0].set_ylabel("y")
    axes_main[0].set_aspect("equal")

    # Right: velocity vectors on speed background
    ux_g = ux_pred.reshape(n_grid, n_grid)
    uy_g = uy_pred.reshape(n_grid, n_grid)
    speed_g = np.sqrt(ux_g**2 + uy_g**2)
    im2 = axes_main[1].contourf(xi, yi, speed_g, levels=30, cmap="viridis")
    plt.colorbar(im2, ax=axes_main[1], shrink=0.8, label="|u|")
    skip = 3
    axes_main[1].quiver(
        xi[::skip, ::skip], yi[::skip, ::skip],
        ux_g[::skip, ::skip], uy_g[::skip, ::skip],
        color="white", alpha=0.7, scale=20,
    )
    axes_main[1].set_title(f"Champ de vitesse — ν={nu:.4f}")
    axes_main[1].set_xlabel("x")
    axes_main[1].set_ylabel("y")
    axes_main[1].set_aspect("equal")
    plt.tight_layout()

    mo.center(fig_main)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Profil de vitesse sur la ligne centrale

    Coupe verticale à **x = 0.5** : profil classique de la cavité entraînée.
    Le profil montre la transition entre la vitesse du couvercle (ux ≈ 1 en y=1)
    et la recirculation (ux < 0 dans la partie basse).

    C'est le diagnostic standard pour valider une simulation de cavité —
    comparez avec les résultats de référence de Ghia et al. (1982).
    """)
    return


@app.cell(hide_code=True)
def _(X, Y, n_grid, nu, ux_pred, uy_pred, np, plt, mo):
    xi_cl = X.reshape(n_grid, n_grid)
    yi_cl = Y.reshape(n_grid, n_grid)
    ux_cl = ux_pred.reshape(n_grid, n_grid)
    uy_cl = uy_pred.reshape(n_grid, n_grid)

    # Vertical centerline at x=0.5 (column index n_grid//2)
    mid_col = n_grid // 2
    y_line = yi_cl[:, mid_col]
    ux_line = ux_cl[:, mid_col]

    # Horizontal centerline at y=0.5 (row index n_grid//2)
    mid_row = n_grid // 2
    x_line = xi_cl[mid_row, :]
    uy_line = uy_cl[mid_row, :]

    fig_cl, (ax_cl1, ax_cl2) = plt.subplots(1, 2, figsize=(12, 4))

    ax_cl1.plot(ux_line, y_line, "b-", linewidth=2)
    ax_cl1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax_cl1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax_cl1.set_xlabel("ux")
    ax_cl1.set_ylabel("y")
    ax_cl1.set_title(f"ux le long de x=0.5 — ν={nu:.4f}")
    ax_cl1.grid(True, alpha=0.3)

    ax_cl2.plot(x_line, uy_line, "r-", linewidth=2)
    ax_cl2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_cl2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax_cl2.set_xlabel("x")
    ax_cl2.set_ylabel("uy")
    ax_cl2.set_title(f"uy le long de y=0.5 — ν={nu:.4f}")
    ax_cl2.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.center(fig_cl)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Spectre POD — énergie par mode

    Le spectre des valeurs singulières montre combien de modes sont nécessaires
    pour capturer l'essentiel de l'information. Pour Stokes (problème linéaire),
    l'énergie est concentrée dans très peu de modes — la courbe chute rapidement.
    La ligne rouge indique le nombre de modes **k** sélectionné ci-dessus.
    """)
    return


@app.cell(hide_code=True)
def _(UX, k, np, plt, mo):
    # Compute full spectrum (all modes)
    mean_full = UX.mean(axis=0)
    U_c = UX - mean_full
    _, s_full, _ = np.linalg.svd(U_c, full_matrices=False)
    energy_cumul = np.cumsum(s_full**2) / (s_full**2).sum()

    fig_spec, (ax_sp1, ax_sp2) = plt.subplots(1, 2, figsize=(12, 4))

    # Singular values
    ax_sp1.semilogy(range(1, len(s_full) + 1), s_full, "ko-", markersize=4)
    ax_sp1.axvline(x=k, color="red", linestyle="--", label=f"k={k}")
    ax_sp1.set_xlabel("Mode index")
    ax_sp1.set_ylabel("Valeur singulière σ")
    ax_sp1.set_title("Spectre des valeurs singulières (ux)")
    ax_sp1.legend()
    ax_sp1.grid(True, alpha=0.3)

    # Cumulative energy
    ax_sp2.plot(range(1, len(energy_cumul) + 1), energy_cumul * 100, "b-o", markersize=4)
    ax_sp2.axvline(x=k, color="red", linestyle="--", label=f"k={k}")
    ax_sp2.axhline(y=99.9, color="gray", linestyle=":", alpha=0.5, label="99.9%")
    ax_sp2.set_xlabel("Nombre de modes")
    ax_sp2.set_ylabel("Énergie cumulée (%)")
    ax_sp2.set_title("Énergie cumulée")
    ax_sp2.set_ylim([0, 101])
    ax_sp2.legend()
    ax_sp2.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.center(fig_spec)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparaison surrogate vs simulation la plus proche

    Le surrogate prédit pour n'importe quel ν, mais le dataset contient 40
    simulations exactes. On compare la prédiction du surrogate avec la simulation
    FreeFEM la plus proche en ν — l'erreur absolue montre la fidélité du modèle réduit.
    """)
    return


@app.cell(hide_code=True)
def _(X, Y, UX, UY, n_grid, nu, nu_all, np, ux_pred, uy_pred, plt, mo):
    # Find closest simulation in dataset
    closest_idx = int(np.argmin(np.abs(nu_all - nu)))
    nu_closest = nu_all[closest_idx]
    ux_true = UX[closest_idx]
    uy_true = UY[closest_idx]

    xi_c = X.reshape(n_grid, n_grid)
    yi_c = Y.reshape(n_grid, n_grid)

    speed_true = np.sqrt(ux_true**2 + uy_true**2)
    speed_surr = np.sqrt(ux_pred**2 + uy_pred**2)
    speed_err = np.abs(speed_surr - speed_true)

    fig_cmp, (ax_c1, ax_c2, ax_c3) = plt.subplots(1, 3, figsize=(15, 4))

    vmin = min(speed_true.min(), speed_surr.min())
    vmax = max(speed_true.max(), speed_surr.max())

    im_c1 = ax_c1.contourf(xi_c, yi_c, speed_surr.reshape(n_grid, n_grid),
                            levels=30, vmin=vmin, vmax=vmax)
    plt.colorbar(im_c1, ax=ax_c1, shrink=0.8)
    ax_c1.set_title(f"Surrogate |u| (ν={nu:.4f})")
    ax_c1.set_aspect("equal")

    im_c2 = ax_c2.contourf(xi_c, yi_c, speed_true.reshape(n_grid, n_grid),
                            levels=30, vmin=vmin, vmax=vmax)
    plt.colorbar(im_c2, ax=ax_c2, shrink=0.8)
    ax_c2.set_title(f"FreeFEM |u| (ν={nu_closest:.4f})")
    ax_c2.set_aspect("equal")

    im_c3 = ax_c3.contourf(xi_c, yi_c, speed_err.reshape(n_grid, n_grid),
                            levels=30, cmap="hot_r")
    plt.colorbar(im_c3, ax=ax_c3, shrink=0.8)
    ax_c3.set_title(f"|erreur| max={speed_err.max():.2e}")
    ax_c3.set_aspect("equal")

    plt.suptitle(f"Comparaison: surrogate (ν={nu:.4f}) vs simulation (ν={nu_closest:.4f})")
    plt.tight_layout()
    mo.center(fig_cmp)
    return


@app.cell(hide_code=True)
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


## ===================================================================
## Partie 2 : Navier-Stokes — le rôle de la viscosité
## ===================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Partie 2 : Navier-Stokes — le rôle de la viscosité

    En ajoutant le terme convectif $(\mathbf{u} \cdot \nabla)\mathbf{u}$, on passe
    aux **équations de Navier-Stokes** :

    $$(\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \, \Delta \mathbf{u} + \nabla p = 0, \quad \nabla \cdot \mathbf{u} = 0$$

    La non-linéarité fait que le champ de vitesse **dépend du nombre de Reynolds**
    Re = UL/ν. À bas Re (ν grand), on retrouve Stokes. À Re élevé (ν petit),
    l'inertie déforme l'écoulement : le vortex central migre, des recirculations
    secondaires apparaissent dans les coins.

    Le dataset contient 30 simulations FreeFEM (Picard) pour Re ∈ [1, 100].
    """)
    return


@app.cell
async def _(mo, np):
    import sys as _sys

    if "pyodide" in _sys.modules:
        from pyodide.http import pyfetch as _pyfetch
        _url_ns = str(mo.notebook_location()).rstrip("/") + "/public/dataset_ns.npz"
        _resp_ns = await _pyfetch(_url_ns)
        if _resp_ns.status != 200:
            raise RuntimeError(f"Failed to fetch N-S dataset: HTTP {_resp_ns.status}")
        with open("/tmp/dataset_ns.npz", "wb") as _f:
            _f.write(await _resp_ns.bytes())
        data_ns = np.load("/tmp/dataset_ns.npz")
    else:
        data_ns = np.load("public/dataset_ns.npz")

    nu_ns = data_ns["nu_values"]
    X_ns, Y_ns = data_ns["X"], data_ns["Y"]
    UX_ns, UY_ns, P_ns = data_ns["UX"], data_ns["UY"], data_ns["P"]
    n_grid_ns = int(round(len(X_ns) ** 0.5))
    return nu_ns, X_ns, Y_ns, UX_ns, UY_ns, P_ns, n_grid_ns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Paramètres du surrogate Navier-Stokes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    k_ns_slider = mo.ui.slider(start=1, stop=25, value=10, step=1, label="k (modes POD)")
    deg_ns_slider = mo.ui.slider(start=1, stop=6, value=3, step=1, label="Degré polynomial")
    mo.hstack([k_ns_slider, deg_ns_slider], gap=2)
    return k_ns_slider, deg_ns_slider


@app.cell
def _(UX_ns, UY_ns, P_ns, deg_ns_slider, fit_surrogate, k_ns_slider, nu_ns):
    k_ns = k_ns_slider.value
    degree_ns = deg_ns_slider.value
    mean_ux_ns, modes_ux_ns, pipe_ux_ns, e_ux_ns = fit_surrogate(nu_ns, UX_ns, k_ns, degree_ns)
    mean_uy_ns, modes_uy_ns, pipe_uy_ns, e_uy_ns = fit_surrogate(nu_ns, UY_ns, k_ns, degree_ns)
    mean_p_ns, modes_p_ns, pipe_p_ns, e_p_ns = fit_surrogate(nu_ns, P_ns, k_ns, degree_ns)
    return (
        k_ns, degree_ns,
        mean_ux_ns, modes_ux_ns, pipe_ux_ns, e_ux_ns,
        mean_uy_ns, modes_uy_ns, pipe_uy_ns, e_uy_ns,
        mean_p_ns, modes_p_ns, pipe_p_ns, e_p_ns,
    )


@app.cell(hide_code=True)
def _(e_p_ns, e_ux_ns, e_uy_ns, k_ns, degree_ns, mo):
    mo.md(f"""
    **Modèle N-S entraîné** — k={k_ns} modes, degré {degree_ns}

    | Champ | Énergie POD |
    |-------|------------|
    | ux | {e_ux_ns * 100:.2f}% |
    | uy | {e_uy_ns * 100:.2f}% |
    | p  | {e_p_ns * 100:.2f}% |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Exploration interactive — effet du Reynolds

    Contrairement à Stokes, ici **ν change réellement l'écoulement** :
    - **Re ~ 1** (ν = 1.0) : écoulement quasi-Stokes, vortex centré
    - **Re ~ 100** (ν = 0.01) : vortex décalé, recirculations dans les coins
    """)
    return


@app.cell(hide_code=True)
def _(mo, nu_ns):
    nu_ns_slider = mo.ui.slider(
        start=float(nu_ns.min()),
        stop=float(nu_ns.max()),
        value=0.1,
        step=0.001,
        label="ν (viscosité cinématique)",
        full_width=True,
    )
    field_ns_dropdown = mo.ui.dropdown(
        options={"Vitesse horizontale (ux)": "ux", "Vitesse verticale (uy)": "uy",
                 "Pression (p)": "p", "Magnitude vitesse (|u|)": "speed"},
        value="Magnitude vitesse (|u|)",
        label="Champ à afficher",
    )
    cmap_ns_dropdown = mo.ui.dropdown(
        options=["viridis", "coolwarm", "plasma", "RdBu_r", "inferno", "cividis"],
        value="viridis",
        label="Colormap",
    )
    mo.vstack([
        nu_ns_slider,
        mo.hstack([field_ns_dropdown, cmap_ns_dropdown], gap=2),
    ])
    return nu_ns_slider, field_ns_dropdown, cmap_ns_dropdown


@app.cell
def _(
    predict_field, np, nu_ns_slider,
    mean_ux_ns, modes_ux_ns, pipe_ux_ns,
    mean_uy_ns, modes_uy_ns, pipe_uy_ns,
    mean_p_ns, modes_p_ns, pipe_p_ns,
):
    nu_v = nu_ns_slider.value
    ux_ns_pred = predict_field(np.array([nu_v]), mean_ux_ns, modes_ux_ns, pipe_ux_ns)[0]
    uy_ns_pred = predict_field(np.array([nu_v]), mean_uy_ns, modes_uy_ns, pipe_uy_ns)[0]
    p_ns_pred = predict_field(np.array([nu_v]), mean_p_ns, modes_p_ns, pipe_p_ns)[0]
    speed_ns_pred = np.sqrt(ux_ns_pred**2 + uy_ns_pred**2)
    return nu_v, ux_ns_pred, uy_ns_pred, p_ns_pred, speed_ns_pred


@app.cell(hide_code=True)
def _(mo, nu_v, np, speed_ns_pred):
    _re = 1.0 / nu_v
    _max_speed = float(np.max(speed_ns_pred))
    mo.md(f"### ν = {nu_v:.4f} — Re = {_re:.1f} — max |u| = {_max_speed:.4f}")
    return


@app.cell(hide_code=True)
def _(X_ns, Y_ns, n_grid_ns, nu_v, ux_ns_pred, uy_ns_pred, p_ns_pred, speed_ns_pred,
      field_ns_dropdown, cmap_ns_dropdown, np, plt, mo):
    xi_ns = X_ns.reshape(n_grid_ns, n_grid_ns)
    yi_ns = Y_ns.reshape(n_grid_ns, n_grid_ns)

    fields_ns = {"ux": ux_ns_pred, "uy": uy_ns_pred, "p": p_ns_pred, "speed": speed_ns_pred}
    labels_ns = {"ux": "ux (vitesse horizontale)", "uy": "uy (vitesse verticale)",
                 "p": "p (pression)", "speed": "|u| (magnitude vitesse)"}
    fk_ns = field_ns_dropdown.value
    fd_ns = fields_ns[fk_ns].reshape(n_grid_ns, n_grid_ns)
    cm_ns = cmap_ns_dropdown.value

    fig_ns, axes_ns = plt.subplots(1, 2, figsize=(13, 5))

    _re_val = 1.0 / nu_v
    im_ns1 = axes_ns[0].contourf(xi_ns, yi_ns, fd_ns, levels=30, cmap=cm_ns)
    plt.colorbar(im_ns1, ax=axes_ns[0], shrink=0.8)
    axes_ns[0].set_title(f"{labels_ns[fk_ns]} — Re={_re_val:.1f}")
    axes_ns[0].set_xlabel("x")
    axes_ns[0].set_ylabel("y")
    axes_ns[0].set_aspect("equal")

    ux_g_ns = ux_ns_pred.reshape(n_grid_ns, n_grid_ns)
    uy_g_ns = uy_ns_pred.reshape(n_grid_ns, n_grid_ns)
    speed_g_ns = np.sqrt(ux_g_ns**2 + uy_g_ns**2)
    im_ns2 = axes_ns[1].contourf(xi_ns, yi_ns, speed_g_ns, levels=30, cmap="viridis")
    plt.colorbar(im_ns2, ax=axes_ns[1], shrink=0.8, label="|u|")
    _skip = 3
    axes_ns[1].quiver(
        xi_ns[::_skip, ::_skip], yi_ns[::_skip, ::_skip],
        ux_g_ns[::_skip, ::_skip], uy_g_ns[::_skip, ::_skip],
        color="white", alpha=0.7, scale=20,
    )
    axes_ns[1].set_title(f"Champ de vitesse — Re={_re_val:.1f}")
    axes_ns[1].set_xlabel("x")
    axes_ns[1].set_ylabel("y")
    axes_ns[1].set_aspect("equal")
    plt.tight_layout()

    mo.center(fig_ns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Spectre POD — Navier-Stokes vs Stokes

    Le spectre N-S décroît **plus lentement** que celui de Stokes : la non-linéarité
    enrichit la dynamique et requiert plus de modes pour capturer l'énergie.
    """)
    return


@app.cell(hide_code=True)
def _(UX_ns, k_ns, np, plt, mo):
    _mean_ns = UX_ns.mean(axis=0)
    _Uc_ns = UX_ns - _mean_ns
    _, _s_ns, _ = np.linalg.svd(_Uc_ns, full_matrices=False)
    _ecum_ns = np.cumsum(_s_ns**2) / (_s_ns**2).sum()

    fig_spec_ns, (ax_sp_ns1, ax_sp_ns2) = plt.subplots(1, 2, figsize=(12, 4))

    ax_sp_ns1.semilogy(range(1, len(_s_ns) + 1), _s_ns, "ko-", markersize=4)
    ax_sp_ns1.axvline(x=k_ns, color="red", linestyle="--", label=f"k={k_ns}")
    ax_sp_ns1.set_xlabel("Mode index")
    ax_sp_ns1.set_ylabel("Valeur singulière σ")
    ax_sp_ns1.set_title("Spectre N-S (ux)")
    ax_sp_ns1.legend()
    ax_sp_ns1.grid(True, alpha=0.3)

    ax_sp_ns2.plot(range(1, len(_ecum_ns) + 1), _ecum_ns * 100, "b-o", markersize=4)
    ax_sp_ns2.axvline(x=k_ns, color="red", linestyle="--", label=f"k={k_ns}")
    ax_sp_ns2.axhline(y=99.9, color="gray", linestyle=":", alpha=0.5, label="99.9%")
    ax_sp_ns2.set_xlabel("Nombre de modes")
    ax_sp_ns2.set_ylabel("Énergie cumulée (%)")
    ax_sp_ns2.set_title("Énergie cumulée N-S")
    ax_sp_ns2.set_ylim([0, 101])
    ax_sp_ns2.legend()
    ax_sp_ns2.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.center(fig_spec_ns)
    return


if __name__ == "__main__":
    app.run()
