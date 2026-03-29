# Guide utilisateur

## Vue d'ensemble

Ce projet construit un **modele reduit** (ROM -- Reduced Order Model)
pour l'ecoulement de Stokes dans une cavite carree entrainee.
Le pipeline passe par quatre etapes :

1. **Simulation** -- FreeFEM resout les equations de Stokes par elements
   finis pour 40 valeurs de viscosite
2. **Export** -- chaque solution est evaluee sur une grille reguliere
   51x51 et sauvee en CSV
3. **Apprentissage** -- decomposition POD (SVD) puis regression
   polynomiale des coefficients en fonction de 1/nu
4. **Exploration** -- notebook marimo avec slider pour predire les
   champs instantanement

## Comprendre le probleme physique

### Les equations de Stokes

L'ecoulement de Stokes decrit un fluide visqueux en regime lent
(nombre de Reynolds negligeable). Les equations sont :

```
-nu * Laplacien(u) + grad(p) = 0    (equilibre forces visqueuses / pression)
div(u) = 0                           (incompressibilite)
```

- **u** = (ux, uy) : champ de vitesse (2 composantes en 2D)
- **p** : pression
- **nu** : viscosite cinematique (le parametre qu'on fait varier)

### La cavite entrainee (lid-driven cavity)

Le domaine est un carre [0,1] x [0,1] :

```
      u = (1, 0)      <-- couvercle en mouvement
    +-----------+
    |           |
u=0 |   fluide  | u=0
    |           |
    +-----------+
      u = (0, 0)       <-- paroi fixe
```

Le couvercle superieur se deplace a vitesse constante vers la droite.
Les trois autres parois sont fixes. Cette configuration cree un
tourbillon de recirculation : le fluide descend le long de la paroi
droite et remonte par la gauche.

### Effet de la viscosite

| nu petit (ex: 0.005) | nu grand (ex: 2.0) |
|---|---|
| Gradients de vitesse forts | Ecoulement tres amorti |
| Recirculation intense | Recirculation faible |
| Couche limite mince pres des parois | Profils lisses |

Pour Stokes (lineaire), la vitesse est exactement proportionnelle
a 1/nu : `u(nu) = u_ref / nu`. C'est cette structure que le
surrogate exploite.

## Generer le dataset

### Prerequis

FreeFEM++ doit etre installe. Voir la section
[Installation de FreeFEM sur macOS](../README.md#installation-de-freefem-sur-macos)
dans le README pour les instructions detaillees (telechargement,
autorisation Gatekeeper, installation GCC).

### Configuration du chemin FreeFEM

Le chemin du binaire est dans `generate_dataset.py` ligne 18 :

```python
FREEFEM_CMD = "/Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++"
```

Adapter si votre installation differe.

### Lancer la generation

```bash
mkdir -p data/snapshots
uv run python generate_dataset.py
```

Le script lance 40 simulations en serie (~1 min au total).
Chaque simulation produit un CSV de 2602 lignes (1 header + 51x51 = 2601
points de grille) avec les colonnes `x, y, ux, uy, p`.

Sortie attendue :

```
[1/40] nu = 0.00500
[2/40] nu = 0.00583
...
[40/40] nu = 2.00000

Dataset sauve : data/dataset.npz
  nu_values shape : (40,)
  UX shape        : (40, 2601)
  UY shape        : (40, 2601)
  P  shape        : (40, 2601)
```

### Parametres de generation

| Parametre | Valeur | Role |
|-----------|--------|------|
| `N_SIM` | 40 | Nombre de simulations |
| `NU_MIN` | 0.005 | Borne inferieure (log-espace) |
| `NU_MAX` | 2.0 | Borne superieure |
| Maillage | 40x40 | Defini dans `stokes_cavity.edp` (variable `Nn`) |
| Grille export | 51x51 | Defini dans `stokes_cavity.edp` (variable `Nout`) |

## Entrainer le surrogate

```bash
MPLBACKEND=Agg uv run python train_surrogate.py --k 5 --deg 3
```

> `MPLBACKEND=Agg` empeche matplotlib d'ouvrir une fenetre GUI
> (utile en mode headless ou SSH). Retirer pour affichage interactif.

### Arguments CLI

| Argument | Defaut | Description |
|----------|--------|-------------|
| `--data` | `data/dataset.npz` | Chemin du dataset |
| `--k` | 5 | Nombre de modes POD a retenir |
| `--deg` | 3 | Degre du polynome de regression |

### Sortie attendue

```
Dataset : 40 simulations, 2601 points/grille
Train : 35 | Test : 5

Energie POD capturee (k=5 modes):
  ux : 89.59%
  uy : 88.41%
  p  : 97.86%

Erreur L2 relative sur test:
  ux : 0.0000%
  uy : 0.0000%
  p  : 0.0000%

Figure sauvee : data/comparison_fields.png
Figure sauvee : data/scalar_comparison.png
```

L'erreur L2 a 0.0000% est attendue : pour Stokes, la regression en
1/nu capture la relation lineaire exactement.

### Interpreration des resultats

**Energie POD** : pourcentage de la variance totale capturee par les
k modes retenus. Avec k=5, on capture ~90% pour la vitesse et ~98%
pour la pression. Avec k=1, on capture deja >99.9% grace a la
structure lineaire de Stokes.

**Erreur L2 relative** : `||predit - vrai||_2 / ||vrai||_2`,
moyennee sur les 5 points de test. Une erreur < 1% indique un
surrogate de haute fidelite.

### Figures produites

- `data/comparison_fields.png` : pour un nu de test, comparaison
  predit vs simule vs erreur absolue pour les 3 champs (ux, uy, p)
- `data/scalar_comparison.png` : max|u| vs nu -- le surrogate
  (courbe) superpose les simulations (points)

## Explorer avec le notebook marimo

```bash
uvx marimo run notebook.py
```

Le notebook s'ouvre dans le navigateur (par defaut `http://localhost:2718`).

### Ce que contient le notebook

**Controles interactifs :**

- **Slider nu** : viscosite cinematique dans [0.005, 2.0]
- **Slider k** : nombre de modes POD (1 a 20) — re-entraine le surrogate en temps reel
- **Slider degre** : degre du polynome de regression (1 a 6)
- **Dropdown champ** : choisir ux, uy, p, ou |u| (magnitude)
- **Dropdown colormap** : viridis, coolwarm, plasma, RdBu_r, inferno, cividis

**Visualisations :**

1. **Introduction** : equations de Stokes, schema de la cavite
2. **Tableau d'energie POD** : se met a jour avec k et degre
3. **Champ scalaire + vecteurs** : cote a cote, le champ choisi et les
   fleches de vitesse sur fond de magnitude
4. **Profils ligne centrale** : ux(y) a x=0.5 et uy(x) a y=0.5 — le
   diagnostic classique de validation (comparable a Ghia et al. 1982)
5. **Spectre POD** : valeurs singulieres et energie cumulee avec
   marqueur rouge au k selectionne
6. **Comparaison surrogate vs simulation** : triptyque predit / FreeFEM
   (simulation la plus proche en nu) / erreur absolue
7. **Validation globale** : courbe max|u| vs nu, surrogate vs simulations

Tous les graphiques se mettent a jour instantanement (~1 ms de calcul)
quand un slider ou un dropdown change.

### Mode edition

Pour modifier le notebook :

```bash
uvx marimo edit notebook.py
```

## Comment ca marche : la methode POD

### Etape 1 : matrice de snapshots

On empile les N solutions (chacune un vecteur de Ngrid valeurs) :

```
U = | u(nu_1)^T |     shape (N, Ngrid)
    | u(nu_2)^T |     ex: (40, 2601) pour ux
    | ...        |
    | u(nu_N)^T |
```

### Etape 2 : centrage et SVD

```
U_c = U - mean(U)          # centrage
U_c = W * Sigma * V^T      # SVD (numpy.linalg.svd)
```

- `V^T[:k]` = les k premiers **modes POD** (formes spatiales dominantes)
- `Sigma[:k]` = les k valeurs singulieres (importances relatives)
- Energie capturee = sum(sigma_i^2 pour i <= k) / sum(sigma_i^2)

### Etape 3 : coefficients et regression

Chaque snapshot se decompose en :

```
u(nu) ≈ mean + alpha_1(nu) * mode_1 + ... + alpha_k(nu) * mode_k
```

Les coefficients alpha_i(nu) sont calcules par projection :
`alpha = U_c @ modes.T`. Puis on entraine un polynome de degre d :

```
alpha_i(nu) ≈ Polynome(1/nu, degre=d)
```

Le choix de 1/nu comme variable est motive par la physique :
pour Stokes, alpha est proportionnel a 1/nu.

### Etape 4 : prediction

Pour un nouveau nu :

1. Evaluer les coefficients : `alpha_pred = poly(1/nu)`
2. Reconstruire le champ : `u_pred = alpha_pred @ modes + mean`

Cout : O(k * Ngrid) -- un produit matrice-vecteur au lieu de
resoudre un systeme lineaire creux.

## Reference

### Fichiers produits par le pipeline

| Fichier | Produit par | Contenu |
|---------|-------------|---------|
| `data/snapshots/fields_nu_*.csv` | `generate_dataset.py` | 40 CSVs, 2602 lignes chacun (header + 51x51 grille) |
| `data/dataset.npz` | `generate_dataset.py` | Archive numpy : nu_values, X, Y, UX, UY, P |
| `data/comparison_fields.png` | `train_surrogate.py` | Grille 3x3 : predit / simule / erreur pour ux, uy, p |
| `data/scalar_comparison.png` | `train_surrogate.py` | max\|u\| et pression moyenne vs nu |

### Format du CSV FreeFEM

```
x,y,ux,uy,p
0,0,-1.73611e-25,-1.73611e-25,4.16667e+07
0.02,0,4.73318e-20,1.34715e-20,4.16667e+07
...
```

- 2602 lignes (1 header + 51*51 = 2601 points)
- Grille reguliere [0, 1] x [0, 1], pas = 1/50
- Ordre : j (y) en boucle externe, i (x) en boucle interne

### Format du dataset.npz

```python
data = np.load("data/dataset.npz")
data["nu_values"]   # (40,)    valeurs de nu log-espacees
data["X"]           # (2601,)  coordonnees x de la grille
data["Y"]           # (2601,)  coordonnees y de la grille
data["UX"]          # (40, 2601)  vitesse horizontale
data["UY"]          # (40, 2601)  vitesse verticale
data["P"]           # (40, 2601)  pression
```

### Fonctions exportables de train_surrogate.py

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `compute_pod` | `(snapshots, k=5) -> (mean, modes, coeffs, energy)` | Decomposition POD par SVD |
| `fit_surrogate` | `(nu_train, snapshots, k=5, degree=3) -> (mean, modes, pipe, energy)` | Entraine POD + regression sur 1/nu |
| `predict_field` | `(nu_query, mean, modes, pipe) -> array` | Predit le champ pour un ou plusieurs nu |
| `relative_l2_error` | `(pred, true) -> float` | Erreur L2 relative moyennee |

### Limites et avertissements

| Limite | Detail |
|--------|--------|
| Plage de validite | Le surrogate interpole dans [0.005, 2.0] -- ne pas extrapoler au-dela |
| Modele physique | Stokes uniquement (lineaire). Pour Navier-Stokes (Re > 1), le surrogate devra etre reentrainee avec plus de modes |
| Pression | La pression est definie a une constante pres (regularisation 1e-10 dans FreeFEM). Les valeurs absolues de p n'ont pas de sens physique, seuls les gradients comptent |
| Maillage fixe | Le surrogate est lie a la grille 51x51. Changer la resolution necessite de regenerer tout le dataset |

## Troubleshooting

### FreeFEM++ : `dyld: Library not loaded: libgfortran`

FreeFEM a besoin de GCC. Installer avec :

```bash
brew install gcc
```

### FreeFEM++ : `getARGV does not exist`

Le fichier `stokes_cavity.edp` doit inclure `getARGV.idp` (ligne 10) :

```
include "getARGV.idp"
```

Cette ligne est deja presente dans le code. Si l'erreur persiste,
verifier que FreeFEM trouve son repertoire `idp/` :

```bash
find /Applications/FreeFem++.app -name "getARGV.idp"
```

### `FileNotFoundError: CSV annonce mais absent`

Le parseur de `run_simulation` lit la ligne "Exported: ..." de la
sortie FreeFEM mais le fichier n'existe pas. Causes possibles :

- Le repertoire `data/snapshots/` n'existe pas : `mkdir -p data/snapshots`
- FreeFEM a echoue silencieusement (verifier stderr)

### `FileNotFoundError: CSV tronque ... lignes, attendu 2602`

FreeFEM a ecrit un fichier incomplet. Peut arriver si le solveur
diverge pour un nu extreme. Verifier la sortie FreeFEM dans stderr
et ajuster la plage de nu si necessaire.

### LinAlgWarning dans Ridge

Le warning `ill-conditioned matrix` apparait quand la matrice de
features polynomiales est mal conditionnee. Augmenter `alpha` dans
`make_regression_pipe` (defaut : 1e-3) ou reduire le degre polynomial.

### plt.show() bloque en mode headless

Utiliser `MPLBACKEND=Agg` pour desactiver l'affichage GUI :

```bash
MPLBACKEND=Agg python train_surrogate.py
```
