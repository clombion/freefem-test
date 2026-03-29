# Stokes Cavity Surrogate

Modele reduit (ROM) pour l'ecoulement de Stokes en cavite entrainee.
Combine des simulations FreeFEM avec une decomposition POD et une
regression polynomiale pour predire instantanement les champs de
vitesse et de pression a partir d'un parametre de viscosite.

## Le probleme

Simuler un ecoulement de Stokes pour une nouvelle viscosite prend ~0.1 s
avec FreeFEM. C'est acceptable pour un calcul unique, mais trop lent pour
une exploration interactive ou un balayage parametrique de centaines de
valeurs.

## La solution

1. **Generer** 40 simulations FreeFEM en faisant varier la viscosite
   cinematique nu dans [0.005, 2.0]
2. **Decomposer** les solutions par POD (SVD de la matrice de snapshots)
   -- quelques modes capturent >99% de l'energie
3. **Regresser** les coefficients POD en fonction de 1/nu avec un polynome
   de degre 3

Resultat : la prediction d'un champ complet (2601 points) prend ~1 ms
au lieu de ~100 ms -- un gain de 100x.

## Pipeline

```
stokes_cavity.edp          FreeFEM: resout Stokes pour un nu donne
       |
generate_dataset.py        40 simulations → data/dataset.npz
       |
train_surrogate.py         POD + regression → metriques + figures
       |
notebook.py                Notebook marimo interactif (slider nu)
```

## Prerequis

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (gestionnaire de paquets)
- **FreeFEM++** -- [telecharger depuis GitHub](https://github.com/FreeFem/FreeFem-sources/releases)
  (macOS Apple Silicon : prendre le `.dmg` correspondant a votre version de macOS)
- **GCC** (fournit `libgfortran` requise par FreeFEM) :
  ```bash
  brew install gcc
  ```

## Installation

```bash
git clone <repo-url>
cd test-claude-nora
uv sync          # installe les dependances Python
```

Verifier que FreeFEM fonctionne :

```bash
mkdir -p data/snapshots
/Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++ -nw stokes_cavity.edp -nu 0.1
```

Sortie attendue (parmi d'autres lignes) :

```
nu=0.1  ||div u||^2=2.75449
Exported: data/snapshots/fields_nu_0.1.csv
```

> **Note :** si FreeFEM est installe ailleurs, modifier `FREEFEM_CMD` dans
> `generate_dataset.py` (ligne 18).

## Demarrage rapide

```bash
# 1. Generer le dataset (~1 min pour 40 simulations)
uv run python generate_dataset.py

# 2. Entrainer le surrogate et produire les figures
MPLBACKEND=Agg uv run python train_surrogate.py --k 5 --deg 3

# 3. Lancer le notebook interactif
uv run marimo run notebook.py
```

Le notebook s'ouvre dans le navigateur. Deplacez le slider pour explorer
l'ecoulement a differentes viscosites.

## Tests

```bash
uv run pytest tests/ -v
```

11 tests couvrent le parsing CSV, la decomposition POD, la regression,
et les metriques d'erreur. Aucun test ne requiert FreeFEM.

## Documentation

- **[Guide utilisateur](docs/guide.md)** -- fonctionnement detaille, physique,
  architecture, troubleshooting
- **[Contributing](CONTRIBUTING.md)** -- environnement de dev, conventions, tests

## Licence

Ce projet est un prototype/PoC.
