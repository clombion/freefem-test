# Contribuer

## Prerequis

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets et environnements)
- FreeFEM++ (uniquement pour regenerer le dataset)
- GCC (`brew install gcc` sur macOS, fournit `libgfortran`)

## Setup

```bash
git clone <repo-url>
cd test-claude-nora
uv sync
```

## Lancer les tests

```bash
uv run pytest tests/ -v
```

17 tests, tous en Python pur (pas besoin de FreeFEM) :

```
tests/test_generate.py    5 tests   load_csv, input vide, subprocess erreur
tests/test_surrogate.py  12 tests   POD, regression, erreurs L2, guards input, edge cases
```

Attendu : `17 passed` en moins de 2 secondes.

## Structure du projet

```
stokes_cavity.edp       Script FreeFEM (Stokes P2/P1, export CSV)
generate_dataset.py     Orchestrateur : 40 appels FreeFEM → dataset.npz
train_surrogate.py      POD + regression polynomiale + CLI + plots
notebook.py             Notebook marimo interactif (sliders k/degre/nu,
                        dropdown champ/colormap, profils ligne centrale,
                        spectre POD, comparaison surrogate vs simulation)
tests/
  test_generate.py      Tests pour load_csv, input vide, subprocess erreur
  test_surrogate.py     Tests pour compute_pod, fit_surrogate, predict_field,
                        guards input (nu=0, k>rank), edge cases (zero-norm)
data/
  snapshots/            CSVs FreeFEM (generes, non versiones)
  dataset.npz           Archive numpy consolidee (genere, non versionne)
requirements.txt        Dependances Python
```

## Conventions

- **Commits** : prefixe `feat:`, `fix:`, ou `docs:` suivi d'une description courte
- **Tests** : TDD -- ecrire le test d'abord quand possible
- **FreeFEM** : le chemin du binaire est dans `FREEFEM_CMD` de `generate_dataset.py`
  -- l'adapter si l'installation differe
- **Donnees** : `data/` est dans `.gitignore` -- ne pas committer les CSV
  ou le `.npz`

## Regenerer le dataset

Si vous modifiez `stokes_cavity.edp` (maillage, conditions limites, etc.) :

```bash
rm -rf data/snapshots/*.csv data/dataset.npz
mkdir -p data/snapshots
uv run python generate_dataset.py
```

Puis relancer le surrogate :

```bash
MPLBACKEND=Agg uv run python train_surrogate.py
```

## Ajouter un champ a exporter

1. Modifier `stokes_cavity.edp` : ajouter la colonne dans la boucle `ofstream`
2. Modifier `load_csv` dans `generate_dataset.py` : ajuster les indices de colonnes
3. Mettre a jour `test_load_csv_parses_columns` dans `tests/test_generate.py`
4. Ajouter le surrogate correspondant dans `train_surrogate.py` (`fit_surrogate` + `predict_field`)
5. Regenerer le dataset
