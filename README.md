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
- **FreeFEM++** -- voir [Installation de FreeFEM sur macOS](#installation-de-freefem-sur-macos) ci-dessous
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

## Installation de FreeFEM sur macOS

FreeFEM n'est pas sur Homebrew. L'installation sur macOS requiert
quelques etapes manuelles pour contourner la verification Gatekeeper.

### 1. Telecharger

Aller sur [FreeFEM Releases](https://github.com/FreeFem/FreeFem-sources/releases)
et telecharger le `.dmg` **Apple Silicon** correspondant a votre version
de macOS (ex: `FreeFEM-v4.15-Apple-Silicon-15.4.dmg` pour macOS 15 Sequoia).

### 2. Installer

1. Ouvrir le `.dmg` telecharge
2. Glisser l'icone FreeFem++ dans `/Applications`

### 3. Autoriser l'application (Gatekeeper)

FreeFEM n'est pas signe par Apple, donc macOS le bloque par defaut.

```bash
sudo xattr -rc /Applications/FreeFem++.app
```

Cela demande votre mot de passe administrateur.

### 4. Premiere ouverture

1. Dans `/Applications`, **clic droit** sur FreeFem++.app
2. Maintenir la touche **Option** enfoncee et cliquer sur **Ouvrir**
3. macOS affiche un message "l'application est corrompue" -- c'est normal
4. Aller dans **Reglages Systeme > Confidentialite et securite**
5. Scroller tout en bas -- un message indique que FreeFem++ a ete bloque
6. Cliquer sur **Ouvrir quand meme**
7. Entrer votre mot de passe quand demande

Apres cette procedure, FreeFEM fonctionne normalement depuis le terminal.

### 5. Installer GCC

FreeFEM depend de `libgfortran` fournie par GCC :

```bash
brew install gcc
```

### 6. Verifier

```bash
/Applications/FreeFem++.app/Contents/ff-4.15.1/bin/FreeFem++ -nw -v 0 -ne stokes_cavity.edp -nu 0.1
```

Si la commande se termine sans erreur et produit un fichier
`data/snapshots/fields_nu_0.1.csv`, l'installation est reussie.

## Licence

Ce projet est un prototype/PoC.
