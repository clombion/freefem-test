# tests/test_generate.py
import numpy as np
import pytest
import os
import tempfile
from pathlib import Path


def test_load_csv_parses_columns():
    """load_csv lit correctement les 5 colonnes x, y, ux, uy, p."""
    from generate_dataset import load_csv

    content = "x,y,ux,uy,p\n0.0,0.0,0.1,0.2,0.5\n0.5,0.5,0.3,0.4,0.1\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        X, Y, UX, UY, P = load_csv(tmp)
        assert len(X) == 2
        assert X[0] == pytest.approx(0.0)
        assert Y[1] == pytest.approx(0.5)
        assert UX[0] == pytest.approx(0.1)
        assert UY[1] == pytest.approx(0.4)
        assert P[0] == pytest.approx(0.5)
    finally:
        os.unlink(tmp)


def test_load_csv_grid_size_2601():
    """Un export FreeFEM 50×50 produit 2601 points (de 0/50 à 50/50 inclus)."""
    from generate_dataset import load_csv

    rows = [
        f"{i/50},{j/50},0.1,0.0,0.5"
        for j in range(51)
        for i in range(51)
    ]
    content = "x,y,ux,uy,p\n" + "\n".join(rows) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        X, Y, UX, UY, P = load_csv(tmp)
        assert len(X) == 2601
        assert len(UX) == 2601
    finally:
        os.unlink(tmp)


def test_load_csv_returns_numpy_arrays():
    """load_csv retourne des np.ndarray."""
    from generate_dataset import load_csv

    content = "x,y,ux,uy,p\n0.0,0.0,0.1,0.2,0.5\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        result = load_csv(tmp)
        for arr in result:
            assert isinstance(arr, np.ndarray)
    finally:
        os.unlink(tmp)
