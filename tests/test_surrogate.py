# tests/test_surrogate.py
import numpy as np
import pytest


def make_stokes_snapshots(n_sim: int = 30, ngrid: int = 100):
    """
    Snapshots synthétiques vérifiant la loi Stokes exacte : u(ν) = u_ref / ν.
    Utilisé pour tester que le surrogate découvre cette structure.
    """
    rng = np.random.default_rng(0)
    nu = np.logspace(-2, 0, n_sim)
    u_ref = rng.random(ngrid)
    UX = u_ref[None, :] / nu[:, None]   # shape (n_sim, ngrid)
    return nu, UX


class TestComputePOD:
    def test_output_shapes(self):
        """compute_pod retourne les bonnes shapes."""
        from train_surrogate import compute_pod

        rng = np.random.default_rng(0)
        snapshots = rng.random((20, 100))
        mean, modes, coeffs, energy = compute_pod(snapshots, k=3)

        assert mean.shape == (100,)
        assert modes.shape == (3, 100)
        assert coeffs.shape == (20, 3)

    def test_energy_in_0_1(self):
        """L'énergie capturée est dans [0, 1]."""
        from train_surrogate import compute_pod

        rng = np.random.default_rng(0)
        snapshots = rng.random((20, 100))
        _, _, _, energy = compute_pod(snapshots, k=5)

        assert 0.0 < energy <= 1.0

    def test_stokes_one_mode_captures_all_energy(self):
        """Pour Stokes u ∝ 1/ν, 1 seul mode POD doit capturer > 99.9% de l'énergie."""
        from train_surrogate import compute_pod

        _, UX = make_stokes_snapshots(30, 100)
        _, _, _, energy = compute_pod(UX, k=1)

        assert energy > 0.999, f"Énergie mode 1 = {energy:.6f}, attendu > 99.9%"

    def test_reconstruction_faithful(self):
        """Reconstruction POD avec tous les modes : erreur ≈ 0 (reconstruction exacte)."""
        from train_surrogate import compute_pod

        rng = np.random.default_rng(0)
        snapshots = rng.random((20, 100))
        # Avec k = N (tous les modes), reconstruction doit être exacte à la précision machine
        k_full = snapshots.shape[0]   # 20
        mean, modes, coeffs, _ = compute_pod(snapshots, k=k_full)
        reconstructed = coeffs @ modes + mean

        err = np.linalg.norm(reconstructed - snapshots) / np.linalg.norm(snapshots)
        assert err < 1e-10, f"Erreur reconstruction exacte = {err:.2e}, attendu < 1e-10"

    def test_k_exceeds_rank_is_clamped(self):
        """compute_pod clamps k to rank when k > number of snapshots."""
        from train_surrogate import compute_pod
        import warnings

        rng = np.random.default_rng(0)
        snapshots = rng.random((5, 100))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mean, modes, coeffs, energy = compute_pod(snapshots, k=20)
        assert modes.shape[0] == 5
        assert coeffs.shape == (5, 5)
        assert len(w) == 1
        assert "clamped" in str(w[0].message)


class TestFitAndPredict:
    def test_predict_shape(self):
        """predict_field retourne (n_query, ngrid)."""
        from train_surrogate import fit_surrogate, predict_field

        nu, UX = make_stokes_snapshots(20, 50)
        mean, modes, pipe, _ = fit_surrogate(nu[:15], UX[:15], k=3, degree=3)
        pred = predict_field(nu[15:], mean, modes, pipe)

        assert pred.shape == (5, 50)

    def test_stokes_l2_error_below_1pct(self):
        """Surrogate POD doit prédire avec erreur < 1% sur données Stokes exactes."""
        from train_surrogate import fit_surrogate, predict_field, relative_l2_error

        nu, UX = make_stokes_snapshots(40, 200)
        nu_train, nu_test = nu[:35], nu[35:]
        UX_train, UX_test = UX[:35], UX[35:]

        mean, modes, pipe, _ = fit_surrogate(nu_train, UX_train, k=3, degree=3)
        UX_pred = predict_field(nu_test, mean, modes, pipe)

        err = relative_l2_error(UX_pred, UX_test)
        assert err < 0.01, f"Erreur L2 = {err*100:.3f}%, attendu < 1%"

    def test_predict_scalar_nu(self):
        """predict_field accepts a scalar nu, not just arrays."""
        from train_surrogate import fit_surrogate, predict_field

        nu, UX = make_stokes_snapshots(20, 50)
        mean, modes, pipe, _ = fit_surrogate(nu[:15], UX[:15], k=3, degree=3)
        pred = predict_field(0.5, mean, modes, pipe)
        assert pred.shape == (1, 50)

    def test_predict_nu_zero_raises(self):
        """predict_field raises ValueError for nu <= 0."""
        from train_surrogate import fit_surrogate, predict_field

        nu, UX = make_stokes_snapshots(20, 50)
        mean, modes, pipe, _ = fit_surrogate(nu[:15], UX[:15], k=3, degree=3)
        with pytest.raises(ValueError, match="nu must be > 0"):
            predict_field(np.array([0.0]), mean, modes, pipe)


class TestRelativeL2Error:
    def test_identical_arrays_gives_zero(self):
        from train_surrogate import relative_l2_error

        a = np.random.rand(5, 100)
        assert relative_l2_error(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_double_pred_gives_one(self):
        """Prédiction = 2×vrai → erreur relative = 1.0."""
        from train_surrogate import relative_l2_error

        true = np.ones((3, 10))
        pred = 2 * np.ones((3, 10))
        err = relative_l2_error(pred, true)
        assert err == pytest.approx(1.0)

    def test_zero_norm_true_returns_finite(self):
        """Zero-norm true vector should return finite value (1e-15 guard)."""
        from train_surrogate import relative_l2_error

        true = np.zeros((3, 10))
        pred = np.ones((3, 10))
        err = relative_l2_error(pred, true)
        assert np.isfinite(err)
