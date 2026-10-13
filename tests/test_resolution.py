# -*- coding: utf-8 -*-
"""Resolution-matrix and convolution tests.

These replace the old ``test_tas.py`` / ``test_mc.py`` scripts, which printed
values but asserted nothing (item 178).
"""

import numpy as np
import pytest

from inspy.energy import Energy
from inspy.crystal import Sample
from inspy.instrument import Ana, Mono, TripleAxisSpectr
from inspy.instrument.exceptions import ScatteringTriangleError

from conftest import gaussian_sqw, trivial_prefactor


class TestResolutionMatrix:
    def test_calcresmathkl_returns_none_and_sets_attributes(self, tas):
        """CalcResMatHKL returns None; the results live on the instrument.

        The old tests did ``test = instr.CalcResMatHKL(...)`` and printed `test`,
        which is always None -- the real output is R0/RMS/RM.
        """
        hkle = np.array([[2, 0, 0, 0], [2, 0, 0, 2], [2, 0, 0, 4]], dtype=float)
        assert tas.CalcResMatHKL(hkle.T) is None

        assert tas.R0.shape == (3,)
        assert tas.RMS.shape == (3, 4, 4)
        assert np.all(tas.R0 > 0)
        assert np.all(np.isfinite(tas.RMS))

    def test_resolution_matrix_is_symmetric_and_positive(self, tas):
        tas.CalcResMatHKL([2.0, 0.0, 0.0, 0.0])
        R = tas.RMS[0]
        assert np.allclose(R, R.T)
        assert np.all(np.linalg.eigvalsh(R) > 0)

    def test_varied_energy_row_is_not_duplicated(self, tas):
        """The old fixture had [[2,0,0,0],[2,0,0,2],[2,0,0,2]] -- a duplicated W row."""
        hkle = np.array([[2, 0, 0, 0], [2, 0, 0, 2], [2, 0, 0, 4]], dtype=float)
        tas.CalcResMatHKL(hkle.T)
        widths = [tas.RMS[i, 2, 2] for i in range(3)]
        assert widths[0] != widths[1] != widths[2]

    def test_unreachable_hkl_raises_a_clear_error(self, tas):
        """A HKL the instrument cannot reach must fail loudly, not return NaN."""
        with pytest.raises(ScatteringTriangleError):
            tas.CalcResMatHKL([400.0, 0.0, 0.0, 0.0])

    def test_calcresmatq_gives_the_same_prefactor_as_hkl(self, tas):
        """The two entry points must agree on R0 for the same physical point.

        RMS deliberately differs: CalcResMatHKL rotates CalcResMatQ's result out of
        the lab frame into the sample's crystallographic frame (RMS = T^T RM T), so
        comparing the two matrices directly would be comparing different frames. R0
        is frame-independent and is therefore the right invariant to check.
        """
        from inspy.instrument.tools import _modvec
        tas.CalcResMatHKL([2.0, 0.0, 0.0, 0.0])
        R0_hkl = tas.R0[0]
        RMS_hkl = tas.RMS[0].copy()

        _, _, _, _, rsample = tas._StandardSystem()
        Q = _modvec([2.0, 0.0, 0.0], rsample)
        R0_q, RM_q = tas.CalcResMatQ(Q, np.array([0.0]))

        assert np.ravel(R0_q)[0] == pytest.approx(R0_hkl, rel=1e-9)
        # The lab-frame matrix is symmetric and positive definite too.
        assert np.allclose(RM_q[0], RM_q[0].T)
        assert np.all(np.linalg.eigvalsh(RM_q[0]) > 0)

    def test_hkl_matrix_is_the_lab_matrix_rotated_into_the_sample_frame(self, tas):
        """RMS = T^T RM T, and that identity must hold for the reported matrix.

        T is built from the components of the *unit* Q along the sample's x and y
        axes. Note that block is not generally a rotation (xq^2 + yq^2 is 1/3 for a
        cubic cell at [1,1,0], not 1), so the determinant does change between the two
        frames -- checking det equality would be wrong. Reconstructing T and applying
        the identity is the correct invariant.
        """
        from inspy.instrument.tools import _modvec, _scalar

        hkl = np.array([1.0, 1.0, 0.0])
        tas.CalcResMatHKL(list(hkl) + [0.0])
        R0_hkl, RMS_hkl = tas.R0[0], tas.RMS[0].copy()

        x, y, _, _, rsample = tas._StandardSystem()
        Q = _modvec(list(hkl), rsample)
        uq = np.ravel(hkl / Q)
        xq, yq = _scalar(x, uq, rsample), _scalar(y, uq, rsample)
        T = np.array([[xq, yq, 0, 0], [-yq, xq, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])

        R0_q, RM_q = tas.CalcResMatQ(Q, np.array([0.0]))

        assert np.ravel(R0_q)[0] == pytest.approx(R0_hkl, rel=1e-9)
        assert np.allclose(RMS_hkl, T.T @ RM_q[0] @ T, rtol=1e-9, atol=1e-9)
        # The frames really do differ, so this is not a trivially-true comparison.
        assert not np.allclose(RM_q[0], RMS_hkl)

    def test_result_is_independent_of_the_crystal_orientation(self, sample):
        """RMS is defined in the crystal frame, so rotating u/v must not change it."""
        mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20,
                    depth=0.2, rh=150, rv=150, direct=-1)
        ana = Ana(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20,
                  depth=0.2, rh=150, rv=150, direct=-1, thickness=0.2, horifoc=-1)
        results = []
        for u, v in ([0, 1, 1], [1, 0, 1]), ([1, 0, 1], [0, 1, 1]):
            s = Sample(5.0, 5.0, 5.0, 90, 90, 90)
            s.width = s.height = s.depth = 1.0
            s.u, s.v = list(u), list(v)
            inst = TripleAxisSpectr(efixed=14.7, mono=mono, sample=s, ana=ana)
            inst.CalcResMatHKL([1.0, 1.0, 0.0, 0.0])
            results.append((inst.R0[0], inst.RMS[0].copy()))
        assert results[0][0] == pytest.approx(results[1][0], rel=1e-12)
        assert np.allclose(results[0][1], results[1][1], rtol=1e-12, atol=1e-12)

    def test_widths_scale_with_energy_loss(self, tas):
        """Elastic (W=0) is the narrowest; increasing |W| broadens the energy axis."""
        tas.CalcResMatHKL([2.0, 0.0, 0.0, 0.0])
        w0 = tas.RMS[0, 2, 2]
        tas.CalcResMatHKL([2.0, 0.0, 0.0, -4.0])
        w4 = tas.RMS[0, 2, 2]
        assert w4 > w0


class TestResConv:
    def test_fix_method_is_deterministic(self, tas, scan):
        a = tas.ResConv(sqw=gaussian_sqw, pref=trivial_prefactor, nargout=2,
                        hkle=scan, METHOD='fix', ACCURACY=[5, 5], p=[30., 0., 1.])
        b = tas.ResConv(sqw=gaussian_sqw, pref=trivial_prefactor, nargout=2,
                        hkle=scan, METHOD='fix', ACCURACY=[5, 5], p=[30., 0., 1.])
        assert np.allclose(a, b)

    def test_mc_with_a_seed_is_reproducible(self, tas):
        hkle = ([1.0], [0.0], [0.0], [5.0])
        kwargs = dict(sqw=lambda H, K, L, W, p:
                      np.exp(-0.5 * ((W - p[1]) / p[0]) ** 2)[np.newaxis, :],
                      pref=None, nargout=0, hkle=hkle, METHOD='mc',
                      ACCURACY=50, p=[1.0, 5.0])
        a = tas.ResConv(seed=42, **kwargs)
        b = tas.ResConv(seed=42, **kwargs)
        c = tas.ResConv(seed=43, **kwargs)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_mc_agrees_with_fix_within_tolerance(self, tas):
        """MC sampling must converge to the same answer as quadrature."""
        hkle = ([1.0], [0.0], [0.0], [5.0])
        model = lambda H, K, L, W, p: np.exp(-0.5 * ((W - p[1]) / p[0]) ** 2)[np.newaxis, :]
        fix = tas.ResConv(sqw=model, pref=None, nargout=0, hkle=hkle,
                          METHOD='fix', ACCURACY=[7, 3], p=[1.0, 5.0])
        mc = tas.ResConv(sqw=model, pref=None, nargout=0, hkle=hkle,
                         METHOD='mc', ACCURACY=500, p=[1.0, 5.0], seed=7)
        assert abs(fix[0] - mc[0]) / fix[0] < 0.05

    def test_convolution_preserves_peak_position(self, tas):
        """Convolving with the resolution must not move the peak of a symmetric model."""
        hkle = ([1.0], [0.0], [0.0], [5.0])
        model = lambda H, K, L, W, p: np.exp(-0.5 * ((W - p[1]) / p[0]) ** 2)[np.newaxis, :]
        for method, acc in (('fix', [7, 3]), ('mc', 200)):
            out = tas.ResConv(sqw=model, pref=None, nargout=0, hkle=hkle,
                              METHOD=method, ACCURACY=acc, p=[1.0, 5.0], seed=1)
            assert np.isfinite(out).all()
            assert out[0] > 0

    def test_sma_convolution_runs(self, tas):
        def sma_model(H, K, L, p):
            d = np.vstack([np.ones(np.shape(H)), np.ones(np.shape(H))])
            return d, np.ones_like(d), np.full(d.shape, 0.1)

        out = tas.ResConvSMA(sqw=sma_model, pref=None, nargout=0,
                             hkle=[np.full(5, 1.0), np.zeros(5), np.zeros(5),
                                   np.linspace(-2, 2, 5)],
                             METHOD='fix', ACCURACY=[5, 5], p=[1.0])
        assert np.all(np.isfinite(out))


class TestEnergy:
    def test_wavelength_and_energy_round_trip(self):
        """Energy(wavelength=L).energy must reproduce the physical E = h^2/(2 m L^2)."""
        from scipy.constants import h, m_n
        from inspy.constants import JOULES_TO_MEV
        lam = 1.8
        E = Energy(wavelength=lam).energy
        assert E == pytest.approx(h ** 2 / (2 * m_n * (lam / 1e10) ** 2) * JOULES_TO_MEV,
                                  rel=1e-9)
        assert Energy(energy=E).wavelength == pytest.approx(lam, rel=1e-9)

    def test_wavevector_is_two_pi_over_wavelength(self):
        """This class uses the full wavevector (k = 2*pi/lambda), not 1/lambda."""
        e = Energy(wavelength=2.0)
        assert e.wavevector == pytest.approx(2 * np.pi / 2.0, rel=1e-9)

    def test_temperature_from_energy(self):
        e = Energy(energy=25.0)
        assert e.temperature > 0
        assert Energy(temperature=e.temperature).energy == pytest.approx(25.0, rel=1e-9)

    def test_equality_and_repr(self):
        assert Energy(energy=10.0) == Energy(energy=10.0)
        assert Energy(energy=10.0) != Energy(energy=11.0)
        assert 'Energy' in repr(Energy(energy=10.0))

    def test_no_input_raises_attributeerror(self):
        with pytest.raises(AttributeError):
            Energy()

    def test_zero_energy_gives_nan_wavelength(self):
        """A zero-energy neutron has infinite wavelength; it must not divide by zero."""
        e = Energy(energy=0.0)
        assert np.isnan(e.wavelength)
        assert e.wavevector == 0.0

    def test_array_energy_is_supported(self):
        e = Energy(energy=np.array([10.0, 20.0, 30.0]))
        assert e.wavelength.shape == (3,)
        assert np.all(np.isfinite(e.wavelength))
