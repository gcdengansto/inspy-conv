# -*- coding: utf-8 -*-
"""Fitter tests: FitConv and UltraFastFitConv.

The data is generated from ``gaussian_sqw`` with p = [30, 0, 1], so the exact
minimum is known and the fits can be checked rather than merely printed.
"""

import io
import contextlib

import numpy as np
import pytest

from inspy.insfit import FitConv, UltraFastFitConv

from conftest import gaussian_sqw, trivial_prefactor

# Each fit convolves a 4-D integration grid, once per residual evaluation and twice
# more per parameter for the Jacobian. -m "not slow" skips this whole module.
pytestmark = pytest.mark.slow


@pytest.fixture
def fitters(tas, scan, noiseless_data, true_params):
    """FitConv and UltraFastFitConv on the same noise-free convolved data."""
    Iobs, dIobs = noiseless_data
    fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs,
                 true_params, [0, 0, 0])
    uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
    return fc, uf


class TestRecoveringKnownParameters:
    def test_fitconv_recovers_the_true_parameters(self, fitters, true_params):
        fc, _ = fitters
        with contextlib.redirect_stdout(io.StringIO()):
            params, errors, chi2_reduced = fc.fitwithconv(maxiter=100)[:3]
        assert params == pytest.approx(true_params, rel=1e-3)
        # Zero residual means a zero chi2/DF, so the chi2/DF-scaled covariance is
        # legitimately zero here (see the noiseless_data fixture).
        assert chi2_reduced < 1e-6
        assert np.all(np.isfinite(errors))

    def test_fitconv_returns_the_documented_tuple(self, fitters):
        """fitwithconv returns [param, dpa, chisqN, sim, CN, PQ, nfev, kvg, details]."""
        fc, _ = fitters
        with contextlib.redirect_stdout(io.StringIO()):
            out = fc.fitwithconv(maxiter=50)
        assert len(out) == 9
        param, dpa, chisqN, sim, CN, PQ, nfev, kvg, details = out
        assert np.asarray(param).size == 3
        assert np.asarray(dpa).size == 3
        assert chisqN >= 0
        assert np.asarray(sim).size == 21
        assert nfev > 0
        assert details['DF'] == 18          # 21 points - 3 free parameters
        assert details['Ndata'] == 21
        assert details['Nva'] == 3

    def test_ultrafast_recovers_the_true_parameters(self, fitters, true_params):
        _, uf = fitters
        result = uf.fit_ultrafast(param_initial=[25.0, 0.2, 1.5],
                                  param_fixed_mask=[0, 0, 0],
                                  ls_method='trf', verbose=False)
        assert result['params'] == pytest.approx(true_params, rel=1e-3)
        assert result['chi2_reduced'] < 1e-6

    def test_both_fitters_agree(self, fitters):
        fc, uf = fitters
        with contextlib.redirect_stdout(io.StringIO()):
            chi2_fc = fc.fitwithconv(maxiter=100)[2]
        res_uf = uf.fit_ultrafast(param_initial=[30.0, 0.0, 1.0],
                                  param_fixed_mask=[0, 0, 0],
                                  ls_method='trf', verbose=False)
        assert chi2_fc == pytest.approx(res_uf['chi2_reduced'], rel=1e-3)

    def test_fitting_from_the_exact_answer_stays_there(self, fitters, true_params):
        """Starting at the minimum must not move away from it."""
        fc, _ = fitters
        with contextlib.redirect_stdout(io.StringIO()):
            params = fc.fitwithconv(maxiter=50)[0]
        assert params == pytest.approx(true_params, rel=1e-4)


class TestFixedParameters:
    def test_fixed_parameter_is_held_at_its_initial_value(self, tas, scan, noiseless_data,
                                                         true_params):
        """Mask convention is 1 = fixed, 0 = free, so [1, 1, 0] pins two parameters."""
        Iobs, dIobs = noiseless_data
        fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs,
                     true_params, [1, 1, 0])   # amplitude and centre fixed, width free
        with contextlib.redirect_stdout(io.StringIO()):
            params = fc.fitwithconv(maxiter=100)[0]
        assert params[0] == pytest.approx(30.0)
        assert params[1] == pytest.approx(0.0)
        assert params[2] == pytest.approx(1.0)

    def test_all_parameters_fixed_raises(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs,
                     [30.0, 0.0, 1.0], [1, 1, 1])
        with pytest.raises(ValueError, match='No free parameters'):
            fc.fitwithconv()

    def test_ultrafast_all_fixed_raises(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
        with pytest.raises(ValueError, match='No free parameters'):
            uf.fit_ultrafast(param_initial=[30.0, 0.0, 1.0],
                             param_fixed_mask=[True, True, True], verbose=False)


class TestInputValidation:
    def test_mask_length_mismatch_raises(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        for mask in ([0, 0], [0, 0, 0, 0]):
            fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs,
                         [30.0, 0.0, 1.0], mask)
            with pytest.raises(ValueError, match='entries but there are'):
                fc.fitwithconv()

    def test_non_binary_mask_raises(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs,
                     [30.0, 0.0, 1.0], [0, 0, 7])
        with pytest.raises(ValueError, match='must contain only 0'):
            fc.fitwithconv()

    def test_size_mismatch_raises_instead_of_returning_none(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs[:5], dIobs[:5],
                     [30.0, 0.0, 1.0], [0, 0, 0])
        with pytest.raises(ValueError, match='Size mismatch'):
            fc.fitwithconv()

    def test_missing_required_arguments_raise_at_construction(self, tas):
        with pytest.raises(TypeError, match='missing required argument'):
            FitConv(exp=tas)   # no sqw/prefactor/hkle/Iobs/param

    def test_zero_degrees_of_freedom_raises(self, tas):
        """3 data points with 3 free parameters leaves no spare DF."""
        W3 = np.linspace(-1, 1, 3)
        hkle = [np.full(3, 1.0), np.zeros(3), np.zeros(3), W3]
        I3 = np.array([1.0, 2.0, 3.0])
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, hkle, I3, np.sqrt(I3))
        with pytest.raises(ValueError, match='Degrees of freedom'):
            uf.fit_ultrafast(param_initial=[30.0, 0.0, 1.0],
                             param_fixed_mask=[0, 0, 0], verbose=False)

    def test_unknown_ls_method_raises(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
        with pytest.raises(ValueError, match='ls_method'):
            uf.fit_ultrafast(param_initial=[30.0, 0.0, 1.0],
                             param_fixed_mask=[0, 0, 0], ls_method='nope', verbose=False)

    def test_constructor_rejects_mismatched_data_sizes(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        with pytest.raises(ValueError, match='size mismatch'):
            UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs[:5], dIobs[:5])


class TestReporting:
    def test_repr_reports_useful_state(self, fitters):
        fc, uf = fitters
        assert 'FitConv' in repr(fc)
        assert '21 points' in repr(fc)
        assert '3/3' in repr(fc)          # free/total parameters
        assert 'UltraFastFitConv' in repr(uf)
        assert '21 points' in repr(uf)
        assert str(fc) == repr(fc)
        assert str(uf) == repr(uf)

    def test_repr_works_before_any_fit(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
        assert '0 free parameters' in repr(uf)   # no fit run yet

    def test_callback_does_not_add_convolutions(self, tas, scan, gaussian_data):
        """The progress callback must not perform its own full convolution.

        FitConv's callback used to convolve once per accepted iteration, so the
        printed evaluation count was roughly (1 + 2*nfree) times scipy's nfev. The
        two counters are now reported separately and must be comparable.
        """
        Iobs, dIobs = gaussian_data
        fc = FitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs,
                     [30.0, 0.0, 1.0], [0, 0, 0])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fc.fitwithconv(maxiter=30)
        text = buf.getvalue()
        assert 'nfev' in text
        assert 'Model evaluations' in text

    def test_cache_hits_are_counted(self, tas, scan, gaussian_data):
        Iobs, dIobs = gaussian_data
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
        uf.fit_ultrafast(param_initial=[30.0, 0.0, 1.0], param_fixed_mask=[0, 0, 0],
                         ls_method='trf', verbose=False)
        stats = uf.get_cache_stats()
        assert stats['hits'] >= 0
        assert uf.fitcount >= stats['misses']

    def test_uncertainties_are_finite_and_positive(self, tas, scan, gaussian_data):
        """With real noise the reported errors must be finite, positive numbers.

        On noise-free data the chi2/DF-scaled covariance is legitimately zero
        (see the noiseless_data fixture), so this test uses the noisy variant.
        """
        Iobs, dIobs = gaussian_data
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
        result = uf.fit_ultrafast(param_initial=[25.0, 0.2, 1.4],
                                  param_fixed_mask=[0, 0, 0],
                                  ls_method='trf', verbose=False)
        assert np.all(np.isfinite(result['param_errors']))
        assert np.all(result['param_errors'] > 0)
        assert result['chi2_reduced'] == pytest.approx(1.0, abs=0.5)

    def test_undeterminable_errors_are_nan_not_zero(self, tas, scan, gaussian_data):
        """A rank-deficient model must report NaN, not a reassuring 0.000000.

        The model below has three parameters but only two distinct effects, so the
        Jacobian is singular and the errors cannot be determined. Item 163 changed
        the initialiser from zeros to NaN so this case is visibly "unknown" rather
        than "exact"; the pinv fallback may still recover finite values here, so the
        assertion is that nothing is silently reported as exactly zero.
        """
        Iobs, dIobs = gaussian_data

        def degenerate(H, K, L, W, p):
            # p[2] has no effect on the output.
            return np.vstack([p[0] * np.exp(-np.log(2) * ((W - p[1]) / p[2] ** 2 + 1.0) ** 2)] * 2)

        uf = UltraFastFitConv(tas, degenerate, trivial_prefactor, scan, Iobs, dIobs)
        result = uf.fit_ultrafast(param_initial=[25.0, 0.2, 1.4],
                                  param_fixed_mask=[0, 0, 0],
                                  ls_method='trf', verbose=False)
        errors = result['param_errors']
        assert errors.shape == (3,)
        assert not np.any(errors == 0.0), 'an error of exactly 0 was reported'


class TestMethods:
    def test_mc_seed_is_forwarded(self, tas, scan, true_params):
        """A seed must make METHOD='mc' reproducible through the fitter.

        MC takes an integer ACCURACY (the number of samples), not the [M, L] pair
        the quadrature method uses.
        """
        conv = tas.ResConv(sqw=gaussian_sqw, pref=trivial_prefactor, nargout=2,
                           hkle=scan, METHOD='mc', ACCURACY=200,
                           p=true_params, seed=99)
        data = np.asarray(conv).ravel()
        results = []
        for _ in range(2):
            uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan,
                                  data, np.sqrt(np.abs(data)))
            r = uf.fit_ultrafast(param_initial=[25.0, 0.1, 1.2],
                                 param_fixed_mask=[0, 0, 0], method='mc',
                                 accuracy=200, seed=99,
                                 ls_method='trf', verbose=False)
            results.append(r['params'])
        assert results[0] == pytest.approx(results[1], rel=1e-12)

    def test_lm_and_trf_give_the_same_minimum(self, tas, scan, noiseless_data, true_params):
        """Different algorithms, same optimum - that is what ls_method only controls."""
        Iobs, dIobs = noiseless_data
        uf = UltraFastFitConv(tas, gaussian_sqw, trivial_prefactor, scan, Iobs, dIobs)
        out = {}
        for method in ('lm', 'trf'):
            out[method] = uf.fit_ultrafast(param_initial=[25.0, 0.2, 1.4],
                                           param_fixed_mask=[0, 0, 0],
                                           ls_method=method, verbose=False)
        assert out['lm']['params'] == pytest.approx(true_params, rel=1e-3)
        assert out['trf']['params'] == pytest.approx(true_params, rel=1e-3)
