# -*- coding: utf-8 -*-
"""Shared fixtures for the inspy test suite.

Every test here is headless: matplotlib uses the Agg backend, Qt uses the
offscreen platform plugin, and no call to ``plt.show()`` is made anywhere (the old
module-level scripts did, which hung pytest on a machine with no display --
item 178).
"""

import os

import pytest

# Must be set before matplotlib or Qt is imported by anything.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from inspy.crystal import Sample, Material, SpaceGroup, Lattice
from inspy.instrument import Mono, Ana, TripleAxisSpectr, get_tau, Chopper
from inspy.insfit import FitConv, UltraFastFitConv


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test leaves open, so tests do not leak state."""
    yield
    plt.close('all')


@pytest.fixture
def sample():
    s = Sample(3, 4, 5, 90, 90, 90)
    s.width = s.height = s.depth = 1.0
    s.u = [0, 1, 1]
    s.v = [1, 0, 1]
    return s


@pytest.fixture
def tas(sample):
    mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20,
                depth=0.2, rh=150, rv=150, direct=-1)
    ana = Ana(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20,
              depth=0.2, rh=150, rv=150, direct=-1, thickness=0.2, horifoc=-1)
    # sample_u/sample_v are deliberately NOT passed: the instrument uses the sample
    # object's own u/v and warns that the arguments were ignored.
    return TripleAxisSpectr(efixed=14.7, mono=mono, sample=sample, ana=ana)


# --------------------------------------------------------------------------------------
# A Gaussian S(Q,w) model and a trivial prefactor, used by the convolution and fitting
# tests. Both are deliberately analytic so the expected answer can be checked by hand.
# --------------------------------------------------------------------------------------

def gaussian_sqw(H, K, L, W, p):
    """One Lorentzian-squared-ish Gaussian mode in energy: amplitude, centre, HWHM."""
    amplitude, centre, hwhm = p[0], p[1], p[2]
    return np.vstack([amplitude * np.exp(-np.log(2) * ((W - centre) / hwhm) ** 2)] * 2)


def trivial_prefactor(H, K, L, W, EXP, p):
    return np.ones(np.shape(H)), np.zeros(np.shape(W))


@pytest.fixture
def scan():
    """[H, K, L, W] for a 21-point energy scan at a single Q."""
    n = 21
    W = np.linspace(-3.0, 3.0, n)
    return [np.full(n, 1.0), np.zeros(n), np.zeros(n), W]


@pytest.fixture
def true_params():
    """Parameters the data will be generated from."""
    return [30.0, 0.0, 1.0]


@pytest.fixture
def noiseless_data(tas, scan, true_params):
    """Convolved model with no measurement noise.

    Data taken from the *bare* (unconvolved) model cannot be inverted by a fit --
    the fitter just compensates by widening the peak. Generating the data through
    the same ResConv the fitter uses is what makes the recovered parameters
    meaningful.

    Note the consequence: with no noise, chi2/DF is ~0 at the minimum, so the
    chi2/DF-scaled covariance is ~0 too and the fitted uncertainties come out at
    zero. That is arithmetically correct, not a bug; use ``gaussian_data`` when
    the uncertainty magnitudes matter.
    """
    conv = tas.ResConv(sqw=gaussian_sqw, pref=trivial_prefactor, nargout=2,
                       hkle=scan, METHOD='fix', ACCURACY=[5, 5], p=true_params)
    data = np.asarray(conv).ravel()
    assert data.size == scan[0].size, 'ResConv returned %d points' % data.size
    return data, np.sqrt(np.abs(data))


@pytest.fixture
def gaussian_data(noiseless_data):
    """Noiseless_data plus 1-sigma noise, so chi2/DF ~ 1 and errors are meaningful."""
    clean, dIobs = noiseless_data
    rng = np.random.default_rng(20240616)   # fixed seed: reproducible
    return clean + rng.normal(0.0, dIobs), dIobs
