# -*- coding: utf-8 -*-
#

import inspect
from numbers import Number
import numpy as np

from ..constants import neutron_mass, hbar
from ..crystal import Sample
from ..energy import Energy
from .exceptions import AnalyzerError, MonochromatorError, ScatteringTriangleError


class _dummy(object):
    #Empty class 

    def __init__(self, name='Dummy', **kwargs):
        self.name = name
        self.__dict__.update(kwargs)

    def __eq__(self, right):
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if not np.all(self_parent_keys == right_parent_keys):
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            if not np.all(value == right_parent_val):
                return False

        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    def __repr__(self):
        return "{1}({0})".format(', '.join(
            ['{0}={1}'.format(key, getattr(self, key)) for key in self.__dict__.keys() if
             getattr(self, key, None) is not None and key != 'name']), self.name)


def _scalar(v1, v2, lattice):
    #Calculates the _scalar product of two vectors, defined by their
    #fractional cell coordinates or Miller indexes.
    #Based on ResLib 3.4c, originally authored by A. Zheludev


    [x1, y1, z1] = v1
    [x2, y2, z2] = v2

    s = x1 * x2 * lattice.a ** 2 + y1 * y2 * lattice.b ** 2 + z1 * z2 * lattice.c ** 2 + \
        (x1 * y2 + x2 * y1) * lattice.a * lattice.b * np.cos(lattice.gamma) + \
        (x1 * z2 + x2 * z1) * lattice.a * lattice.c * np.cos(lattice.beta) + \
        (z1 * y2 + z2 * y1) * lattice.c * lattice.b * np.cos(lattice.alpha)

    return s


def _star(lattice):
    #Given lattice parametrs, calculate unit cell volume V, reciprocal
    #volume Vstar, and reciprocal lattice parameters.
    #Based on ResLib 3.4c, originally authored by A. Zheludev

    V = 2 * lattice.a * lattice.b * lattice.c * \
        np.sqrt(np.sin((lattice.alpha + lattice.beta + lattice.gamma) / 2) *
                np.sin((-lattice.alpha + lattice.beta + lattice.gamma) / 2) *
                np.sin((lattice.alpha - lattice.beta + lattice.gamma) / 2) *
                np.sin((lattice.alpha + lattice.beta - lattice.gamma) / 2))

    Vstar = (2 * np.pi) ** 3 / V

    latticestar = Sample(0, 0, 0, 0, 0, 0)
    latticestar.a = 2 * np.pi * lattice.b * lattice.c * np.sin(lattice.alpha) / V
    latticestar.b = 2 * np.pi * lattice.a * lattice.c * np.sin(lattice.beta) / V
    latticestar.c = 2 * np.pi * lattice.b * lattice.a * np.sin(lattice.gamma) / V
    latticestar.alpha = np.arccos((np.cos(lattice.beta) * np.cos(lattice.gamma) -
                                   np.cos(lattice.alpha)) / (np.sin(lattice.beta) * np.sin(lattice.gamma)))
    latticestar.beta = np.arccos((np.cos(lattice.alpha) * np.cos(lattice.gamma) -
                                  np.cos(lattice.beta)) / (np.sin(lattice.alpha) * np.sin(lattice.gamma)))
    latticestar.gamma = np.arccos((np.cos(lattice.alpha) * np.cos(lattice.beta) -
                                   np.cos(lattice.gamma)) / (np.sin(lattice.alpha) * np.sin(lattice.beta)))

    return [V, Vstar, latticestar]


def _modvec(v, lattice):
    #Calculates the modulus of a vector, defined by its fractional cell

    return np.sqrt(_scalar(v, v, lattice))


def get_tau(x, getlabel=False):
    # get the τ-value for common monochromator and analyzer crystals by name or value.

    choices = {'pg(002)'.lower(): 1.87325,
               'pg(004)'.lower(): 3.74650,
               'ge(111)'.lower(): 1.92366,
               'ge(220)'.lower(): 3.14131,
               'ge(311)'.lower(): 3.68351,
               'be(002)'.lower(): 3.50702,
               'pg(110)'.lower(): 5.49806,
               'Cu2MnAl(111)'.lower(): 2 * np.pi / 3.437,
               'Co0.92Fe0.08(200)'.lower(): 2 * np.pi / 1.771,
               'Ge(511)'.lower(): 2 * np.pi / 1.089,
               'Ge(533)'.lower(): 2 * np.pi / 0.863,
               'Si(111)'.lower(): 2 * np.pi / 3.135,
                #20261010: L4 Fix: 'Cu(111)' was defined twice (kept entry below; the
                #   duplicate 2*pi/2.095 was removed). The sibling Cu(002)=1.807 and
                #   Cu(220)=1.278 entries imply a=3.614 A -> d111=a/sqrt(3)=2.087
                #   (2.095 was 0.4% off).
               'Cu(111)'.lower(): 2 * np.pi / 2.087,
               'Cu(002)'.lower(): 2 * np.pi / 1.807,
               'Cu(220)'.lower(): 2 * np.pi / 1.278}

    if getlabel:
        # return the index/label of the closest monochromator
        choices_ = dict((key, np.abs(value - x)) for (key, value) in choices.items())
        index = min(choices_, key=choices_.get)
        if np.abs(choices_[index]) < 5e-4:
            return index  # the label
        else:
            return ''
    #20261010: L4 Fix: also accept numpy scalars -- np.int64/np.float32 are not subclasses
    #   of Python int/float, so they fell through to x.lower() -> AttributeError
    elif isinstance(x, (int, float, np.integer, np.floating)):
        return x
    else:
        try:
            return choices[x.lower()]
        except KeyError:
            #20261010: L5 Fix: f_locals["self"] raised KeyError when the caller had no self,
            #   and falling through both branches returned None silently (GetTau.m errors);
            #   now every path raises
            calling_class = repr(inspect.stack()[1][0].f_locals.get("self", ""))
            if calling_class.startswith('Monochromator'):
                raise MonochromatorError("Invalid Monochromator crystal type: {0}".format(repr(x)))
            elif calling_class.startswith('Analyzer'):
                raise AnalyzerError("Invalid Analyzer crystal type: {0}".format(repr(x)))
            else:
                raise ValueError("Unknown crystal type: {0}".format(repr(x)))


def _cleanargs(*varargin):
    #Broadcasts arguments to common length N (longest input); repeats LAST value (MATLAB
    #CleanArgs repeats first). No row-vector reshape is done, despite the old wording.
    #Based on ResLib 3.4c, originally authored by A. Zheludev

    varargout = []
    lengths = np.array([], dtype=np.int32)
    for arg in varargin:
        if not isinstance(arg, list) and not isinstance(arg, np.ndarray):
            arg = [arg]
        varargout.append(np.array(arg))
        lengths = np.concatenate((lengths, [len(arg)]))

    length = max(lengths)
    bad = np.where(lengths < length)
    if len(bad[0]) > 0:
        for i in bad[0]:
            varargout[i] = np.concatenate((varargout[i], [varargout[i][-1]] * int(length - lengths[i])))
            lengths[i] = len(varargout[i])

    if len(np.where(lengths < length)[0]) > 0:
        raise ValueError('All inputs must have the same lengths: inputs had lengths {0}'.format(lengths))

    return [length] + varargout


def _voigt(x, a):
    def _approx1(t):
        return (t * 0.5641896) / (0.5 + t ** 2)

    def _approx2(t, u):
        return (t * (1.410474 + u * 0.5641896)) / (0.75 + (u * (3. + u)))

    def _approx3(t):
        return (16.4955 + t *
                (20.20933 + t *
                 (11.96482 + t *
                  (3.778987 + 0.5642236 * t)))) / (16.4955 + t *
                                                   (38.82363 + t *
                                                    (39.27121 + t *
                                                     (21.69274 + t *
                                                      (6.699398 + t)))))

    def _approx4(t, u):
        return (t * (36183.31 - u *
                     (3321.99 - u *
                      (1540.787 - u *
                       (219.031 - u *
                        (35.7668 - u *
                         (1.320522 - u *
                          0.56419)))))) / (32066.6 - u *
                                           (24322.8 - u *
                                            (9022.23 - u *
                                             (2186.18 - u *
                                              (364.219 - u *
                                               (61.5704 - u *
                                                (1.84144 - u))))))))

    #20261010 Bug #1.3: make _voigt shape-preserving so it accepts N-D input.
    #   WHAT WAS WRONG
    #     * `nx = x.size` / `y = np.zeros(nx)` flattened only the OUTPUT to 1-D
    #       (length = x.size), while `a`, `t`, `ax`, `s`, `u` kept their original
    #       k-D shape. Every `np.where(...)` below therefore returned a k-tuple of
    #       index arrays, and indexing the 1-D `y` with a k-tuple (k > 1) raised
    #       "IndexError: too many indices for array" -> all three call sites in
    #       ResConvSMA (tas_spectr.py:1527, :1594, :1642) were unreachable.
    #     * The guard `if len(a) == 1` inspected a's FIRST AXIS length: for a k-D
    #       array with modes == 1 it is 1 even when `a` holds L*P elements (so
    #       `np.ones(nx) * a` mis-broadcast), and for a scalar `a` it raised
    #       TypeError, although ResLib's `if length(a)==1, a(1:nx)=a; end`
    #       explicitly supports a scalar.
    #   WHAT CHANGED
    #     * `x`/`a` are coerced with np.asarray and flattened to 1-D (`xf`, `af`)
    #       BEFORE any arithmetic; the caller's shape is remembered in `shape`.
    #     * The broken `len(a) == 1` test is replaced by a single
    #       `np.broadcast_to(a, shape).ravel()`, which is correct for scalars,
    #       1-element arrays and already-matching shapes alike.
    #     * `y` is allocated from `xf.size` (== the old `nx` for 1-D input), so
    #       the seven masks below now operate on 1-D arrays, where `np.where`
    #       yields a 1-tuple and `y[good] = ...` is legal. Only the operand
    #       NAMES changed there (x/a -> xf/af); every threshold and boolean
    #       expression is unchanged.
    #     * the result is returned as `y.reshape(shape)`, i.e. the same shape the
    #       caller passed in, so the call sites can broadcast normally.
    #   WHAT DELIBERATELY DID NOT CHANGE
    #     * The seven approximation branches and their thresholds stay byte-for-
    #       byte identical to ResLib ConvResSMA.m:147-159 (including the
    #       overlapping-mask quirk in the final fallback), so numerics keep
    #       tracking the reference.
    #     * dtypes stay as before (complex64 workspace, real output).
    x = np.asarray(x)
    a = np.asarray(a)
    shape = x.shape

    xf = x.ravel()
    af = np.broadcast_to(a, shape).ravel()

    nx = xf.size
    y = np.zeros(nx, dtype=np.complex64)

    t = af - 1j * xf
    ax = np.abs(xf)
    s = ax + af
    u = t ** 2

    #20261010 Bug #1.3: masks now index the FLATTENED xf/af (not the caller's
    #   possibly N-D x/a). Thresholds and boolean logic are unchanged.
    good = np.where(af == 0)
    y[good] = np.exp(-xf[good] ** 2)

    good = np.where((af >= 15) | (s >= 15))
    y[good] = _approx1(t[good])

    good = np.where((s < 15) & (af < 15) & (af >= 5.5))
    y[good] = _approx2(t[good], u[good])

    good = np.where((s < 15) & (s >= 5.5) & (af < 5.5))
    y[good] = _approx2(t[good], u[good])

    good = np.where((s < 5.5) & (af < 5.5) & (af >= 0.75))
    y[good] = _approx3(t[good])

    good = np.where((s < 5.5) & (af >= 0.195 * ax - 0.176) & (af < 0.75))
    y[good] = _approx3(t[good])

    good = np.where((~((s < 5.5) & (af >= 0.195 * ax - 0.176))) & (af < 0.75))
    y[good] = np.exp(u[good]) - _approx4(t[good], u[good])

    y = np.real(y)
    #20261010 Bug #1.3: restore the caller's original shape (see comment at entry).
    return y.reshape(shape)


def project_into_plane(index, r0, rm):
    #Projects out-of-plane resolution into a plane by a gaussian integral over the
    #axis named by `index` (sole caller passes 0 -- not a fixed "third" axis).


    r = np.sqrt(2 * np.pi / rm[index, index]) * r0
    mp = rm

    b = rm[:, index] + rm[index, :].T
    b = np.delete(b, index, 0)

    mp = np.delete(mp, index, 0)
    mp = np.delete(mp, index, 1)

    mp -= 1 / (4. * rm[index, index]) * np.outer(b, b.T)

    return [r, mp]


def _ellipse(saxis1, saxis2, phi=0, origin=None, npts=31):
    #Returns an ellipse.

    if origin is None:
        origin = [0., 0.]

    theta = np.linspace(0., 2. * np.pi, npts)

    x = np.array(saxis1 * np.cos(theta) * np.cos(phi) - saxis2 * np.sin(theta) * np.sin(phi)) + origin[0]
    y = np.array(saxis1 * np.cos(theta) * np.sin(phi) + saxis2 * np.sin(theta) * np.cos(phi)) + origin[1]
    return np.vstack((x, y))


def get_bragg_widths(RM):
    #Returns the Bragg widths given a resolution matrix.
    #20261010 Bug #2.1: REMOVED the trailing `* 2` — these values were 2x too large.
    #   WHAT WAS WRONG
    #     * each element is sqrt(8*ln2)/sqrt(RM[i,i]) = 2*sqrt(2*ln2)*sigma, which is
    #       ALREADY the FWHM of a Gaussian of variance RM[i,i]:
    #           sqrt(8*ln2) = sqrt(4 * 2*ln2) = 2*sqrt(2*ln2)
    #     * `return bragg * 2` then doubled it a second time -> returned 2x FWHM.
    #   WHY REMOVING IT IS THE CORRECT MINIMAL FIX
    #     * ResLib computes the HALF width and doubles only when printing:
    #         ResPlot.m:151-153  XBWidth=sqrt(2*log(2))./sqrt(RMS(1,1,:))   -> HWHM
    #         ResPlot.m:166,169   num2str(XBWidth(center)*2, 4)             -> FWHM
    #       i.e. ResLib's *printed* number is the FWHM, which is what the Python
    #       consumer also claims to print:
    #         tas_spectr.py:2302  'Bragg width in [Q_1,Q_2,E] (FWHM):'
    #         tas_spectr.py:2303-2305  formats bragg_widths[...] directly, with NO
    #                                  further *2 (unlike the fproject/XBWidth path at
    #                                  :1880, :2048, :2227, which does *2 at print and
    #                                  therefore matches ResPlot.m exactly).
    #     * So the function contract (README: "Bragg FWHM from resolution matrix") is
    #       satisfied by returning FWHM directly, and the single call site needs no
    #       change. This also fixes element [3], because get_phonon_width()[1] at
    #       :355 (the `fwhm = ...` line) is sqrt(8*ln2)/sqrt(MP[0,0]) = FWHM and was
    #       being doubled too.
    #   NOT CHANGED
    #     * the five element definitions, the axis order [Q1, Q2, E, phonon, Qz],
    #       and get_phonon_width() itself.

    bragg = np.array([np.sqrt(8 * np.log(2)) / np.sqrt(RM[0, 0]),
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[1, 1]),
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[2, 2]),
                      get_phonon_width(0, RM, [0, 0, 0, 1])[1],
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[3, 3])])

    return bragg


def get_phonon_width(r0, M, C):
    T = np.diag(np.ones(4))
    T[3, :] = np.array(C)
    S = np.linalg.inv(T)
    MP = S.T.conj() @ M @ S
    [rp, MP] = project_into_plane(0, r0, MP)
    [rp, MP] = project_into_plane(0, rp, MP)
    [rp, MP] = project_into_plane(0, rp, MP)
    fwhm = np.sqrt(8 * np.log(2)) / np.sqrt(MP[0, 0])

    return [rp, fwhm]


def fproject(mat, i):
    if i == 0:
        v = 2
        j = 1
    elif i == 1:
        v = 0
        j = 2
    elif i == 2:
        v = 0
        j = 1
    else:
        raise ValueError('i={0} is an invalid value!'.format(i))

    [a, b, c] = mat.shape
    proj = np.zeros((2, 2, c))
    proj[0, 0, :] = mat[i, i, :] - mat[i, v, :] ** 2 / mat[v, v, :]
    proj[0, 1, :] = mat[i, j, :] - mat[i, v, :] * mat[j, v, :] / mat[v, v, :]
    proj[1, 0, :] = mat[j, i, :] - mat[j, v, :] * mat[i, v, :] / mat[v, v, :]
    proj[1, 1, :] = mat[j, j, :] - mat[j, v, :] ** 2 / mat[v, v, :]
    hwhm = proj[0, 0, :] - proj[0, 1, :] ** 2 / proj[1, 1, :]
    hwhm = np.sqrt(2. * np.log(2.)) / np.sqrt(hwhm)

    return hwhm


def calc_proj_hwhm(MP):
    #
    theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
    S = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    MP = S @ MP @ S.T.conj()

    hwhm_xp = 1.17741 / np.sqrt(MP[0, 0])
    hwhm_yp = 1.17741 / np.sqrt(MP[1, 1])

    return hwhm_xp, hwhm_yp, theta


def get_angle_ki_Q(ki, kf, Q, gonio_dir=-1, outside_scat_tri=False):
    #Returns the angle between ki and Q for rotation from
    #[ki, ki_perp, kz, w] to [q_perp, q_para, qz, w] reference frame.

    if Q == 0:
        angle = np.pi / 2.0
    else:
        c = (ki ** 2 - kf ** 2 + Q ** 2) / (2.0 * ki * Q)
        if abs(c) > 1.0:
            raise ScatteringTriangleError

        angle = np.arccos(c)

    if outside_scat_tri:
        angle = np.pi - angle

    return angle * np.sign(gonio_dir)


def get_kfree(W, kfixed, ki_fixed=True):
    #Calculates the free wavevector, either ki or kf, as specified by

    kE_sq = Energy(energy=W).wavevector ** 2
    if ki_fixed:
        kE_sq = -kE_sq

    k_sq = kE_sq + kfixed ** 2

    if k_sq < 0.0:
        raise ScatteringTriangleError
    else:
        return np.sqrt(k_sq)


def chop(matrix, tol=1e-12):
    #Rounds values within tol of zero down(up) to zero.
    #KNOWN ISSUE (inherited from neutronpy, pending code fix): the scalar branch below
    #   tests `if matrix < tol` WITHOUT abs(), so EVERY negative scalar is zeroed, not
    #   only those within tol of zero; the array branches correctly use np.abs
    #   (proj_review_comments_v4.md §2.1 #6).

    if isinstance(matrix, tuple):
        return tuple(chop(item) for item in matrix)
    elif isinstance(matrix, list):
        return list(chop(item) for item in matrix)
    elif isinstance(matrix, (np.ndarray, np.matrixlib.defmatrix.matrix)):
        if np.iscomplexobj(matrix):
            rmat = matrix.real.copy()
            rmat[np.abs(rmat) < tol] = 0.0

            imat = matrix.imag.copy()
            imat[np.abs(imat) < tol] = 0.0

            return rmat + 1j * imat
        else:
            matrix[np.abs(matrix) < tol] = 0.0
            return matrix
    elif isinstance(matrix, Number):
        if np.iscomplex(matrix):
            real = matrix.real
            imag = matrix.imag
            if real < tol:
                real = 0.0
            if imag < tol:
                imag = 0.0
            return real + imag * 1j
        else:
            if matrix < tol:
                return 0.0
            else:
                return matrix
    else:
        print(type(matrix))
        return matrix
