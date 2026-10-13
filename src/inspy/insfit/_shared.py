# -*- coding: utf-8 -*-
"""Shared internals for :class:`~inspy.insfit.fitconv.FitConv` and
:class:`~inspy.insfit.uffitconv.UltraFastFitConv`.

The two fitters implement the same algorithm twice -- the ``ResConv`` model call
(review item 186), the central-difference Jacobian (187), the residual assembly
and covariance (188). The review's preferred end state is a shared ``FitBase``
class (#189); that is a large restructure of two working, heavily-patched fitters,
so this module takes the incremental route: one implementation of each piece,
called from both. Behaviour is unchanged; only the duplication is gone, and
promoting these to a base class later is now a mechanical change.

Everything here is pure: no fitter state is mutated, and every function takes what
it needs as arguments. That is what makes it safe to share between a
straightforward fitter and a caching one.
"""

import numpy as np


def model_eval(instrument, sqw, prefactor, hkle, params, method='fix',
               accuracy=(5, 5), seed=None):
    """The one place either fitter calls ``ResConv``.

    Every convolution in the package goes through here, so the arguments are
    stated once: ``nargout=2`` (a two-output prefactor), the ``METHOD`` /
    ``ACCURACY`` / ``p`` / ``seed`` spellings, and ``ACCURACY`` always coerced to
    an array. Previously nine separate call sites in ``fitconv.py`` and three in
    ``uffitconv.py`` each restated this, and they had already drifted -- the
    ultra-fast fitter passed ``ACCURACY=np.array(...)`` while ``fitconv.py`` did
    not.

    Returns the convolution unchanged; the caller counts the evaluation.
    """
    accuracy = np.asarray(accuracy)
    return instrument.ResConv(sqw=sqw, pref=prefactor, nargout=2, hkle=hkle,
                              METHOD=method, ACCURACY=accuracy, p=params,
                              seed=seed)


def assemble_residual(observed, model, inv_errors):
    """Weighted residual, identically in both fitters.

    ``FitConv`` computed ``(Iobs - f) / dIobs`` and ``UltraFastFitConv``
    ``(Iobs - f) * inv_dIobs``; these differ only in whether the reciprocal was
    precomputed. One implementation, taking the reciprocal that the fast path
    already has.
    """
    return (observed - model) * inv_errors


def central_difference_jacobian(evaluate, free_params, ivar, full_params_template,
                                steps):
    """Central-difference Jacobian: ``d(residual)/d(free parameter)``.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(full_params) -> model``. Supplying this is what lets the two
        fitters share the algorithm while keeping their own caching policy: the
        fast one passes a cached evaluator, the plain one a direct one.
    free_params : ndarray
        Current values of the free parameters.
    ivar : ndarray of int
        Indices of the free parameters within ``full_params``.
    full_params_template : ndarray
        The full parameter vector; ``free_params`` is written into it at ``ivar``.
    steps : ndarray
        Absolute finite-difference step per free parameter. The caller owns the
        step-size policy (``rel_step * |p| + abs_step``); this function just uses
        them, so the two fitters keep their own defaults without forking the
        algorithm.

    Returns
    -------
    (npoints, n_free) array of ``d residual / d free parameter``.
    """
    full_params = full_params_template.copy()
    full_params[ivar] = free_params

    npoints = None
    jacobian = None
    for i, param_idx in enumerate(ivar):
        step = steps[i]

        forward = full_params.copy()
        backward = full_params.copy()
        forward[param_idx] += step
        backward[param_idx] -= step

        f_forward = evaluate(forward)
        f_backward = evaluate(backward)

        if jacobian is None:
            npoints = len(f_forward)
            jacobian = np.zeros((npoints, len(ivar)))

        # Residuals are (Iobs - f) * inv_dIobs, so
        #   d(residual)/dp = -df/dp * inv_dIobs
        # and df/dp is the central difference of the two model evaluations.
        jacobian[:, i] = -(f_forward - f_backward) / (2 * step)

    return jacobian


def parameter_errors(jacobian, chisq_reduced, ivar, nparam, eps=None):
    """Parameter uncertainties from the Jacobian, or NaN where undeterminable.

    ``inv(J.T @ J)`` must be scaled by the reduced chi-squared to be a covariance
    (item 63): the raw inverse gives the errors of an exact fit, which are
    systematically too large with sqrt(Iobs) counting errors.

    ``np.linalg.inv`` does not raise on a merely ill-conditioned matrix, so the
    condition number is checked explicitly and a pseudo-inverse is used instead
    (item 64).

    Parameters with no entry in ``ivar`` (the fixed ones) come back as NaN. So do
    all of them if the Jacobian is unusable. NaN rather than 0 is deliberate: a
    0.0 prints as ``+- 0.000000``, which reads as an exact value (item 163).
    """
    errors = np.full(nparam, np.nan)
    if jacobian is None:
        return errors

    eps = np.finfo(float).eps if eps is None else eps
    jtj = jacobian.T @ jacobian
    try:
        if np.linalg.cond(jtj) > 1 / eps:
            raise np.linalg.LinAlgError('ill-conditioned JTJ')
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj)

    cov = cov * chisq_reduced
    diag = np.diag(cov).copy()
    diag[diag < 0] = 0.
    errors[ivar] = np.sqrt(diag)
    return errors


def covariance_matrix(jacobian, chisq_reduced, nparam, eps=None):
    """The full covariance matrix (zeros for the fixed parameters).

    Same scaling and conditioning rules as :func:`parameter_errors`; kept
    separate because ``FitConv`` returns ``C`` (and a correlation matrix ``CN``) in
    its ``details`` dictionary and ``UltraFastFitConv`` does not.
    """
    eps = np.finfo(float).eps if eps is None else eps
    if jacobian is None:
        return np.zeros((nparam, nparam))

    jtj = jacobian.T @ jacobian
    try:
        if np.linalg.cond(jtj) > 1 / eps:
            raise np.linalg.LinAlgError('ill-conditioned JTJ')
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj)
    return cov * chisq_reduced


def validate_param_mask(mask, nparam, name='paramfixed', allow_bool=True):
    """Validate a fixed/free mask and return it as a boolean array.

    ``1`` (or ``True``) means fixed, ``0`` (or ``False``) means free -- the
    convention throughout the package.

    A mask of the wrong length is silently harmful rather than loud: ``ivar``
    indexes *into* the parameter vector, so a short mask leaves trailing
    parameters frozen at their initial values while the fitter still counts them
    as free, and a long mask raises ``IndexError`` from deep inside the fit. A
    non-binary mask coerces silently too (``2`` becomes ``True``, i.e. fixed).
    Both are rejected here instead.

    Raises
    ------
    ValueError
        On a length mismatch, or on values that are not 0/1.
    """
    mask = np.asarray(mask)
    if mask.size != nparam:
        raise ValueError(
            '{0} has {1} entries but there are {2} parameters; the mask must have '
            'one entry per parameter.'.format(name, mask.size, nparam))
    if mask.dtype != bool:
        if not allow_bool or not np.all(np.isin(mask, (0, 1))):
            raise ValueError(
                '{0} must contain only 0 (free) or 1 (fixed){1}; got {2}.'.format(
                    name, ' or booleans' if allow_bool else '',
                    np.unique(mask).tolist()))
    return mask.astype(bool)
