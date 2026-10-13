#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Fast FitConv - Optimized for maximum speed
Removes interface compatibility constraints for performance gains
"""
import numpy as np
import scipy as spy
from scipy.optimize import least_squares
import time
from functools import lru_cache
import warnings

from ..instrument.tas_spectr import TripleAxisSpectr
from ..instrument.tools import _scalar, _star,_modvec, _cleanargs


class UltraFastFitConv(object):
    """
    Ultra-optimized FitConv class prioritizing speed over interface compatibility
    """
    
    def __init__(self, exp, sqw, prefactor, hkle, Iobs, dIobs=None, 
                 cache_size=128, precompute_data=True):
        """
        Streamlined initialization with mandatory parameters
        
        Parameters:
        exp: instrument object
        sqw: S(Q,w) model function  
        prefactor: prefactor function
        hkle: [H, K, L, E] arrays as numpy arrays
        Iobs: observed intensities as numpy array
        dIobs: intensity errors as numpy array (optional, will use sqrt(Iobs) if None)
        cache_size: currently STORED BUT UNUSED -- the model-eval cache is hard-coded @lru_cache(maxsize=128) in _cached_model_eval (§2.2 #15)
        precompute_data: whether to precompute and validate data upfront
        """
        
        # Store references (no property overhead)
        self.exp = exp
        self.sqw = sqw
        self.prefactor = prefactor
        
        # Preprocess and validate data once
        if precompute_data:
            self._preprocess_data(hkle, Iobs, dIobs)
        else:
            #20261010: M10 Fix: this fast path must also set npoints/inv_dIobs -- they
            #   are used at :137/:148/:179/:216/:222 and were only set by
            #   _preprocess_data, so precompute_data=False died with AttributeError
            self.hkle = np.asarray(hkle, dtype=np.float64)
            self.Iobs = np.asarray(Iobs, dtype=np.float64)
            self.dIobs = np.asarray(dIobs, dtype=np.float64) if dIobs is not None else np.sqrt(self.Iobs)
            self.npoints = self.hkle[0].size
            self.inv_dIobs = 1.0 / self.dIobs
        
        # Performance tracking
        self.fitcount = 0
        self._cache_size = cache_size  # stored but never read (hard-coded @lru_cache(128) below; §2.2 #15)
        
        # Pre-allocate arrays for gradient calculations
        self._gradient_workspace = None
        self._last_params_hash = None
        
    def _preprocess_data(self, hkle, Iobs, dIobs):
        """Preprocess and validate all data once for maximum speed"""
        
        # Convert to optimal numpy arrays
        self.hkle = np.asarray(hkle, dtype=np.float64)
        self.Iobs = np.asarray(Iobs, dtype=np.float64).ravel()
        
        if dIobs is None:
            self.dIobs = np.sqrt(np.maximum(self.Iobs, 1e-10))  # Avoid division by zero
        else:
            self.dIobs = np.asarray(dIobs, dtype=np.float64).ravel()
            
        # Validate dimensions
        [H, K, L, W] = self.hkle
        self.npoints = H.size
        
        if self.Iobs.size != self.npoints:
            raise ValueError(f"Data size mismatch: hkle has {self.npoints} points, Iobs has {self.Iobs.size}")
        if self.dIobs.size != self.npoints:
            raise ValueError(f"Data size mismatch: hkle has {self.npoints} points, dIobs has {self.dIobs.size}")
            
        # Pre-calculate inverse errors for speed (avoid repeated division)
        self.inv_dIobs = 1.0 / self.dIobs
        
        # Check for problematic data
        if np.any(self.dIobs <= 0):
            warnings.warn("Zero or negative errors detected, replacing with sqrt(Iobs)")
            bad_errors = self.dIobs <= 0
            self.dIobs[bad_errors] = np.sqrt(np.maximum(self.Iobs[bad_errors], 1e-10))
            self.inv_dIobs = 1.0 / self.dIobs

    @lru_cache(maxsize=128)
    def _cached_model_eval(self, params_tuple, method='fix', accuracy_tuple=(7, 0)):
        """Cached model evaluation to avoid repeated identical calculations"""
        params_array = np.array(params_tuple)
        accuracy_array = np.array(accuracy_tuple)
        
        result = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                 hkle=self.hkle, METHOD=method,
                                 ACCURACY=accuracy_array, p=params_array)
        self.fitcount += 1
        return result

    def residual_vectorized(self, free_params, ivar, full_params_template, method='fix', accuracy=(7, 0)):
        """
        Ultra-fast vectorized residual function
        
        Parameters:
        free_params: array of free parameters only
        ivar: indices of free parameters
        full_params_template: template for full parameter array
        method: resolution method
        accuracy: accuracy parameters
        """
        # Reconstruct full parameters (fast in-place operation)
        full_params = full_params_template.copy()
        full_params[ivar] = free_params
        
        # Use caching for identical parameter sets
        params_tuple = tuple(full_params)
        accuracy_tuple = tuple(accuracy) if isinstance(accuracy, np.ndarray) else accuracy
        
        try:
            f = self._cached_model_eval(params_tuple, method, accuracy_tuple)
        except TypeError:
            # Fallback for non-hashable types
            f = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                               hkle=self.hkle, METHOD=method,
                               ACCURACY=np.array(accuracy), p=full_params)
            self.fitcount += 1
        
        # Vectorized residual calculation
        return (self.Iobs - f) * self.inv_dIobs

    def jacobian_optimized(self, free_params, ivar, full_params_template, 
                          method='fix', accuracy=(7, 0), rel_step=1e-8):
        """
        Optimized Jacobian calculation with adaptive step sizes
        """
        full_params = full_params_template.copy()
        full_params[ivar] = free_params
        
        nparams = len(free_params)
        jacobian = np.zeros((self.npoints, nparams))
        
        # Adaptive step size calculation
        steps = np.where(np.abs(free_params) > 1e-12, 
                        rel_step * np.abs(free_params),
                        rel_step)
        
        # Vectorized gradient computation where possible
        for i, param_idx in enumerate(ivar):
            step = steps[i]
            
            # Forward and backward parameter arrays
            params_forward = full_params.copy()
            params_backward = full_params.copy()
            params_forward[param_idx] += step
            params_backward[param_idx] -= step
            
            # Function evaluations with caching
            try:
                f_forward = self._cached_model_eval(tuple(params_forward), method, tuple(accuracy))
                f_backward = self._cached_model_eval(tuple(params_backward), method, tuple(accuracy))
            except TypeError:
                f_forward = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                           hkle=self.hkle, METHOD=method,
                                           ACCURACY=np.array(accuracy), p=params_forward)
                f_backward = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                            hkle=self.hkle, METHOD=method,
                                            ACCURACY=np.array(accuracy), p=params_backward)
                self.fitcount += 2
            
            # Central difference with pre-computed inverse errors
            jacobian[:, i] = -(f_forward - f_backward) * self.inv_dIobs / (2 * step)
        
        return jacobian

    def fit_ultrafast(self, param_initial, param_fixed_mask=None, 
                     method='fix', accuracy=(7, 0),
                     maxfev=None, xtol=1e-8, ftol=1e-8, gtol=1e-8,
                     use_analytical_jacobian=True, verbose=True,
                     early_stopping=True, convergence_window=5):
        """
        Ultra-fast fitting with all optimizations enabled
        
        Parameters:
        param_initial: initial parameter values
        param_fixed_mask: boolean mask (True = free, False = fixed)
        method: resolution convolution method
        accuracy: accuracy parameters
        maxfev: maximum function evaluations (auto-calculated if None)
        xtol, ftol, gtol: convergence tolerances
        use_analytical_jacobian: use our optimized jacobian
        verbose: display progress
        early_stopping: stop if convergence stalls
        convergence_window: window for early stopping detection
        """
        
        t_start = time.time()
        self.fitcount = 0
        
        # Parameter setup
        param_initial = np.asarray(param_initial, dtype=np.float64)
        if param_fixed_mask is None:
            param_fixed_mask = np.ones(len(param_initial), dtype=bool)
        else:
            param_fixed_mask = np.asarray(param_fixed_mask, dtype=bool)
        
        ivar = np.where(param_fixed_mask)[0]
        nfree = len(ivar)
        DF = self.npoints - nfree
        
        if maxfev is None:
            maxfev = max(200 * nfree, 1000)
        
        if verbose:
            print(f"Ultra-fast fitting: {self.npoints} data points, {nfree} free parameters")
            print(f"Method: {method}, Max evaluations: {maxfev}")
            print("FuncEvals     Chi²/DF        Time(s)")
            print("-" * 40)
        
        # Optimization setup
        x0 = param_initial[ivar]
        full_params_template = param_initial.copy()
        
        # Create optimized functions with pre-bound parameters
        def fast_residual(free_params):
            return self.residual_vectorized(free_params, ivar, full_params_template, method, accuracy)
        
        if use_analytical_jacobian:
            def fast_jacobian(free_params):
                return self.jacobian_optimized(free_params, ivar, full_params_template, method, accuracy)
            jac = fast_jacobian
        else:
            jac = '2-point'  # Let scipy handle it
        
        # Progress tracking for LM method
        progress_data = {'last_chi2': float('inf'), 'stall_count': 0, 'call_count': 0}
        
        def tracked_residual(free_params):
            progress_data['call_count'] += 1
            residuals = fast_residual(free_params)
            
            if verbose and progress_data['call_count'] % 20 == 0:  # Every 20 calls
                chi2 = np.sum(residuals**2) / DF
                elapsed = time.time() - t_start
                print(f"{self.fitcount:8d}   {chi2:10.6f}   {elapsed:8.2f}")
                
                # Early stopping check
                if early_stopping:
                    if abs(progress_data['last_chi2'] - chi2) < ftol * chi2:
                        progress_data['stall_count'] += 1
                    else:
                        progress_data['stall_count'] = 0
                    progress_data['last_chi2'] = chi2
                    
                    if progress_data['stall_count'] >= convergence_window:
                        if verbose:
                            print("Early stopping: convergence detected")
            
            return residuals
        
        # Run optimization with all speed optimizations
        try:
            result = least_squares(
                tracked_residual,
                x0,
                method='lm',
                jac=jac,
                max_nfev=maxfev,
                xtol=xtol,
                ftol=ftol,
                gtol=gtol,
                verbose=0
            )
        except KeyboardInterrupt:
            if verbose:
                print("\nFitting interrupted by user")
            raise
        
        # Process results
        final_params = full_params_template.copy()
        final_params[ivar] = result.x
        
        # Calculate final statistics
        final_residuals = fast_residual(result.x)
        chisq = np.sum(final_residuals**2)
        chisq_reduced = chisq / DF if DF > 0 else np.inf
        
        # Parameter errors from Jacobian
        param_errors = np.zeros(len(param_initial))
        if result.jac is not None:
            try:
                # Calculate covariance matrix
                JTJ = result.jac.T @ result.jac
                C = np.linalg.inv(JTJ)
                param_errors[ivar] = np.sqrt(np.diag(C))
            except np.linalg.LinAlgError:
                if verbose:
                    print("Warning: Could not calculate parameter errors (singular matrix)")
        
        # Final model evaluation
        final_model = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                     hkle=self.hkle, METHOD=method,
                                     ACCURACY=np.array(accuracy), p=final_params)
        self.fitcount += 1
        
        elapsed_total = time.time() - t_start
        
        if verbose:
            print("-" * 40)
            print(f"Fitting completed in {elapsed_total:.2f} seconds")
            print(f"Total function evaluations: {self.fitcount}")
            print(f"Final χ²/DF: {chisq_reduced:.6f}")
            print(f"Convergence: {'Success' if result.success else 'Failed'}")
            print(f"Exit message: {result.message}")
            
            print("\nFinal Parameters:")
            for i, (val, err) in enumerate(zip(final_params, param_errors)):
                status = "free" if param_fixed_mask[i] else "fixed"
                print(f"  {i:2d}: {val:12.6f} ± {err:12.6f} ({status})")
        
        # Return comprehensive results
        return {
            'params': final_params,
            'param_errors': param_errors,
            'chi2_reduced': chisq_reduced,
            'chi2': chisq,
            'model': final_model,
            'residuals': final_residuals,
            'success': result.success,
            'message': result.message,
            'nfev': self.fitcount,
            'time_elapsed': elapsed_total,
            'degrees_of_freedom': DF,
            'scipy_result': result
        }

    def clear_cache(self):
        """Clear the LRU cache to free memory"""
        self._cached_model_eval.cache_clear()
        
    def get_cache_stats(self):
        """Get cache performance statistics"""
        return self._cached_model_eval.cache_info()