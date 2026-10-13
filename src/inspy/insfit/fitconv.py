#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:27:29 2020

@author: dgc

Improved version using scipy.optimize for faster fitting
"""
import numpy as np
from numpy import matlib
import sys
import scipy as spy
from scipy.optimize import least_squares
import time

from ..instrument.tas_spectr import TripleAxisSpectr
from ..instrument.tools import _scalar, _star,_modvec, _cleanargs


class FitConv(object):
    """
    Improved FitConv class using scipy.optimize for faster fitting
    """
    
    def __init__(self, exp=None, sqw=None, prefactor=None, hkle=None, Iobs=None, dIobs=None, param=None, paramfixed=None, 
                 method=1, maxiter=200, tol=1e-6, dtol=1e-6):
        
        if exp is None:
            print("exp is not provided")
            exp=TripleAxisSpectr(efixed=5)
            return

        if sqw is None:
            print("sqw is not provided")
            return

        if prefactor is None:
            print("prefactor is not provided")
            return

        self.exp       =    exp
        self.sqw       =    sqw
        self.prefactor =    prefactor
        self.hkle      =    np.array(hkle)
        self.Iobs      =    Iobs
        self.dIobs     =    dIobs
        self.param     =    param
        self.paramfixed=    paramfixed
        self.method    =    method
        self.accuracy  =    None
        self.maxiter   =    maxiter
        self.tol       =    tol
        self.dtol      =    dtol
        self.fitcount  =    0
        self.chisq     =    0
        self.alpha     =    0
        self.beta      =    0
        

    def __repr__(self):
        return " this is a FitConv class for fitting the exp by convoluting the TAS instrument resolution"
    
    def __str__(self):
        return "this is a FitConv class"
    
    @property
    def exp(self):
        return self._exp

    @exp.setter
    def exp(self, value):
        self._exp = value

    @property
    def sqw(self):
        return self._sqw

    @sqw.setter
    def sqw(self, value):
        self._sqw = value  

    @property
    def prefactor(self):
        return self._prefactor

    @prefactor.setter
    def prefactor(self, value):
        self._prefactor = value  
        
    @property
    def hkle(self):
        return self._hkle

    @hkle.setter
    def hkle(self, value):
        self._hkle = value        
        
    @property
    def Iobs(self):
        return self._Iobs

    @Iobs.setter
    def Iobs(self, value):
        self._Iobs = value      
        

    @property
    def dIobs(self):
        return self._dIobs

    @dIobs.setter
    def dIobs(self, value):
        self._dIobs = value               

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        self._param = value       

    @property
    def paramfixed(self):
        return self._paramfixed

    @paramfixed.setter
    def paramfixed(self, value):
        self._paramfixed = value
        
    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value            
        
    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        self._maxiter = value    

    @property
    def tol(self):
        return self._tol

    @tol.setter  # Fixed: was @param.setter
    def tol(self, value):
        self._tol = value    
        
    @property
    def dtol(self):
        return self._dtol

    @dtol.setter
    def dtol(self, value):
        self._dtol = value            
    
    @property
    def fitcount(self):
        return self._fitcount

    @fitcount.setter
    def fitcount(self, value):
        self._fitcount = value

    @property
    def chisq(self):
        return self._chisq

    @chisq.setter
    def chisq(self, value):
        self._chisq = value    
        
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value    
        
    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value    

    def residual_function(self, free_params):
        """
        Residual function for scipy.optimize.least_squares
        
        Parameters:
        free_params: array of free (varying) parameters
        
        Returns:
        residuals: weighted residuals for least squares fitting
        """
        # Reconstruct full parameter array
        full_params = self.param.copy()
        full_params[self.ivar] = free_params
        
        # Calculate model prediction
        f = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2, 
                           hkle=self.hkle, METHOD=self.method, 
                           ACCURACY=self.accuracy, p=full_params)
        self.fitcount += 1
        
        # Return weighted residuals
        return (self.Iobs - f) / self.dIobs

    def jacobian_function(self, free_params):
        """
        Jacobian function for scipy.optimize.least_squares
        Central numerical differences (two ResConv calls per free parameter, see
        below); there is no analytical Jacobian in this file (proj_review_comments_v4.md §2.2 #11).
        
        Parameters:
        free_params: array of free (varying) parameters
        
        Returns:
        jacobian: matrix of partial derivatives
        """
        # Reconstruct full parameter array
        full_params = self.param.copy()
        full_params[self.ivar] = free_params
        
        npoint = len(self.Iobs)
        jacobian = np.zeros((npoint, len(free_params)))
        
        for i, param_idx in enumerate(self.ivar):
            # Calculate step size
            h = np.zeros(len(full_params))
            step = self.dtol * full_params[param_idx]
            if full_params[param_idx] == 0:
                step = 1e-8
            h[param_idx] = step
            
            # Forward and backward evaluations
            f1 = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                hkle=self.hkle, METHOD=self.method,
                                ACCURACY=self.accuracy, p=full_params + h)
            self.fitcount += 1
            
            f2 = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                hkle=self.hkle, METHOD=self.method,
                                ACCURACY=self.accuracy, p=full_params - h)
            self.fitcount += 1
            
            # Numerical derivative, normalized by error
            jacobian[:, i] = -(f1 - f2) / (2 * step * self.dIobs)
        
        return jacobian

    def fitwithconv(self, exp=None, sqw=None, prefactor=None, hkle=None, Iobs=None, dIobs=None, param=None, paramfixed=None, 
                 method='fix', maxiter=200, tol=1e-6, dtol=1e-6, use_jacobian=True):
        """
        Improved fitting function using scipy.optimize.least_squares
        
        Parameters:
        use_jacobian: bool; True -> jacobian_function (central numerical differences), False -> scipy '2-point'; neither is analytical (§2.2 #11)
        """
        
        t = time.time()
        self.exp = exp
        self.sqw = sqw
        self.prefactor = prefactor
        
        if hkle is not None and not isinstance(hkle, np.ndarray):
            self.hkle = np.array(hkle)

        if Iobs is not None and not isinstance(Iobs, np.ndarray):
            self.Iobs = np.array(Iobs)
            
        if dIobs is not None and not isinstance(dIobs, np.ndarray):
            self.dIobs = np.array(dIobs)
        
        if param is not None:
            self.param = np.array(param)  
           
        if paramfixed is not None:
            self.paramfixed = np.array(paramfixed)
        else:
            self.paramfixed = np.ones(self.param.shape)

        self.method = method
        self.accuracy = None
        self.maxiter = maxiter
        self.tol = tol
        self.dtol = dtol
        self.fitcount = 0
        
        # Process input data
        [H, K, L, W] = self.hkle
        dataLength = H.size

        if dIobs is None:
            dIobs = np.sqrt(self.Iobs)
        #20261010: M11 Fix: the local dIobs was never converted (the np.array() at the top
        #   only updated self.dIobs), so a list argument died on .reshape(-1)
        self.dIobs = np.asarray(dIobs).reshape(-1)

        if not dataLength == self.dIobs.size:
            print('Fatal error: size mismatch between H,K,L,W or EXP and dIobs')
            return
 
        if self.method is None:
            self.method = 'fix'
        
        if self.accuracy is None:
            self.accuracy = np.array([7, 0])

        # Set up optimization
        self.ivar = np.where(self.paramfixed > 0)[0]  # indices of varying parameters
        DF = len(self.Iobs) - len(self.ivar)  # degrees of freedom
        
        dispia = np.where(self.paramfixed > 0, 1, 0)
        print("Fitting {} to {} data points with {} free parameters and {} fixed parameters\n".format(
            sqw, len(self.Iobs), np.sum(dispia), len(self.param) - np.sum(dispia)))

        # Initial parameter values for free parameters only
        x0 = self.param[self.ivar]
        
        #20261010: §2.2 #12 note: `bounds` is NOT passed to least_squares below, so
        #   customizing it currently has no effect (pending wiring, code fix).
        bounds = (-np.inf, np.inf)  # No bounds by default
        
        # Choose jacobian
        jac = self.jacobian_function if use_jacobian else '2-point'
        
        # Perform optimization using scipy's least_squares
        print('Using scipy.optimize.least_squares with trf method')
        print('Iteration info will be displayed by scipy...\n')
        
        # Callback function to display progress
        def callback(xk, rk=None):
            if hasattr(callback, 'nit'):
                callback.nit += 1
            else:
                callback.nit = 1
            
            # Calculate chi-squared for display
            residuals = self.residual_function(xk)
            chisq = np.sum(residuals**2)
            print(f' {callback.nit:8d} {self.fitcount:13d} {chisq/DF:13f} {(time.time()-t):13.1f}s')

        print('Iteration #   FunCount     chi^2/DF           time')
        
        # Run optimization
        result = least_squares(
            self.residual_function,
            x0,
            method='trf',  # Trust Region Reflective (LM would be 'lm')
            jac=jac,
            max_nfev=maxiter * len(x0) * 10,  # Maximum function evaluations
            xtol=tol,
            ftol=tol,
            verbose=0,  # We handle our own progress display
            callback=callback  # already active (the old "uncomment" note was stale; §3.3)
        )
        
        # Update parameters
        self.param[self.ivar] = result.x
        
        # Calculate final statistics
        residuals = self.residual_function(result.x)
        chisq = np.sum(residuals**2)
        self.chisq = chisq
        
        # Calculate errors and correlation matrix
        # Use the Jacobian at the solution
        if result.jac is not None:
            J = result.jac
        else:
            J = self.jacobian_function(result.x)
        
        # Calculate covariance matrix
        try:
            # JTJ = J.T @ J, but we need to be careful about conditioning
            JTJ = np.dot(J.T, J)
            C = np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix encountered. Using pseudo-inverse.")
            C = np.linalg.pinv(JTJ)
        
        # Parameter errors
        dpa = np.zeros(len(self.param))
        dpa[self.ivar] = np.sqrt(np.diag(C))
        
        # Normalized chi-squared
        chisqN = chisq / DF
        
        # Normalized correlation matrix
        CN = C / np.sqrt(np.abs(np.diag(C).reshape(-1, 1) * np.diag(C)))
        
        # Goodness of fit probability
        PQ = 1 - spy.special.gammainc(DF/2, chisq/2)
        
        # Final simulation
        sim = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                             hkle=self.hkle, METHOD=self.method,
                             ACCURACY=self.accuracy, p=self.param)
        self.fitcount += 1
        
        # Convergence analysis
        if result.success:
            kvg = 1
            print(f'\nStop: Converged normally in {result.nfev} function evaluations.')
        else:
            if result.nfev >= maxiter * len(x0) * 10:
                kvg = 0
                print('\nStop: max allowed number of function evaluations exceeded.')
            else:
                kvg = 2
                print(f'\nStop: {result.message}')
        
        print(f'\nTotal fitting time: {time.time() - t:.1f} seconds')
        print(f'Total function evaluations: {self.fitcount}')
        print(f'Final chi^2/DF: {chisqN:.6f}')
        print('\nFinal parameters:\n')
        
        for index, (par, err) in enumerate(zip(self.param, dpa)):
            fixed_str = "fixed" if self.paramfixed[index] == 0 else "free"
            print(f'{index:6d} \t {par:12.6f} ± {err:12.6f} ({fixed_str})')
        
        # Details dictionary
        details = {
            "chisq": chisq,
            "Ndata": len(self.Iobs),
            "Npar": len(self.param),
            "Nva": len(self.ivar),
            "DF": DF,
            "C": C,
            "result": result,
            "final_residuals": residuals
        }
        
        return [self.param, dpa, chisqN, sim, CN, PQ, result.nfev, kvg, details]

    # Keep the old methods for backward compatibility
    def dfdp(self, param, ivar):
        """Backward-compat numerical differences -- has its OWN unpack bug (self.hkle is 4xlength, not npoint x m at :450); prefer jacobian_function (§3.3)"""
        [npoint, m] = self.hkle.shape
        dfp = np.zeros((npoint, len(ivar))) 
        
        for i, item in enumerate(ivar):
            h = np.zeros(len(param))
            t = param[item] + self.dtol * param[item]
            h[item] = t - param[item] 
            if param[item] == 0:
                h[item] = 1e-8
            
            pa1 = param + h
            pa2 = param - h
            
            f1 = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                hkle=self.hkle, METHOD=self.method,
                                ACCURACY=self.accuracy, p=pa1)  
            self.fitcount += 1
            
            f2 = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                                hkle=self.hkle, METHOD=self.method,
                                ACCURACY=self.accuracy, p=pa2)
            self.fitcount += 1

            dfp[:, i] = (f1 - f2) / (2 * h[item])
            
        return dfp

    def marqit(self, param, ivar):
        """Backward compatibility method"""
        f = self.exp.ResConv(sqw=self.sqw, pref=self.prefactor, nargout=2,
                           hkle=self.hkle, METHOD=self.method,
                           ACCURACY=self.accuracy, p=param)
        self.fitcount += 1

        wdiff = (self.Iobs - f) / self.dIobs
        chisq = np.sum(wdiff**2)
                                        
        dfp = self.dfdp(param, ivar)
        NP = len(ivar)
        beta = np.sum(np.matlib.repmat(wdiff/self.dIobs, NP, 1) * dfp.T, 1)
        alpha = dfp.T / np.matlib.repmat(self.dIobs, NP, 1)
        alpha = np.matmul(alpha, alpha.T)
        
        return [chisq, alpha, beta]