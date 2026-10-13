# -*- coding: utf-8 -*-
# This code is adapted from neutronpy (ResLib 3.4c naming conventions)

import numpy as np

from .tools import get_tau


#20261010: §2.1 #5 Fix: the tau table below was an orphaned string literal (not attached
#   to any class or function; Mono.__doc__ was None). Converted to comments; the values
#   match tools.get_tau exactly.
#A structure that describes the monochromator.
#
#Attributes
#----------
#tau : str or float
#    The monochromator reciprocal lattice vector in Å^(-1)
#    One can use one of the following keyword strings or numerical numbers:
#
#    +------------------+--------------+-----------+
#    | String           |     τ        |           |
#    +==================+==============+===========+
#    | Be(002)          | 3.50702      |           |
#    +------------------+--------------+-----------+
#    | Co0.92Fe0.08(200)| 3.54782      | (Heusler) |
#    +------------------+--------------+-----------+
#    | Cu(002)          | 3.47714      |           |
#    +------------------+--------------+-----------+
#    | Cu(111)          | 3.01063      |           |
#    +------------------+--------------+-----------+
#    | Cu(220)          | 4.91642      |           |
#    +------------------+--------------+-----------+
#    | Cu2MnAl(111)     | 1.82810      | (Heusler) |
#    +------------------+--------------+-----------+
#    | Ge(111)          | 1.92366      |           |
#    +------------------+--------------+-----------+
#    | Ge(220)          | 3.14131      |           |
#    +------------------+--------------+-----------+
#    | Ge(311)          | 3.68351      |           |
#    +------------------+--------------+-----------+
#    | Ge(511)          | 5.76968      |           |
#    +------------------+--------------+-----------+
#    | Ge(533)          | 7.28063      |           |
#    +------------------+--------------+-----------+
#    | PG(002)          | 1.87325      |           |
#    +------------------+--------------+-----------+
#    | PG(004)          | 3.74650      |           |
#    +------------------+--------------+-----------+
#    | PG(110)          | 5.49806      |           |
#    +------------------+--------------+-----------+
#    | Si(111)          | 2.00421      |           |
#    +------------------+--------------+-----------+
#
#20261010: L4 Fix: doc value for Cu(111) was 2.99913 (= 2*pi/2.095, the duplicate that was
#   removed from tools.get_tau) -- updated to 2*pi/2.087 = 3.01063, consistent with the
#   Cu(002)/Cu(220) entries (a = 3.614 A).

class Mono(object):
    #Monochromator

    def __init__(self, tau, mosaic, vmosaic=None, height=None, width=None, depth=None, rh=None, rv=None, direct=-1):
        self._tau   = tau
        self.mosaic = mosaic
        if vmosaic is not None:
            self.vmosaic = vmosaic
        if rh is not None:
            self.rh = rh
        if rv is not None:
            self.rv = rv
        if height is not None:
            self.height = height
        if width  is not None:
            self.width  = width
        if depth  is not None:
            self.depth  = depth
        
        self.dir = direct
        self.d = 2 * np.pi / get_tau(tau)
        
        
    def __repr__(self):
        args   = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in ['tau', 'mosaic']])
        kwargs = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in
                            [ 'vmosaic', 'height', 'width', 'depth', 'rh', 'rv', 'direct'] if
                            getattr(self, key, None) is not None])
        return "Monochromator({0})".format(', '.join([args, kwargs]))

    def __eq__(self, right):
        self_parent_keys  = sorted(list(self.__dict__.keys()))
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

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self.d = 2 * np.pi / get_tau(tau)
        

    @property
    def direct(self):
        return self._dir

    @direct.setter
    def direct(self, value):
        self._dir = value
        
    @property
    def mosaic(self):
        return self._mosaic

    @mosaic.setter
    def mosaic(self, value):
        self._mosaic = value
        
    @property
    def vmosaic(self):
        return self._vmosaic

    @vmosaic.setter
    def vmosaic(self, value):
        self._vmosaic = value
    
    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        
    @property
    def depth(self):
        return self._depth
    @depth.setter
    def depth(self, value):
        self._depth = value
        
    @property
    def rh(self):
        return self._rh
    @rh.setter
    def rh(self, value):
        self._rh = value  
        
    @property
    def rv(self):
        return self._rv
        
    @rv.setter
    def rv(self, value):
        self._rv = value        
        
        
        