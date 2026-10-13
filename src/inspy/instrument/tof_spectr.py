# -*- coding: utf-8 -*-

#Define a Time Of Flight instrument for resolution calculations
#unfinished

import numpy as np
from ..constants import e, hbar, neutron_mass
from ..crystal import Sample
from ..energy import Energy
from .components import Chopper, Detector,Guide
from .exceptions import DetectorError
from .tools import _cleanargs, chop, get_angle_ki_Q, get_kfree


class TimeOfFlightSpectr:
   
    

    def __init__(self, ei=3.0, choppers=None, sample=None, detector=None, guides=None, theta_i=0, phi_i=0, **kwargs):
        self._ei = Energy(energy=ei)
        return
        


    def __repr__(self):
        return "Instrument('tof',  ei={0})".format(self.ei)

    @property
    def ei(self):
        r"""Incident Energy object of type :py:class:`.Energy`

        """
        return self._ei

    @ei.setter
    def ei(self, value):
        self._ei = Energy(energy=value)

    @property
    def orient1(self):
        return self.sample.u

    @property
    def orient2(self):
        return self.sample.v

    def CalcResMatQ(self, Q, W):
        return


    def CalcResMatHKL(self, hkle):
        return
