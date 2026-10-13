# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# This code is adapted from neutronpy


#Chopper class for Time of Flight spectrometer


import numpy as np

from .exceptions import ChopperError


class Chopper(object):
    # Chopper Class for using Time of Flight spectrometer
 
    def __init__(self, distance, speed, width, chopper_type, acceptance, counter_rot=False, radius=None, depth=None, tau=None):
        self.distance = distance
        self.speed = speed
        self.width = width
        self.chopper_type = chopper_type
        self.acceptance = acceptance
        if counter_rot:
            self.counter_rot = 2.0
        else:
            self.counter_rot = 1.0
        if radius is not None:
            self.radius = radius
        if depth is not None:
            self.depth = depth
        if tau is not None:
            self.tau_override = tau


    def __repr__(self):
        args = ', '.join(
            [str(getattr(self, key)) for key in ['distance', 'speed', 'width', 'chopper_type', 'acceptance']])
        kwargs = ', '.join(
            ['{0}={1}'.format(key, getattr(self, key)) for key in ['depth', 'tau'] if getattr(self, key, None) is not None])
        return "Chopper({0})".format(', '.join([args, kwargs]))

    @property
    def tau(self):
        """Calculate the time resolution of the chopper

        Returns
        -------
        tau : float
            Chopper time-resolution standard deviation.
            NOTE (units inconsistency, pending code fix): only the disk-no-radius and
            fermi branches apply the 1e6 / 360 conversion to microseconds; the
            disk+radius branch returns seconds (proj_review_comments_v4.md §2.1 #4).
        """
        if hasattr(self, 'tau_override'):
            return self.tau_override

        elif self.chopper_type == 'disk' and hasattr(self, 'radius'):
            return self.acceptance / (self.radius * self.speed * self.counter_rot) / np.sqrt(8 * np.log(2))
        elif self.chopper_type == 'disk' and ~hasattr(self, 'radius'):
            return 1e6 / (self.speed * self.acceptance * self.counter_rot) / 360.0
        elif self.chopper_type == 'fermi':
            try:
                return 1e6 / (self.speed * 2.0 * np.arctan(self.acceptance / self.depth)) / 360.
            except AttributeError:
                raise ChopperError("'depth' not specified, and is a required value for a Fermi Chopper.")

        else:
            raise ChopperError("'{0}' is an invalid chopper_type. Choose 'disk' or 'fermi', or specify custom tau \
                                via `tau_override` attribute".format('chopper_type'))




class Detector(object):
    #Detector Class defining for Time of Flight spectrometer

    def __init__(self, shape, width, height, radius, hpixels, vpixels, tau=0.1, thickness=1, orientation=None, dead_angles=None):
        self.shape = shape
        self.width = width
        self.height = height
        self.radius = radius
        self.tau = tau
        self.thickness = thickness
        self.hpixels = hpixels
        self.vpixels = vpixels
        if dead_angles:
            self.dead_angles = dead_angles
        if orientation:
            self.orientation = orientation

    def __repr__(self):
        args = ', '.join([str(getattr(self, key)) for key in ['shape', 'width', 'height', 'radius']])
        kwargs = ', '.join(
            ['{0}={1}'.format(key, getattr(self, key, None)) for key in ['resolution', 'orientation', 'dead_angles']])
        return "Detector({0})".format(', '.join([args, kwargs]))
        

class Guide(object):
    #Neutron Guide Class

    def __init__(self, m, length, width, height):
        self.m = m
        self.length = length
        self.width = width
        self.height = height

    @property
    def sigma_l(self):
        return np.sqrt(self.length ** 2 + self.width ** 2 + self.height ** 2) - self.length

    @property
    def sigma_theta(self):
        return np.arcsin(np.sqrt(self.m * np.sin(2.03e-3) ** 2))

    @property
    def sigma_phi(self):
        return self.sigma_theta
