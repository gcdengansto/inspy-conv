# -*- coding: utf-8 -*-
# This code is adapted from neutronpy




from .mono import Mono


class Ana(Mono):
    #Class analyzer

    def __init__(self, tau, mosaic, vmosaic=None, height=None, width=None, depth=None, rh=None, rv=None,
                 horifoc=-1, thickness=None,  direct=-1, Q=None):
        super(Ana, self).__init__(tau=tau, mosaic=mosaic, vmosaic=vmosaic, height=height, width=width, depth=depth, rh=rh, rv=rv, direct=direct)
        if thickness is not None:
            self.thickness = thickness
        if Q is not None:
            self.Q = Q
        self.horifoc = horifoc

    def __repr__(self):
        args = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in ['tau', 'mosaic']])
        kwargs = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in
                            [ 'vmosaic', 'height', 'width', 'depth', 'rh', 'rv', 'horifoc', 'thickness', 'direct','Q']
                            if getattr(self, key, None) is not None])
        return "Analyzer({0})".format(', '.join([args, kwargs]))
