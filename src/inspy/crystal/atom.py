# -*- coding: utf-8 -*-
# This file is adapted from neutronpy


import numpy as np
from ..constants import periodic_table, scattering_lengths


class Atom(object):
    #Class for adding atoms to the Material class.

    def __init__(self, ion, pos, occupancy=1., Mcell=None, massNorm=False, Uiso=0, Uaniso=np.zeros((3, 3))):
        self.ion = ion
        self.pos = np.array(pos)
        self.occupancy = occupancy
        self.Mcell = Mcell
        self.Uiso = Uiso
        self.Uaniso = np.array(Uaniso)

        if isinstance(scattering_lengths()[ion]['Coh b'], list):
            b = complex(*scattering_lengths()[ion]['Coh b'])
        else:
            b = scattering_lengths()[ion]['Coh b']

        if massNorm is True:
            self.mass = periodic_table()[ion]['mass']

            self.b = (b * self.occupancy * self.Mcell / np.sqrt(self.mass))
        else:
            self.b = b / 10.

        self.coh_xs = scattering_lengths()[ion]['Coh xs']
        self.inc_xs = scattering_lengths()[ion]['Inc xs']
        self.abs_xs = scattering_lengths()[ion]['Abs xs']

    def __repr__(self):
        return "Atom('{0}')".format(self.ion)


class MagneticAtom(object):
    #Class for adding magnetic atoms to the Material class.


    def __init__(self, ion, pos, moment, occupancy):
        self.ion = ion
        self.pos = np.array(pos)
        self.moment = moment
        self.occupancy = occupancy

    def __repr__(self):
        return "MagneticAtom('{0}')".format(self.ion, self.pos, self.moment, self.occupancy)
