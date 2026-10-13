# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

# Material constructor

import inspy.constants as const
import numpy as np

from .atom import Atom, MagneticAtom
from .sample import Sample
from .structure_factors import MagneticStructureFactor, NuclearStructureFactor
from .symmetry import SpaceGroup

# dead: this module never existed in the repo (neutronpy remnant; §3.3)


class MagneticUnitCell(Sample):
    # defining a magnetic unit cell

    def __init__(self, unit_cell):
        if 'chirality' not in unit_cell:
            self.chirality = 0
        if 'ph' not in unit_cell:
            self.phase = 0
        self.atoms = []
        if 'space_group' in unit_cell:
            self.space_group = SpaceGroup(unit_cell['space_group'])
            for atom in unit_cell['atoms']:
                symmetrized_positions = self.space_group.symmetrize_position(atom['pos'])
                for pos in symmetrized_positions:
                    self.atoms.append(MagneticAtom(atom['ion'],
                                                   pos,
                                                   atom['moment'],
                                                   atom['occupancy']))
        else:
            for atom in unit_cell['atoms']:
                self.atoms.append(MagneticAtom(atom['ion'],
                                               atom['pos'],
                                               atom['moment'],
                                               atom['occupancy']))

        self.propagation_vector = unit_cell['propagation_vector']

        a, b, c = unit_cell['lattice']['abc']
        alpha, beta, gamma = unit_cell['lattice']['abg']
        super(MagneticUnitCell, self).__init__(a, b, c, alpha, beta, gamma)

    def __repr__(self):
        return "MagneticUnitCell('{0}')".format(self.propagation_vector)


class Material(Sample, NuclearStructureFactor, MagneticStructureFactor):
    #Class for the Material being supplied for the structure factor calculation


    def __init__(self, crystal):
        self.name = crystal['name']

        if 'formulaUnits' not in crystal:
            crystal['formulaUnits'] = 1.

        self.muCell = 0.
        for item in crystal['composition']:
            if 'occupancy' not in item:
                item['occupancy'] = 1.
            self.muCell += const.periodic_table()[item['ion']]['mass'] * item['occupancy']

        self.Mcell = self.muCell * crystal['formulaUnits']

        if 'lattice' in crystal:
            a, b, c = crystal['lattice']['abc']
            alpha, beta, gamma = crystal['lattice']['abg']

        if 'wavelength' in crystal:
            self.wavelength = crystal['wavelength']
        else:
            self.wavelength = 2.359

        if 'space_group' in crystal:
            self.space_group = SpaceGroup(crystal['space_group'])
            self.atoms = []
            for atom in crystal['composition']:
                if 'Uiso' not in atom:
                    atom['Uiso'] = 0
                if 'Uaniso' not in atom:
                    atom['Uaniso'] = np.array(np.zeros((3, 3)))
                if 'occupancy' not in atom:
                    atom['occupancy'] = 1.
                symmetrized_positions = self.space_group.symmetrize_position(atom['pos'])
                for pos in symmetrized_positions:
                    self.atoms.append(Atom(atom['ion'],
                                           pos,
                                           atom['occupancy'],
                                           self.Mcell,
                                           crystal['massNorm'],
                                           atom['Uiso'],
                                           atom['Uaniso']))
        else:
            self.atoms = []
            for item in crystal['composition']:
                if 'Uiso' not in item:
                    item['Uiso'] = 0
                if 'Uaniso' not in item:
                    item['Uaniso'] = np.array(np.zeros((3, 3)))
                if 'occupancy' not in item:
                    item['occupancy'] = 1.
                self.atoms.append(Atom(item['ion'],
                                       item['pos'],
                                       item['occupancy'],
                                       self.Mcell,
                                       crystal['massNorm'],
                                       item['Uiso'],
                                       item['Uaniso']))

        if 'magnetic_unit_cell' in crystal:
            self.magnetic_unit_cell = MagneticUnitCell(crystal['magnetic_unit_cell'])

        if 'mosaic' not in crystal:
            crystal['mosaic'] = None
        if 'vmosaic' not in crystal:
            crystal['vmosaic'] = None
        if 'u' not in crystal:
            crystal['u'] = None
        if 'v' not in crystal:
            crystal['v'] = None
        if 'dir' not in crystal:
            crystal['dir'] = 1

        #20261010 Bug #2.5: pass Sample's optional arguments BY NAME -- the old
        #             positional call mis-bound all five of them.
        #   WHAT WAS WRONG
        #     * `Sample.__init__` (sample.py:34-35) declares
        #         (a, b, c, alpha, beta, gamma, u=None, v=None, mosaic=None,
        #          vmosaic=None, direct=1, width=None, height=None, depth=None,
        #          shape='rectangular', distance=None)
        #       but this call passed the five optionals positionally as
        #         (..., crystal['mosaic'], crystal['vmosaic'],
        #                    crystal['dir'],  crystal['u'], crystal['v'])
        #       i.e. a 5-way rotation of the parameter list:
        #         mosaic -> u,   vmosaic -> v,   dir -> mosaic,
        #         u -> vmosaic,  v -> direct
        #     * with the defaults that material.py:118-127 installs into the dict
        #       (mosaic/vmosaic/u/v = None, dir = 1) the old code produced
        #         self._u     = [1,0,0]     (u=None -> Sample's default, right by luck)
        #         self._v     = [0,0,1]     (v=None -> Sample's default, right by luck)
        #         self.mosaic = 1           (from dir=1 -- a mosaic never asked for)
        #         self.vmosaic unset        (from u=None)
        #         self.dir    = None        (from v=None)   <-- should be 1
        #       so any `.dir` reader saw None instead of 1 (`Material.__repr__` itself
        #       formats only `name`), and callers that DID supply options were wrong in all five:
        #       `crystal['u']` landed in vmosaic, `crystal['mosaic']` landed in u (a
        #       float scalar handed to `Umatrix`/`gram_schmidt`), `crystal['dir']` in
        #       mosaic, `crystal['v']` in direct.
        #   WHY THIS IS THE CORRECT MINIMAL FIX
        #     * keyword arguments bind by name, so the mapping is correct by construction
        #       and cannot silently rot again if `Sample.__init__` is ever reordered.
        #       The one place that already builds a `Sample` correctly uses this same
        #       style (tas_spectr.py:126-128).
        #     * only the five optional arguments are named; the six lattice arguments stay
        #       positional and untouched, as do all `crystal[...]` lookups.
        #   OBSERVABLE BEHAVIOUR CHANGE
        #     * `Material(...).dir` now equals `crystal['dir']` (default 1) rather than
        #       `None`.
        #     * `Material(...).mosaic` is now left UNSET when `crystal['mosaic']` is
        #       absent, because `Sample.__init__` only assigns it for non-None
        #       (sample.py:47-48). The old value `1` was an artifact of the rotation (it
        #       came from `crystal['dir']`), not an intended default. Every in-repo reader
        #       of `sample.mosaic` already guards with `hasattr` (tas_spectr.py:1085), and
        #       `Sample.__repr__` reads it with `getattr(..., None)` (sample.py:67-68).
        #     * note `Material` is never constructed inside this repo -- it is public API
        #       exported from `inspy`/`inspy.crystal` (README's `inspy.Material(dict)`),
        #       so nothing here depended on the old rotation.
        #   NOT CHANGED
        #     * `Sample.__init__` itself, `MagneticUnitCell.__init__`'s super() call
        #       (material.py:46, already correct -- six positional lattice args only), and
        #       the pre-pass at material.py:118-127 that installs the dict defaults. That
        #       pre-pass's own missing-default issues (`massNorm`, `lattice` with no
        #       `else`) are tracked separately under §2.5's "related issues".
        #     * pre-existing and out of scope: `Sample` assigns the plain attribute
        #       `self.dir` (sample.py:52) while also declaring a `direct` property backed
        #       by `self._dir` (sample.py:109-115) that `__init__` never assigns, so
        #       `sample.direct` raises AttributeError. Nothing in this repo reads it.
        super(Material, self).__init__(a, b, c, alpha, beta, gamma,
                                       u=crystal['u'], v=crystal['v'],
                                       mosaic=crystal['mosaic'], vmosaic=crystal['vmosaic'],
                                       direct=crystal['dir'])

    def __repr__(self):
        return "Material('{0}')".format(self.name)

    @property
    def total_scattering_cross_section(self):
        r"""Returns total scattering cross-section of unit cell
        """
        total = 0
        for atom in self.atoms:
            total += (atom.coh_xs + atom.inc_xs)
        return total

    def N_atoms(self, mass):
        #Number of formula units (cells) in the given sample mass: N_A*mass/muCell.

        return const.N_A * mass / self.muCell

    def calc_optimal_thickness(self, energy=25.3, transmission=1 / np.exp(1)):
        #Calculates the optimal sample thickess to avoid problems with
        #extinction, multiple coherent scattering and absorption.


        sigma_coh = np.sum([atom.occupancy * atom.coh_xs for atom in self.atoms])
        sigma_inc = np.sum([atom.occupancy * atom.inc_xs for atom in self.atoms])
        sigma_abs = np.sum([atom.occupancy * atom.abs_xs for atom in self.atoms])

        sigma_T = (sigma_coh + sigma_inc + sigma_abs * np.sqrt(25.3 / energy)) / self.volume

        return -np.log(transmission) / sigma_T

    def calc_incoh_elas_xs(self, mass=None):
        #Calculates the incoherent elastic cross section.


        INC_XS = 0
        for atom in self.atoms:
            INC_XS += atom.inc_xs * np.exp(-8 * np.pi ** 2 * atom.Uiso * np.sin(
                np.deg2rad(self.get_two_theta(atom.pos, self.wavelength) / 2.)) ** 2 / self.wavelength ** 2)

        if mass is not None:
            return self.N_atoms(mass) / (4 * np.pi) * INC_XS
        else:
            return INC_XS / (4 * np.pi)
