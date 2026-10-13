# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

"""
Useful constants for neutron scattering calculations, including:

* ``magnetic_ion_j()`` :        Magnetic Ion j-values
* ``periodic_table()`` :        Periodic table values
* ``scattering_lengths()`` :    Neutron scattering lengths
* ``symmetry()`` :              Space group information
* ``JOULES_TO_MEV`` :           Joules-to-meV conversion factor
* ``BOLTZMANN_IN_MEV_K`` :      Boltzmann constant in meV/K
* ``N_A`` :                     Avogadro constant
* ``neutron_mass`` :            Mass of a neutron in grams
* ``e`` :                       Electric charge of an electron in Coulombs
"""


import json
import os

def magnetic_ion_j():
    #Loads j values for Magnetic ions.
    #Returns:  ion_j values : dict 
    with open(os.path.join(os.path.dirname(__file__),
                           "database/magnetic_form_factors.json"), 'r') as infile:
        return json.load(infile)

def periodic_table():
    #Loads periodic table database. mass, and long-form name.
    #Returns atom information dict, including mass, atomic number, density, mass, and name
    with open(os.path.join(os.path.dirname(__file__),
                           "database/periodic_table.json"), 'r') as infile:
        return json.load(infile)

def scattering_lengths():
    #Loads neutron scattering lengths.
    #Returns: scattering_lengths : dict

    with open(os.path.join(os.path.dirname(__file__),
                           "database/scattering_lengths.json"), 'r') as infile:
        return json.load(infile)

def symmetry():
    #Loads crystal lattice space groups.
    #Returns    lattice_space_groups : dict Database of 230 crystal lattice space groups and their generators
    with open(os.path.join(os.path.dirname(__file__),
                           "database/symmetry.json"), 'r') as infile:
        return json.load(infile)


JOULES_TO_MEV      = 1. / 1.6021766208e-19 * 1.e3  # Joules to meV
BOLTZMANN_IN_MEV_K = 8.6173303e-05 * 1.e3          # Boltzmann constant in meV/K
N_A                = 6.022140857e+23
neutron_mass       = 1.674927211e-24               # mass of a neutron in grams
hbar               = 1.054571628e-34               # hbar in m2 kg / s
e                  = 1.602176487e-19               # coulombs
