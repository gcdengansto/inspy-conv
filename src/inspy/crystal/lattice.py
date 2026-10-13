# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

# Handles lattice geometries to find rotations and transformations

import numpy as np

class Lattice(object):
    #Class to describe a generic lattice system defined by lattice six parameters.


    def __init__(self, a, b, c, alpha, beta, gamma):
        self.abc = [a, b, c]
        self.abg = [alpha, beta, gamma]

    def __repr__(self):
        return "Lattice({0}, {1}, {2}, {3}, {4}, {5})".format(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    @property
    def a(self):
 
        return self.abc[0]

    @a.setter
    def a(self, a):
        self.abc[0] = a

    @property
    def b(self):

        return self.abc[1]

    @b.setter
    def b(self, b):
        self.abc[1] = b

    @property
    def c(self):

        return self.abc[2]

    @c.setter
    def c(self, c):
        self.abc[2] = c

    @property
    def alpha(self):

        return self.abg[0]

    @alpha.setter
    def alpha(self, alpha):
        self.abg[0] = alpha

    @property
    def beta(self):

        return self.abg[1]

    @beta.setter
    def beta(self, beta):
        self.abg[1] = beta

    @property
    def gamma(self):

        return self.abg[2]

    @gamma.setter
    def gamma(self, gamma):
        self.abg[2] = gamma

    @property
    def alpha_rad(self):

        return self.abg_rad[0]

    @property
    def beta_rad(self):

        return self.abg_rad[1]

    @property
    def gamma_rad(self):

        return self.abg_rad[2]

    @property
    def astar(self):

        return self.b * self.c * np.sin(self.alpha_rad) / self.volume * 2 * np.pi

    @property
    def bstar(self):

        return self.a * self.c * np.sin(self.beta_rad) / self.volume * 2 * np.pi

    @property
    def cstar(self):

        return self.a * self.b * np.sin(self.gamma_rad) / self.volume * 2 * np.pi

    @property
    def alphastar_rad(self):

        return np.arccos((np.cos(self.beta_rad) * np.cos(self.gamma_rad) -
                          np.cos(self.alpha_rad)) /
                         (np.sin(self.beta_rad) * np.sin(self.gamma_rad)))

    @property
    def betastar_rad(self):

        return np.arccos((np.cos(self.alpha_rad) *
                          np.cos(self.gamma_rad) -
                          np.cos(self.beta_rad)) /
                         (np.sin(self.alpha_rad) * np.sin(self.gamma_rad)))

    @property
    def gammastar_rad(self):

        return np.arccos((np.cos(self.alpha_rad) * np.cos(self.beta_rad) -
                          np.cos(self.gamma_rad)) /
                         (np.sin(self.alpha_rad) * np.sin(self.beta_rad)))

    @property
    def alphastar(self):

        return np.rad2deg(self.alphastar_rad)

    @property
    def betastar(self):

        return np.rad2deg(self.betastar_rad)

    @property
    def gammastar(self):

        return np.rad2deg(self.gammastar_rad)

    @property
    def reciprocal_abc(self):

        return [self.astar, self.bstar, self.cstar]

    @property
    def reciprocal_abg(self):

        return [self.alphastar, self.betastar, self.gammastar]

    @property
    def reciprocal_abg_rad(self):

        return [self.alphastar_rad, self.betastar_rad, self.gammastar_rad]

    @property
    def abg_rad(self):

        return np.deg2rad(self.abg)

    @property
    def lattice_type(self):

        if len(np.unique(self.abc)) == 3 and len(np.unique(self.abg)) == 3:
            return 'triclinic'
        elif len(np.unique(self.abc)) == 3 and self.abg[1] != 90 and np.all(np.array(self.abg)[:3:2] == 90):
            return 'monoclinic'
        elif len(np.unique(self.abc)) == 3 and np.all(np.array(self.abg) == 90):
            return 'orthorhombic'
        elif len(np.unique(self.abc)) == 1 and len(np.unique(self.abg)) == 1 and np.all(
                        np.array(self.abg) < 120) and np.all(np.array(self.abg) != 90):
            return 'rhombohedral'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg) == 90):
            return 'tetragonal'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg)[0:2] == 90) and \
                        self.abg[2] == 120:
            return 'hexagonal'
        elif len(np.unique(self.abc)) == 1 and np.all(np.array(self.abg) == 90):
            return 'cubic'
        else:
            raise ValueError('Provided lattice constants and angles do not resolve to a valid Bravais lattice')

    @property
    def volume(self):

        return np.sqrt(np.linalg.det(self.G))

    @property
    def reciprocal_volume(self):

        return np.sqrt(np.linalg.det(self.Gstar))

    @property
    def G(self):
        #Metric tensor of the real space lattice


        a, b, c = self.abc
        alpha, beta, gamma = self.abg_rad

        return np.array([[a ** 2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                          [a * b * np.cos(gamma), b ** 2, b * c * np.cos(alpha)],
                          [a * c * np.cos(beta), b * c * np.cos(alpha), c ** 2]], dtype=float)

    @property
    def Gstar(self):
        #Metric tensor of the reciprocal lattice

        return np.linalg.inv(self.G) * 4 * np.pi ** 2

    @property
    def Bmatrix(self):
        #Cartesian basis matrix in reciprocal units such that Bmatrix.T*Bmatrix = Gstar
        #20261010 Bug #2.3: header said "Bmatrix*Bmatrix.T = Gstar" -> "Bmatrix.T*Bmatrix".
        #   WHAT WAS WRONG
        #     * This layout puts the direct-a axis in COLUMN 0: B[:,0] = [a*, 0, 0].
        #       Hence (B.T @ B)[0,0] = a*2, but (B @ B.T)[0,0] = row0.row0 =
        #       a*2 + b*2*cos^2(gammastar) + c*2*cos^2(betastar) > a*2 in general.
        #       Gstar = 4*pi^2 * inv(G) has [0,0] = a*2 exactly, so only B.T @ B
        #       equals Gstar. (Verified symbolically above and numerically on
        #       a=5, b=6, c=7, alpha=80, beta=90, gamma=100.)
        #   NOT CHANGED
        #     * code on this property -- this line is a comment only.
        return np.array([[self.astar, self.bstar * np.cos(self.gammastar_rad),  self.cstar * np.cos(self.betastar_rad)],
                          [0, self.bstar * np.sin(self.gammastar_rad), -self.cstar * np.sin(self.betastar_rad) * np.cos(self.alpha_rad)],
                          #20261010 Bug #2.3: [2][2] must use sin(alpha) -- the DIRECT angle, not alphastar.
                          #   WHAT WAS WRONG
                          #     * `np.sin(self.alphastar_rad)` gave X = c* * sin(betastar) * sin(alphastar),
                          #       whereas the Busing-Levy third element is X = c* * sin(betastar) * sin(alpha).
                          #     * Consequence: ‖col 2‖² = c*²(cos²β* + sin²β*·cos²α) + X² ≠ c*², i.e.
                          #       only (Bᵀ·B)[2,2] ≠ G*[2,2]. (The [1,2] cross-term already used
                          #       the direct α via row 1 -- unchanged by this fix -- and already
                          #       equalled b*c*·cosα*.)
                          #   WHY THIS IS THE CORRECT MINIMAL FIX
                          #     * the three diagonal entries of Bᵀ·B pin X down uniquely:
                          #         [0,0] = a*² (free), [1,1] = b*² (free),
                          #         [2,2] = c*²( cos²β* + sin²β*·cos²A ) + X² = c*²
                          #           ⇒ X = c*·sin(β*)·sin(A)
                          #       and [1,2] = b*c*( cosγ*·cosβ* − sinγ*·sinβ*·cosA ) = b*c*·cosα*
                          #       forces A = α (the direct angle). Symmetric entry [0,1] = a*b*·cosγ*
                          #       is already satisfied independently of A, so this is the only
                          #       condition left -- and it needs sin(alpha).
                          #     * numeric cross-check (a=5,b=6,c=7,α=80°,β=90°,γ=100°;
                          #       α*=100.15°, β*=91.78°, γ*=79.84°): with A=α the three
                          #       diagonal entries and [1,2] reproduce G* to rounding; with
                          #       A=α* they do not.
                          #   IMPACT BEFORE THE FIX
                          #     * exactly zero for the listed alpha = 90 deg systems, because
                          #       those also force alpha* = 90 deg (so sin = sin* = 1):
                          #       cubic/tetragonal/orthorhombic -- all angles 90 deg;
                          #       hexagonal -- alpha = beta = 90 deg, gamma = 120 deg;
                          #       monoclinic-unique-b -- alpha = gamma = 90 deg.
                          #       That is why the ResLib comparison passed.
                          #     * NOT zero for a triclinic cell that merely happens to have
                          #       alpha = 90 deg with beta,gamma != 90: then cos(alpha*) =
                          #       cos(beta)cos(gamma)/(sin(beta)sin(gamma)) != 0, so
                          #       sin(alpha*) != 1 and X is still wrong.
                          #     * wrong for triclinic cells and rhombohedral cells in native
                          #       axes (the hex-axis setting has alpha=beta=90 -> zero-impact);
                          #       feeds `UBmatrix` (sample.py:129) -> every hkl->Q conversion.
                          #   NOT CHANGED
                          #     * the other five entries (incl. [1][2], which already uses the
                          #       direct `alpha_rad` -- the two are a matched pair), the
                          #       `dtype=float`, `G`/`Gstar`, and all callers.
                          [0, 0, self.cstar * np.sin(self.betastar_rad) * np.sin(self.alpha_rad)]], dtype=float)

    def get_d_spacing(self, hkl):
        #Returns the d-spacing of a given reciprocal lattice vector.

        hkl = np.array(hkl)

        return float(1 / np.sqrt(np.dot(np.dot(hkl, self.Gstar / 4 / np.pi ** 2), hkl)))

    def get_angle_between_planes(self, v1, v2):
        #Returns the angle :math:`\phi` between two reciprocal lattice vectors

        v1, v2 = np.array(v1), np.array(v2)

        return float(np.rad2deg(np.arccos(np.inner(np.inner(v1, self.Gstar), v2) /
                                          np.sqrt(np.inner(np.inner(v1, self.Gstar), v1)) /
                                          np.sqrt(np.inner(np.inner(v2, self.Gstar), v2)))))

    def get_two_theta(self, hkl, wavelength):
        #Returns the detector angle  for a given reciprocal lattice vector and incident wavelength.

        return 2 * np.rad2deg(np.arcsin(wavelength / 2 /
                                        self.get_d_spacing(hkl)))

    def get_q(self, hkl):
        #Returns the magnitude of *Q* for a given reciprocal lattice
 
        return 2 * np.pi / self.get_d_spacing(hkl)
