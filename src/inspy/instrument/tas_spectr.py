# -*- coding: utf-8 -*-
# This code is modified from neutronpy

# Define a TAS instrument for resolution calculations

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag as blkdiag

import plotly.offline as plyplt
import plotly.subplots as plysub 
import plotly.graph_objects as go

from ..crystal import Sample
from ..energy  import Energy
from .exceptions import ScatteringTriangleError
from .mono import Mono
from .ana  import Ana
from .tools import get_tau, _cleanargs, _modvec, _scalar, _star, _voigt, get_bragg_widths


CONVERT1 = np.pi / 60. / 180. / np.sqrt(8 * np.log(2))
CONVERT2 = 2.07242  # ℏ²/(2m) in meV·Å² units

class SimpleComp:
    """Simple component class for TAS instrument parts."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        

class TripleAxisSpectr:
    """
    An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.
    """
    def __init__(self,
                    # --- spectrometer configuration ---
                    efixed=14.7,
                    method=1,
                    infin=-1,
                    moncor=1,
                    instr_type='tas',
                    description_string='No description yet',
                    # --- components (None → built from sub-parameters below) ---
                    mono=None,
                    sample=None,
                    ana=None,
                    guide=None,
                    monitor=None,
                    detector=None,
                    # --- monochromator sub-parameters ---
                    mono_tau='PG(002)',
                    mono_mosaic=30,
                    mono_vmosaic=30,
                    mono_height=20,
                    mono_width=20,
                    mono_depth=0.2,
                    mono_rh=150,
                    mono_rv=150,
                    mono_direct=-1,
                    # --- analyzer sub-parameters ---
                    ana_tau='PG(002)',
                    ana_mosaic=30,
                    ana_vmosaic=30,
                    ana_height=20,
                    ana_width=20,
                    ana_depth=0.2,
                    ana_rh=150,
                    ana_rv=150,
                    ana_direct=-1,
                    ana_horifoc=-1,
                    #ana_KQ=0.1,  # extinction coefficient of the crystal, only needed if depth is specified
                    # --- sample sub-parameters ---
                    sample_a=6,
                    sample_b=7,
                    sample_c=8,
                    sample_alpha=90,
                    sample_beta=90,
                    sample_gamma=90,
                    sample_u=None,
                    sample_v=None,
                    sample_width=5,
                    sample_height=5,
                    sample_depth=5,
                    # --- guide / monitor / detector sub-parameters ---
                    guide_width=5,
                    guide_height=18,
                    monitor_width=5,
                    monitor_height=18,
                    detector_width=2.5,
                    detector_height=15,
                    # --- collimation ---
                    hcol=None,
                    vcol=None,
                    # --- arm distances ---
                    L0=100.,
                    L1=100.,
                    L2=100.,
                    L3=100.,
                    L1mon=100.,
                    # --- scan points ---
                    hkle=None,
                    **kwargs):

            # --- sample defaults ---
            if sample_u is None:
                sample_u = [1, 0, 0]
            if sample_v is None:
                sample_v = [0, 1, 0]

            # --- collimation defaults ---
            if hcol is None:
                hcol = [40, 40, 40, 40]
            if vcol is None:
                vcol = [120, 120, 120, 120]

            # --- HKLE default ---
            if hkle is None:
                hkle = [[0, 0, 1], [0, 0, 1.5], [0, 0, 2], [0, 1, 2]]

            # --- build components from sub-parameters if not supplied directly ---
            if sample is None:
                sample = Sample(sample_a, sample_b, sample_c,
                                sample_alpha, sample_beta, sample_gamma, width=sample_width, height=sample_height, depth=sample_depth,
                                u=sample_u, v=sample_v)

            if mono is None:
                mono = Mono(tau=mono_tau, mosaic=mono_mosaic, vmosaic=mono_vmosaic,
                            height=mono_height, width=mono_width, depth=mono_depth,
                            rh=mono_rh, rv=mono_rv, direct=mono_direct)

            if ana is None:
                ana = Ana(tau=ana_tau, mosaic=ana_mosaic, vmosaic=ana_vmosaic,
                        height=ana_height, width=ana_width, depth=ana_depth,
                        rh=ana_rh, rv=ana_rv, direct=ana_direct, horifoc=ana_horifoc
                        #, KQ=ana_KQ
                        )

            if guide is None:
                guide = SimpleComp(guide_width, guide_height)

            if monitor is None:
                monitor = SimpleComp(monitor_width, monitor_height)

            if detector is None:
                detector = SimpleComp(detector_width, detector_height)

            # --- set instance attributes ---
            self.instr_type        = instr_type
            self.method            = method
            self.infin             = infin
            self.moncor            = moncor
            self.description_string = description_string
            self.efixed            = efixed
            self.mono              = mono
            self.ana               = ana
            self.sample            = sample
            self.orient1           = np.array(sample.u)
            self.orient2           = np.array(sample.v)
            self.guide             = guide
            self.monitor           = monitor
            self.detector          = detector
            self.hcol              = np.array(hcol)
            self.vcol              = np.array(vcol)
            self.HKLE              = hkle
            # --- arm distances ---
            self.arms              = np.array([L0, L1, L2, L3, L1mon ])
            # --- any extra keyword arguments ---
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __repr__(self):
        return f"Instrument('tas', efixed={self.efixed})"

    def __eq__(self, right):
        if not isinstance(right, TripleAxisSpectr):
            return False
            
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if self_parent_keys != right_parent_keys:
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            try:
                if not np.array_equal(value, right_parent_val, equal_nan=True):
                    return False
            except (TypeError, ValueError):
                if value != right_parent_val:
                    return False
        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    @property
    def mono(self):
        return self._mono

    @mono.setter
    def mono(self, value):
        self._mono = value

    @property
    def ana(self):
        return self._ana

    @ana.setter
    def ana(self, value):
        self._ana = value

    @property
    def method(self):
        """
        Selects the computation method.
        method=0 or left undefined: Cooper-Nathans calculation
        method=1: a Popovici calculation
        """
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def moncor(self):
        """
        Selects the type of normalization used to calculate R0
        moncor=1 or left undefined: R0 is calculated in normalization to monitor counts.
        1/ki monitor efficiency correction is included automatically. 
        moncor=0: normalize R0 to source flux.
        """
        return self._moncor

    @moncor.setter
    def moncor(self, value):
        self._moncor = value

    @property
    def hcol(self):
        """
        The horizontal Soller collimations in minutes of arc (FWHM beam divergence) 
        starting from the in-pile collimator. In case of a horizontally-focusing 
        analyzer hcol[2] is the angular size of the analyzer, as seen from the sample 
        position. If the beam divergence is limited by a neutron guide, the 
        corresponding element of hcol is the negative of the guide's m-value. 
        For example, for a 58-Ni guide (m = 1.2) before the monochromator, 
        hcol[0] should be -1.2.
        """
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = np.array(value)

    @property
    def vcol(self):
        """
        The vertical Soller collimations in minutes of arc (FWHM beam divergence) 
        starting from the in-pile collimator. In case of a vertically-focusing 
        analyzer vcol[2] is the angular size of the analyzer, as seen from the 
        sample position. If the beam divergence is limited by a neutron guide, 
        the corresponding element of vcol is the negative of the guide's m-value. 
        For example, for a 58-Ni guide (m = 1.2) before the monochromator, 
        vcol[0] should be -1.2.
        """
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = np.array(value)

    @property
    def arms(self):
        """
        Distances between the source and monochromator, monochromator
        and sample, sample and analyzer, analyzer and detector, and
        monochromator and monitor, respectively. The 5th element is only needed if moncor=1
        """
        return self._arms

    @arms.setter
    def arms(self, value):
        self._arms = np.array(value)

    @property
    def efixed(self):
        """The fixed incident or final neutron energy, in meV."""
        return self._efixed

    @efixed.setter
    def efixed(self, value):
        self._efixed = value

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

    @property
    def orient1(self):
        """Miller indexes of the first reciprocal-space orienting vector for the S coordinate system"""
        return self._sample.u

    @orient1.setter
    def orient1(self, value):
        self._sample.u = np.array(value)

    @property
    def orient2(self):
        """Miller indexes of the second reciprocal-space orienting vector the S coordinate system"""
        return self._sample.v

    @orient2.setter
    def orient2(self, value):
        self._sample.v = np.array(value)

    @property
    def infin(self):
        """
        A flag set to -1 or left unassigned if the final energy is fixed, 
        or set to +1 in a fixed-incident setup.
        """
        return self._infin

    @infin.setter
    def infin(self, value):
        self._infin = value

    @property
    def guide(self):
        """A structure that describes the source"""
        return self._guide

    @guide.setter
    def guide(self, value):
        self._guide = value

    @property
    def detector(self):
        """A structure that describes the detector"""
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def monitor(self):
        """A structure that describes the monitor"""
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def Smooth(self):
        """
        Defines the smoothing parameters. Leave this field unassigned if you do not want 
        this correction done.
        * Smooth.E is the smoothing FWHM in energy (meV). A small number means 
          "no smoothing along this direction".
        * Smooth.X is the smoothing FWHM along the first orienting vector (x0 axis) in Ų⁻¹.
        * Smooth.Y is the smoothing FWHM along the y axis in Ų⁻¹.
        * Smooth.Z is the smoothing FWHM along the vertical direction in Ų⁻¹.
        """
        return self._Smooth

    @Smooth.setter
    def Smooth(self, value):
        self._Smooth = value

    def get_lattice(self):
        """
        Extracts lattice parameters from the sample and returns the direct and reciprocal
        lattice objects (Sample / Lattice), built with tools._star as in the ResLib port.
        Returns the direct and reciprocal lattice sample classes
        """
        lattice = Sample(self.sample.a,  #20261010 §4.1 Fix: angles passed in DEGREES (Lattice stores abg in degrees and derives abg_rad itself, lattice.py:156-158); the old np.deg2rad() here double-converted them for any non-90° cell
                         self.sample.b,
                         self.sample.c,
                         self.sample.alpha,
                         self.sample.beta,
                         self.sample.gamma)
        rlattice = _star(lattice)[-1]

        return [lattice, rlattice]

    def _StandardSystem(self):
        """
        Returns rotation matrices to calculate resolution in the sample view 
        instead of the instrument view
        """
        lattice, rlattice = self.get_lattice()

        orient1 = self.orient1
        orient2 = self.orient2

        modx = _modvec(orient1, rlattice)
        x = orient1 / modx

        proj = _scalar(orient2, x, rlattice)
        y = orient2 - x * proj
        mody = _modvec(y, rlattice)

        if np.any(mody <= 0):
            raise ScatteringTriangleError('Orienting vectors are collinear')

        y /= mody

        z = np.array([x[1] * y[2] - y[1] * x[2],
                      x[2] * y[0] - y[2] * x[0],
                      -x[1] * y[0] + y[1] * x[0]], dtype=np.float64)

        proj = _scalar(z, x, rlattice)
        z -= x * proj

        proj = _scalar(z, y, rlattice)
        z -= y * proj

        modz = _modvec(z, rlattice)
        z /= modz

        return [x, y, z, lattice, rlattice]

    def _ki_kf(self, W):
        """
        Returns (ki, kf) arrays for energy transfers W,
        given the fixed-energy mode (self.infin) and self.efixed.
        """
        W = np.atleast_1d(W)
        efixed = self.efixed
        if self.infin > 0:          # fixed ki
            ei = np.full_like(W, efixed, dtype=float)
            ef = efixed - W
        else:                       # fixed kf (default)
            ei = efixed + W
            ef = np.full_like(W, efixed, dtype=float)
        ki = np.sqrt(ei / CONVERT2)
        kf = np.sqrt(ef / CONVERT2)
        return ki, kf

    def get_angles_and_Q(self, hkle):
        """
        Returns the Triple Axis Spectrometer angles and Q-vector given position in reciprocal space
        
        Args:
            hkle (list): Array of the scattering vector and energy transfer [H, K, L, W]
        
        Returns:
            list: [A, Q] where A is angles (A1--A6 in a list of floats) and Q is ndarray
        """
        h, k, l, w = hkle
        
        # Compute angles
        FX = 2 if getattr(self, "infin", None) != 1 else 1

        kfix = Energy(energy=self.efixed).wavevector
        #f = 0.4826  # f converts from energy units into k^2, f=0.4826 for meV, actually  f = 1 / CONVERT2
        
        # Kinematical equations
        ki = np.sqrt(kfix ** 2 + (FX - 1 ) * w/CONVERT2)
        kf = np.sqrt(kfix ** 2 - (2  - FX) * w/CONVERT2)

        # Compute the transversal Q component, and A3 (sample rotation)
        # from McStas templateTAS.instr and TAS MAD ILL
        a = np.array([self.sample.a, self.sample.b, self.sample.c]) / (2 * np.pi)
        alpha = np.deg2rad([self.sample.alpha, self.sample.beta, self.sample.gamma])
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        
        cc = np.sum(cosa * cosa)
        cc = 1 + 2 * np.prod(cosa) - cc
        cc = np.sqrt(cc)
        
        b = sina / (a * cc)
        c1 = np.roll(cosa[np.newaxis].T, -1)
        c2 = np.roll(c1, -1)
        s1 = np.roll(sina[np.newaxis].T, -1)
        s2 = np.roll(s1, -1)
        
        # Use flatten to change the array from (3,1) to 1D array (3)
        cosb = ((c1 * c2 - cosa[np.newaxis].T) / (s1 * s2)).flatten()
        sinb = np.sqrt(1 - cosb * cosb)

        bb = np.array([[b[0], 0, 0],
                       [b[1] * cosb[2], b[1] * sinb[2], 0],
                       [b[2] * cosb[1], -b[2] * sinb[1] * cosa[0], 1 / a[2]]], 
                      dtype=np.float64)
        bb = bb.T

        aspv = np.hstack((self.orient1[np.newaxis].T, self.orient2[np.newaxis].T))

        vv = np.zeros((3, 3))
        vv[0:2, :] = np.transpose(np.dot(bb, aspv))

        for m in range(2, 0, -1):
            vt = (np.roll(np.roll(vv, -1, axis=0), -1, axis=1) * 
                  np.roll(np.roll(vv, -2, axis=0), -2, axis=1) - 
                  np.roll(np.roll(vv, -1, axis=0), -2, axis=1) * 
                  np.roll(np.roll(vv, -2, axis=0), -1, axis=1))
            vv[m, :] = vt[m, :]

        c = np.sqrt(np.sum(vv * vv, axis=0))
        vv = vv / np.tile(c, (3, 1))
        s = vv.T * bb

        qt = np.squeeze(np.dot(np.array([h, k, l]).T, s.T))
        qs = np.sum(qt ** 2)
        Q = np.sqrt(qs)

        sm = self.mono.dir
        ss = self.sample.dir
        sa = self.ana.dir
        dm = 2 * np.pi / get_tau(self.mono.tau)
        da = 2 * np.pi / get_tau(self.ana.tau)
        
        # Theta angles for analyser and monochromator
        thetaa = sa * np.arcsin(np.pi / (da * kf))
        thetam = sm * np.arcsin(np.pi / (dm * ki))

        if np.abs(np.pi / (dm * ki)) > 1.0:
            raise ScatteringTriangleError('Monochromator Bragg condition cannot be satisfied.')
        if np.abs(np.pi / (da * kf)) > 1.0:
            raise ScatteringTriangleError('Analyzer Bragg condition cannot be satisfied.')
        
        # Scattering angle from sample
        thetas = ss * 0.5 * np.arccos((ki ** 2 + kf ** 2 - Q ** 2) / (2 * ki * kf))

        A3 = (-np.arctan2(qt[1], qt[0]) - 
              np.arccos((np.dot(kf, kf) - np.dot(Q, Q) - np.dot(ki, ki)) / 
                       (-2 * np.dot(Q, ki))))
        A3 = ss * A3

        A1 = thetam
        A2 = 2 * A1
        A4 = 2 * thetas
        A5 = thetaa
        A6 = 2 * A5
        
        temp_a = np.array([A1, A2, A3, A4, A5, A6], dtype=np.float32)
        A = np.squeeze(np.rad2deg(temp_a))

        return [A, Q]

    def get_spec_angles(self, hkle=None):
        """
        Calculate the angles from the HKLE with the current Ef
        Calculate shaft angles given momentum transfer H, K, L, energy transfer W, and
        spectrometer and sample parameters held on self. 
        Based on ResLib 3.4c, originally authored by A. Zheludev
        
        Args:
            hkle (list, optional): [H, K, L, W] values. Defaults to None.
            
        Returns:
            list: [M1, M2, S1, S2, A1, A2, Q] angles and momentum transfer
        """
        if hkle is None:
            print("Error: No hkle is given!")
            return None

        #20261010: L6 Fix: validate BEFORE unpacking -- the old order (unpack first) made
        #   the len==4 check unreachable: a wrong-length hkle died at the unpack with a
        #   raw ValueError instead of the intended message. NOTE: local fix only -- 7 other methods still unpack before validating (v3 §9).
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
        H, K, L, W = hkle
            
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")

        mono = self.mono
        ana = self.ana
        
        taum = get_tau(mono.tau)
        taua = get_tau(ana.tau)

        qx, qy, qz, Q = self.R2S(H, K, L)

        curdir = [mono.dir, self.sample.dir, ana.dir]

        ki, kf = self._ki_kf(W)
        
        M1, M2, A1, A2, S1, S2 = [], [], [], [], [], []
        
        # Calculate one by one but not as arrays
        for  kki, kkf, qq, qqx, qqy in zip(ki, kf, Q, qx, qy):
            # Monochromator angles
            if np.abs(taum / (2.0 * kki)) > 1:
                M1.append(np.nan)
                M2.append(np.nan)
                print("Warning: Mono angle is wrong.")
            else:
                M1.append(np.arcsin(taum / (2.0 * kki)) * curdir[0])
                M2.append(2 * np.arcsin(taum / (2.0 * kki)) * curdir[0])

            # Analyzer angles
            if np.abs(taua / (2.0 * kkf)) > 1:
                A1.append(np.nan)
                A2.append(np.nan)
                print("Warning: Ana angle is wrong.")
            else:
                A1.append(np.arcsin(taua / (2.0 * kkf)) * curdir[2])
                A2.append(2 * np.arcsin(taua / (2.0 * kkf)) * curdir[2])

            # Sample angles
            temp_value = (kki**2 + kkf**2 - qq**2) / (2 * kki * kkf)
            if np.abs(temp_value) > 1.0:
                S2.append(np.nan)
                S1.append(np.nan)
                print("Warning: Scattering triangle cannot close.")
            else:
                ss2 = np.arccos(temp_value) * curdir[1]
                S2.append(ss2)
                phi = np.arctan2(-kkf * np.sin(ss2), kki - kkf * np.cos(ss2))
                psi = np.arctan2(qqy, qqx)
                S1.append(-psi + phi)

        #20261010: L6 Fix: keep float64 (MATLAB SpecWhere works in doubles) -- the old
        #   float32 cast threw away ~7 significant digits on the motor angles
        temp_ang = np.array([M1, M2, S1, S2, A1, A2], dtype=np.float64)
        M1, M2, S1, S2, A1, A2 = np.squeeze(np.rad2deg(temp_ang))  # change from radian to angle

        return [M1, M2, S1, S2, A1, A2, Q]

    def get_hkl_and_Q(self, M2, S1, S2, A2):
        """
        For given values of M2,S1,S2 and A2 spectrometer motors
        calculates the wave vector transfer (H, K, L), Q=|(H,K,L)|, E, and Ei and Ef
        
        Args:
            M2, S1, S2, A2 (arrays): Spectrometer motor angles
            
        Returns:
            list: [H, K, L, W, Q, Ei, Ef] arrays
        """
        length, M2, S1, S2, A2 = _cleanargs(M2, S1, S2, A2)

        mono = self.mono
        ana  = self.ana

        taum = get_tau(mono.tau)
        taua = get_tau(ana.tau)

        # Convert angles to radians for calculation
        M2_rad = np.deg2rad(M2)
        A2_rad = np.deg2rad(A2)
        S1_rad = np.deg2rad(S1)
        S2_rad = np.deg2rad(S2)

        ki = taum / np.sqrt(2 - 2 * np.cos(M2_rad))
        Ei = CONVERT2* ki**2
        kf = taua / np.sqrt(2 - 2 * np.cos(A2_rad))
        Ef = CONVERT2 * kf**2
        W = Ei - Ef         # W : energy transfer  
        Q = np.sqrt(ki**2 + kf**2 - 2 * ki * kf * np.cos(S2_rad))

        # Angle from ki to Q
        phi = np.arctan2(-kf * np.sin(S2_rad), ki - kf * np.cos(S2_rad))
        # Angle from first orienting vector to Q
        psi = -S1_rad + phi
        qx = Q * np.cos(psi)
        qy = Q * np.sin(psi)

        H, K, L, Q = self.S2R(qx, qy, 0)

        return [H, K, L, W, Q, Ei, Ef]

    def CalcResMatQ(self, Q, W):
        """
        For a momentum transfer Q and energy transfers W, calculates the Cooper-Nathans or Popovici
        resolution matrix RM and resolution prefactor R0 in the Q coordinate system.
        
        Args:
            Q (ndarray or list): Momentum transfer values
            W (float or list): Energy transfer values
            
        Returns:
            list: [R0, RM] Resolution pre-factor (R0) and resolution matrix (RM)
            
        Based on ResLib 3.4c, originally authored by A. Zheludev
        """

        length, Q, W = _cleanargs(Q, W)

        RM = np.zeros((length, 4, 4), dtype=np.float64)
        R0 = np.zeros(length, dtype=np.float64)
        RM_ = np.zeros((4, 4), dtype=np.float64)

        # Assign default values and decode parameters
        method = self.method
        moncor = self.moncor

        # Initialize default component dimensions
        mono   = self.mono
        ana    = self.ana
        sample = self.sample

        alpha = np.array(self.hcol) * CONVERT1
        beta = np.array(self.vcol) * CONVERT1

        etam = np.array(mono.mosaic) * CONVERT1
        etamv = np.copy(etam)

        etaa = np.array(ana.mosaic) * CONVERT1
        etaav = np.copy(etaa)

        if method == 1 or method == 'popovici':
            etamv = np.array(mono.vmosaic) * CONVERT1
            etaav = np.array(ana.vmosaic) * CONVERT1

        # Extract beam/guide dimensions
        beam = self.guide
        beamw = beam.width ** 2 / 12.
        beamh = beam.height ** 2 / 12.
        bshape = np.diag([beamw, beamh])

        # Extract monitor dimensions
        monitor = self.monitor
        monitorw = monitor.width ** 2 / 12.
        monitorh = monitor.height ** 2 / 12.
        monitorshape = np.diag([monitorw, monitorh])

        # Extract detector dimensions
        detector = self.detector
        detectorw = detector.width ** 2 / 12.
        detectorh = detector.height ** 2 / 12.
        dshape = np.diag([detectorw, detectorh])

        # Extract monochromator dimensions
        monow = mono.width ** 2 / 12.
        monoh = mono.height ** 2 / 12.
        monod = mono.depth ** 2 / 12.
        mshape = np.diag([monod, monow, monoh])

        # Extract analyzer dimensions
        anaw = ana.width ** 2 / 12.
        anah = ana.height ** 2 / 12.
        anad = ana.depth ** 2 / 12.
        ashape = np.diag([anad, anaw, anah])

        # Extract sample shape information
        if sample.shape_type == 'cylindrical':
            sshape_factor = 16.
        elif sample.shape_type == 'rectangular':
            sshape_factor = 12.
        else:
            raise ValueError(f"Unknown sample shape type: {sample.shape_type}. Supported types are 'cylindrical' and 'rectangular'.")
                

        _sshape = (np.diag([sample.depth, sample.width, sample.height]).astype(np.float64) ** 2 / sshape_factor)
        sshapes = np.repeat(_sshape[np.newaxis].reshape((1, 3, 3)), length, axis=0)

        # Extract arm lengths for Popovici method
        arms = self.arms
        L0, L1, L2, L3 = arms[:4]
        L1mon = np.copy(L1)
        if len(arms) > 4:
            L1mon = np.copy(arms[4])

        # Extract curvature radii
        monorv = mono.rv
        monorh = mono.rh
        anarv  = ana.rv
        anarh  = ana.rh

        horifoc = ana.horifoc
        if horifoc == 1:
            alpha[2] = alpha[2] * np.sqrt(8. * np.log(2.) / 12.)

        taum = get_tau(mono.tau)
        taua = get_tau(ana.tau)

        sm = self.mono.dir
        ss = self.sample.dir
        sa = self.ana.dir
 
        # get the correct ki and kf, depending on const ei or consta ef mode. 
        ki, kf = self._ki_kf(W)                                          # shape (length,)

        #old: risk to have invalid arcsin arguments if tau, efixed, or energy transfer range are not compatible.
        #thetam = np.arcsin(taum / (2. * ki)) * sm                        # (length,)
        #thetaa = np.arcsin(taua / (2. * kf)) * sa                        # (length,)

        #fix: check arcsin arguments before calculating angles, to avoid invalid values and provide informative error messages.
        arcsin_arg_m = taum / (2. * ki)
        arcsin_arg_a = taua / (2. * kf)

        if np.any(np.abs(arcsin_arg_m) > 1.0):
            raise ScatteringTriangleError(
                'Monochromator Bragg condition cannot be satisfied. '
                'Check tau, efixed, or energy transfer range.')

        if np.any(np.abs(arcsin_arg_a) > 1.0):
            raise ScatteringTriangleError(
                'Analyzer Bragg condition cannot be satisfied. '
                'Check tau, efixed, or energy transfer range.')

        thetam = np.arcsin(arcsin_arg_m) * sm
        thetaa = np.arcsin(arcsin_arg_a) * sa
 
        cos_arg = (ki ** 2 + kf ** 2 - Q ** 2) / (2. * ki * kf)
        if np.any(np.abs(cos_arg) > 1.0):
            raise ScatteringTriangleError(
                'KI, KF, Q triangle will not close. Change the value of KFIX, FX, QH, QK or QL.')
        s2theta = np.arccos(cos_arg) * ss                                 # (length,)
 
        thetas = s2theta / 2.                                             # (length,)
        phi    = np.arctan2(-kf * np.sin(s2theta),
                             ki - kf * np.cos(s2theta))                  # (length,)
 
        # --- guide divergence correction, vectorised over 4 collimators ---
        alpha_temp = np.tile(alpha, (length, 1))                          # (length, 4)
        beta_temp  = np.tile(beta,  (length, 1))                          # (length, 4)
 
        k_arms = np.column_stack([ki, ki, kf, kf])                        # (length, 4)
        guide_factor = 0.1 * 60. * (2. * np.pi / k_arms) / 0.427 / np.sqrt(3.)
 
        neg_a = alpha_temp < 0
        neg_b = beta_temp  < 0
        alpha_temp = np.where(neg_a, -alpha_temp * guide_factor, alpha_temp)
        beta_temp  = np.where(neg_b, -beta_temp  * guide_factor, beta_temp)
 
        # --- sample-shape rotation for all points at once ---
        psi = thetas - phi                                                # (length,)
 
        cp = np.cos(psi)                                                  # (length,)
        sp = np.sin(psi)                                                  # (length,)
        z  = np.zeros(length)
        o  = np.ones(length)
 
        # rot: (length, 3, 3)
        rot = np.array([[ cp,  sp,  z],
                        [-sp,  cp,  z],
                        [  z,   z,  o]]).transpose(2, 0, 1)
 
        # sshapes: (length, 3, 3)  →  rot @ sshape @ rot.T  for each point
        sshape = rot @ sshapes @ rot.transpose(0, 2, 1)                  # (length, 3, 3)
 
        # --- matrix G: (length, 8, 8) ---
        # collimator values stacked as (length, 8): [a0,a1,b0,b1,a2,a3,b2,b3]
        g_diag = 1. / np.column_stack([alpha_temp[:, :2],
                                        beta_temp[:, :2],
                                        alpha_temp[:, 2:],
                                        beta_temp[:, 2:]]) ** 2          # (length, 8)
        G = np.zeros((length, 8, 8))
        for k in range(8):
            G[:, k, k] = g_diag[:, k]
 
        # --- matrix F: (length, 4, 4)  (same for all points) ---
        f_diag = 1. / np.array([etam, etamv, etaa, etaav],
                                dtype=np.float64) ** 2                   # (4,)
        F      = np.zeros((length, 4, 4))
        for k in range(4):
            F[:, k, k] = f_diag[k]
 
        # trig shorthands
        cotm  = 1. / np.tan(thetam)                                       # (length,)
        cota  = 1. / np.tan(thetaa)
        sinm  = np.sin(thetam)
        sina  = np.sin(thetaa)
        sins  = np.sin(thetas)
        coss  = np.cos(thetas)
        cosm  = np.cos(thetam)
        cosa  = np.cos(thetaa)
        z_    = np.zeros(length)
 
        # --- matrix A: (length, 6, 8) ---
        A = np.zeros((length, 6, 8))
        A[:, 0, 0] =  ki / 2. * cotm
        A[:, 0, 1] = -ki / 2. * cotm
        A[:, 1, 1] =  ki
        A[:, 2, 3] =  ki
        A[:, 3, 4] =  kf / 2. * cota
        A[:, 3, 5] = -kf / 2. * cota
        A[:, 4, 4] =  kf
        A[:, 5, 6] =  kf
 
        # --- matrix C: (length, 4, 8) ---
        C = np.zeros((length, 4, 8))
        C[:, 0, 0] =  0.5
        C[:, 0, 1] =  0.5
        C[:, 1, 2] =  1. / (2. * sinm)
        C[:, 1, 3] = -1. / (2. * sinm)
        C[:, 2, 4] =  0.5
        C[:, 2, 5] =  0.5
        C[:, 3, 6] =  1. / (2. * sina)
        C[:, 3, 7] = -1. / (2. * sina)
 
        # --- matrix Bmatrix: (length, 4, 6) ---
        cosphi   = np.cos(phi)
        sinphi   = np.sin(phi)
        cos_diff = np.cos(phi - s2theta)
        sin_diff = np.sin(phi - s2theta)
 
        Bmatrix = np.zeros((length, 4, 6))
        Bmatrix[:, 0, 0] =  cosphi
        Bmatrix[:, 0, 1] =  sinphi
        Bmatrix[:, 0, 3] = -cos_diff
        Bmatrix[:, 0, 4] = -sin_diff
        Bmatrix[:, 1, 0] = -sinphi
        Bmatrix[:, 1, 1] =  cosphi
        Bmatrix[:, 1, 3] =  sin_diff
        Bmatrix[:, 1, 4] = -cos_diff
        Bmatrix[:, 2, 2] =  1.
        Bmatrix[:, 2, 5] = -1.
        Bmatrix[:, 3, 0] =  2. * CONVERT2 * ki
        Bmatrix[:, 3, 3] = -2. * CONVERT2 * kf
 
        # --- matrix S (Popovici): (length, 13, 13) ---
        """ 
        Sinv = np.zeros((length, 13, 13))
        for ind in range(length):
            Sinv[ind] = blkdiag(np.array(bshape, dtype=np.float64), mshape,
                                sshape[ind], ashape, dshape)
        S = np.linalg.inv(Sinv)                              # (length, 13, 13)
        """
        #new method to calculate S without for loop, by inverting each block separately and then putting them together
        S = np.zeros((length, 13, 13), dtype=np.float64)
        S[:, 0:2, 0:2]     = np.linalg.inv(bshape)
        S[:, 2:5, 2:5]     = np.linalg.inv(mshape)
        S[:, 5:8, 5:8]     = np.linalg.inv(sshape)
        S[:, 8:11, 8:11]   = np.linalg.inv(ashape)
        S[:, 11:13, 11:13] = np.linalg.inv(dshape)
        
        # --- matrix T: (length, 4, 13) ---
        T = np.zeros((length, 4, 13))
        T[:, 0,  0] = -1. / (2. * L0)
        T[:, 0,  2] =  cosm * (1. / L1 - 1. / L0) / 2.
        T[:, 0,  3] =  sinm * (1. / L0 + 1. / L1 - 2. / (monorh * sinm)) / 2.
        T[:, 0,  5] =  sins / (2. * L1)
        T[:, 0,  6] =  coss / (2. * L1)
        T[:, 1,  1] = -1. / (2. * L0 * sinm)
        T[:, 1,  4] =  (1. / L0 + 1. / L1 - 2. * sinm / monorv) / (2. * sinm)
        T[:, 1,  7] = -1. / (2. * L1 * sinm)
        T[:, 2,  5] =  sins / (2. * L2)
        T[:, 2,  6] = -coss / (2. * L2)
        T[:, 2,  8] =  cosa * (1. / L3 - 1. / L2) / 2.
        T[:, 2,  9] =  sina * (1. / L2 + 1. / L3 - 2. / (anarh * sina)) / 2.
        T[:, 2, 11] =  1. / (2. * L3)
        T[:, 3,  7] = -1. / (2. * L2 * sina)
        T[:, 3, 10] =  (1. / L2 + 1. / L3 - 2. * sina / anarv) / (2. * sina)
        T[:, 3, 12] = -1. / (2. * L3 * sina)
 
        # --- matrix D: (length, 8, 13) ---
        D = np.zeros((length, 8, 13))
        D[:, 0,  0] = -1. / L0
        D[:, 0,  2] = -cosm / L0
        D[:, 0,  3] =  sinm / L0
        D[:, 1,  2] =  cosm / L1
        D[:, 1,  3] =  sinm / L1
        D[:, 1,  5] =  sins / L1
        D[:, 1,  6] =  coss / L1
        D[:, 2,  1] = -1. / L0
        D[:, 2,  4] =  1.  / L0
        D[:, 3,  4] = -1. / L1
        D[:, 3,  7] =  1.  / L1
        D[:, 4,  5] =  sins / L2
        D[:, 4,  6] = -coss / L2
        D[:, 4,  8] = -cosa / L2
        D[:, 4,  9] =  sina / L2
        D[:, 5,  8] =  cosa / L3
        D[:, 5,  9] =  sina / L3
        D[:, 5, 11] =  1.  / L3
        D[:, 6,  7] = -1. / L2
        D[:, 6, 10] =  1.  / L2
        D[:, 7, 10] = -1. / L3
        D[:, 7, 12] =  1.  / L3
 
        # --- resolution matrix M for all points ---
        if method == 1 or method == 'popovici':
            K     = S + T.transpose(0, 2, 1) @ F @ T                     # (length, 13, 13)
            H_mat = np.linalg.inv(D @ np.linalg.inv(K) @ D.transpose(0, 2, 1))
            Ninv  = A @ np.linalg.inv(H_mat + G) @ A.transpose(0, 2, 1)
        else:
            H_mat = G + C.transpose(0, 2, 1) @ F @ C                     # (length, 8, 8)
            Ninv  = A @ np.linalg.inv(H_mat) @ A.transpose(0, 2, 1)
            # Horizontally focusing analyzer if needed
            if horifoc > 0:
                Ninv = np.linalg.inv(Ninv)
                Ninv[:, 3, 3] = (np.tan(thetaa) / (etaa * kf)) ** 2
                Ninv[:, 4, 4] = (1. / (kf * alpha_temp[:, 2])) ** 2
                Ninv[:, 3, 4] = 0.
                Ninv[:, 4, 3] = 0.
                Ninv = np.linalg.inv(Ninv)
 
        Minv = Bmatrix @ Ninv @ Bmatrix.transpose(0, 2, 1)               # (length, 4, 4)
        M    = np.linalg.inv(Minv)                                        # (length, 4, 4)
        RM_  = M.copy()                                                   # (length, 4, 4)
 
        # Correct matrix element assignments (swap energy and vertical momentum)
        RM_[:, 0, 2] = M[:, 0, 3]
        RM_[:, 2, 0] = M[:, 3, 0]
        RM_[:, 2, 2] = M[:, 3, 3]
        RM_[:, 2, 1] = M[:, 3, 1]
        RM_[:, 1, 2] = M[:, 1, 3]
 
        RM_[:, 0, 3] = M[:, 0, 2]
        RM_[:, 3, 0] = M[:, 2, 0]
        RM_[:, 3, 3] = M[:, 2, 2]
        RM_[:, 3, 1] = M[:, 2, 1]
        RM_[:, 1, 3] = M[:, 1, 2]
 
        # --- prefactor R0, normalised to source ---
        Rm  = ki ** 3 / np.tan(thetam)                                    # (length,)
        Ra  = kf ** 3 / np.tan(thetaa)
        R0_ = Rm * Ra * (2. * np.pi) ** 4 / (64. * np.pi ** 2 * sinm * sina)
 
        if method == 1 or method == 'popovici':
            # Popovici
            R0_ = R0_ * np.sqrt(np.linalg.det(F) /
                                 np.linalg.det(H_mat + G))               # (length,)
        else:
            # Cooper-Nathans
            R0_ = R0_ * np.sqrt(np.linalg.det(F) /
                                 np.linalg.det(H_mat))
 
        # --- normalisation to flux on monitor ---
        if moncor == 1:
            g = G[:, :4, :4]                                              # (length, 4, 4)
            f = F[:, :2, :2]                                              # (length, 2, 2)
            c = C[:, :2, :4]                                              # (length, 2, 4)
 
            # monitor t matrix: (length, 2, 7)
            t = np.zeros((length, 2, 7))
            t[:, 0, 0] = -1. / (2. * L0)
            t[:, 0, 2] =  cosm * (1. / L1mon - 1. / L0) / 2.
            t[:, 0, 3] =  sinm * (1. / L0 + 1. / L1mon
                                   - 2. / (monorh * sinm)) / 2.
            t[:, 0, 6] =  1. / (2. * L1mon)
            t[:, 1, 1] = -1. / (2. * L0 * sinm)
            t[:, 1, 4] =  (1. / L0 + 1. / L1mon
                            - 2. * sinm / monorv) / (2. * sinm)
 
            sinv     = blkdiag(np.array(bshape, dtype=np.float64),
                               mshape, monitorshape)                      # (7, 7) — same for all points
            s_mon    = np.linalg.inv(sinv)                                # (7, 7)
            s_mon_b  = np.broadcast_to(s_mon, (length, 7, 7)).copy()     # (length, 7, 7)
 
            # monitor d matrix: (length, 4, 7)
            d = np.zeros((length, 4, 7))
            d[:, 0, 0] = -1. / L0
            d[:, 0, 2] = -cosm / L0
            d[:, 0, 3] =  sinm / L0
            d[:, 1, 2] =  cosm / L1mon
            d[:, 1, 3] =  sinm / L1mon
            d[:, 1, 6] =  1.  / L1mon
            d[:, 2, 1] = -1. / L0
            d[:, 2, 4] =  1.  / L0
            d[:, 3, 4] = -1. / L1mon
 
            if method == 1 or method == 'popovici':
                # Popovici
                K_mon = s_mon_b + t.transpose(0, 2, 1) @ f @ t           # (length, 7, 7)
                inner = np.linalg.inv(d @ np.linalg.inv(K_mon)
                                      @ d.transpose(0, 2, 1)) + g        # (length, 4, 4)
                Rmon  = (Rm * (2 * np.pi) ** 2 / (8 * np.pi * sinm) *
                         np.sqrt(np.linalg.det(f) / np.linalg.det(inner)))
            else:
                # Cooper-Nathans
                inner = g + c.transpose(0, 2, 1) @ f @ c                 # (length, 4, 4)
                Rmon  = (Rm * (2 * np.pi) ** 2 / (8 * np.pi * sinm) *
                         np.sqrt(np.linalg.det(f) / np.linalg.det(inner)))
 
            R0_ = R0_ / Rmon
            R0_ = R0_ * ki                                                # 1/ki monitor efficiency
 
        # Transform prefactor to Chesser-Axe normalization
        R0_ = R0_ / (2. * np.pi) ** 2 * np.sqrt(np.linalg.det(RM_))
        # Include kf/ki part of cross section
        R0_ = R0_ * kf / ki
 
        # --- sample mosaic correction ---
        if hasattr(sample, 'mosaic'):
            etas  = sample.mosaic * CONVERT1
            etasv = np.copy(etas)
            if hasattr(sample, 'vmosaic'):
                etasv = sample.vmosaic * CONVERT1
            R0_ = R0_ / np.sqrt((1 + (Q * etas)  ** 2 * RM_[:, 3, 3]) *
                                  (1 + (Q * etasv) ** 2 * RM_[:, 1, 1]))
            Minv_temp          = np.linalg.inv(RM_)                             # (length, 4, 4)
            Minv_temp[:, 1, 1] = Minv_temp[:, 1, 1] + Q ** 2 * etas  ** 2
            Minv_temp[:, 3, 3] = Minv_temp[:, 3, 3] + Q ** 2 * etasv ** 2
            RM_ = np.linalg.inv(Minv_temp)
 
        # --- analyzer reflectivity correction ---
        if hasattr(ana, 'depth') and hasattr(ana, 'KQ'):
            KQ  = ana.KQ                                                        #   KQ: extinction coefficient of the crystal
            KT  = ana.depth
            toa = (taua / 2.) / np.sqrt(kf ** 2 - (taua / 2.) ** 2)             # (length,)
            
            smallest  = np.minimum(alpha_temp[:, 2], alpha_temp[:, 3])          # smallest collimator per point: min(alpha_temp[:,2], alpha_temp[:,3])
            Qdsint    = KQ * toa                                                # (length,)
            
            dth_grid  = (np.arange(1, 201) / 200.) * np.sqrt(2. * np.log(2.))   # dth grid: (200,) broadcast against (length,) smallest
            dth       = dth_grid[np.newaxis, :] * smallest[:, np.newaxis]       # (length, 200)
            wdth      = np.exp(-dth ** 2 / 2. / etaa ** 2)                      # (length, 200)
            sdth      = (KT * Qdsint[:, np.newaxis] * wdth
                         / etaa / np.sqrt(2. * np.pi))                          # (length, 200)
            rdth      = 1. / (1 + 1. / sdth)                                    # (length, 200)
            reflec    = np.sum(rdth, axis=1) / np.sum(wdth, axis=1)             # (length,)
            R0_       = R0_ * reflec
 
        R0 = R0_                                                                # (length,)
        RM = RM_                                                                # (length, 4, 4)
 
        return [R0, RM]

    def CalcResMatHKL(self, hkle):
        """
        For a scattering vector (H,K,L) and energy transfers W, given
        experimental conditions specified on the instrument, calculates the resolution
        matrix RMS and prefactor R0 (Cooper-Nathans OR Popovici -- CalcResMatQ follows
        self.method, default method=1 = Popovici) in the sample's crystallographic frame.
        
        Args:
            hkle (list): Array of the scattering vector and energy transfer at which the
                        calculation should be performed
                        
        Returns None; stores self.R0, self.RMS, self.RM. Based on ResLib 3.4c (A. Zheludev).
        """

        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
            
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")

        self.HKLE = [H, K, L, W]

        x, y, z, sample, rsample = self._StandardSystem()
        del z, sample

        Q = _modvec([H, K, L], rsample)
        uq = np.vstack((H / Q, K / Q, L / Q))

        xq = _scalar(x, uq, rsample)
        yq = _scalar(y, uq, rsample)

        tmat = np.array([np.array([[xq[i], yq[i], 0, 0], 
                                  [-yq[i], xq[i], 0, 0], 
                                  [0, 0, 1., 0], 
                                  [0, 0, 0, 1.]], dtype=np.float64) 
                        for i in range(len(xq))])

        RMS = np.zeros((length, 4, 4), dtype=np.float64)

        R0, RM = self.CalcResMatQ(Q, W)

        #####################################################################################
        RMS = tmat.transpose(0, 2, 1) @ RM @ tmat                        # (length, 4, 4)

        # Apply smoothing corrections if specified
        if hasattr(self, 'Smooth') and hasattr(self.Smooth, 'X') and self.Smooth.X:
            mul = np.diag([1 / (self.Smooth.X ** 2 / 8 / np.log(2)),
                           1 / (self.Smooth.Y ** 2 / 8 / np.log(2)),
                           1 / (self.Smooth.E ** 2 / 8 / np.log(2)),
                           1 / (self.Smooth.Z ** 2 / 8 / np.log(2))])   # (4, 4)

            inv_mul = np.linalg.inv(mul)                                  # (4, 4)  — same for all points
            inv_RMS = np.linalg.inv(RMS)                                  # (length, 4, 4)

            # inv_mul broadcast against (length, 4, 4) for the addition
            denom   = inv_mul[np.newaxis] + inv_RMS                       # (length, 4, 4)

            R0  = (R0
                   / np.sqrt(np.linalg.det(inv_RMS))                     # (length,)
                   * np.sqrt(np.linalg.det(denom)))                       # (length,)

            RMS = np.linalg.inv(denom)                                    # (length, 4, 4)
        #####################################################################################

        #old: risk of shape mismatch if length=1 and R0, RMS, RM are returned as scalars or 2D arrays with shape (4,4) instead of (1,4,4).
        #self.R0, self.RMS, self.RM = [np.squeeze(item) for item in (R0, RMS, RM)]

        # After (safe):
        self.R0  = R0.reshape(-1)           # always (length,)
        self.RMS = RMS.reshape(-1, 4, 4)    # always (length, 4, 4)
        self.RM  = RM.reshape(-1, 4, 4)     # always (length, 4, 4)

        return

    def ResConv(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        """
        Numerically calculate the convolution of a user-defined cross-section function with the 
        resolution function for a 3-axis neutron scattering experiment.
        
        Args:
            sqw (func): User-supplied "fast" model cross section.
            pref (func): User-supplied "slow" cross section prefactor and background function.
            nargout (int): Number of arguments returned by the pref function
            hkle (tuple): Tuple of H, K, L, and W, specifying the wave vector and energy
                         transfers at which the convolution is to be calculated (i.e.
                         define Q₀). H, K, and L are given in reciprocal lattice units and W in meV.
            METHOD (str): Specifies which 4D-integration method to use. 
                         'fix' (Default): sample the cross section on a fixed grid of points uniformly
                         distributed in φ-space. 2*ACCURACY[0]+1 points are sampled
                         along φ₁, φ₂, and φ₃, and 2*ACCURACY[1]+1 along φ₄ (vertical direction). 
                         'mc': 4D Monte Carlo integration. The cross section is sampled in 
                         1000*ACCURACY randomly chosen points, uniformly distributed in φ-space.
            ACCURACY (array or int): Determines the number of sampling points in the integration.
            p (list): A parameter that is passed on, without change to sqw and pref.
            seed (int, optional): Random seed for Monte Carlo method.
            
        Returns:
            array: conv - Calculated value of the cross section, folded with the resolution function 
                   at the given Q₀
                   
        Based on ResLib 3.4c, originally authored by A. Zheludev
        """
        GAUSS_NORM_1D = np.sqrt(2.0 / np.pi)   # ≈ 0.79788, normalization for M=0 (collapsed) dimension
        self.CalcResMatHKL(hkle)
        R0, RMS = np.copy(self.R0), self.RMS.copy()

        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
            
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")


        xvec, yvec, zvec = self._StandardSystem()[:3]

        # Extract resolution matrix elements with corrected indexing
        Mxx = RMS[:, 0, 0]
        Mxy = RMS[:, 0, 1]
        Mxw = RMS[:, 0, 2]  # Energy-related terms
        Myy = RMS[:, 1, 1]
        Myw = RMS[:, 1, 2]  # Energy-related terms
        Mzz = RMS[:, 3, 3]  # Vertical momentum
        Mww = RMS[:, 2, 2]  # Energy

        # Matrix reduction for integration
        Mxx -= Mxw ** 2. / Mww
        Mxy -= Mxw * Myw / Mww
        Myy -= Myw ** 2. / Mww
        MMxx = Mxx - Mxy ** 2. / Myy

        detM = MMxx * Myy * Mzz * Mww

        # Transformation parameters
        tqz = 1. / np.sqrt(Mzz)
        tqx = 1. / np.sqrt(MMxx)
        tqyy = 1. / np.sqrt(Myy)
        tqyx = -Mxy / Myy / np.sqrt(MMxx)
        tqww = 1. / np.sqrt(Mww)
        tqwy = -Myw / Mww / np.sqrt(Myy)
        tqwx = -(Mxw / Mww - Myw / Mww * Mxy / Myy) / np.sqrt(MMxx)

        inte = sqw(H, K, L, W, p)
        modes, points = inte.shape

        # Handle prefactor function
        if pref is None:
            prefactor = np.ones((modes, points))
            bgr = 0
        else:
            if nargout == 2:
                prefactor, bgr = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Invalid number of output arguments in prefactor function')

        # Choose integration method
        if METHOD == 'fix':
            if ACCURACY is None:
                ACCURACY = np.array([7, 0])
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            
            cw, cx, cy = np.meshgrid(dd1, dd1, dd1, indexing='ij')
            # Use flatten() to ensure proper array handling
            tx = np.tan(cx.flatten())
            ty = np.tan(cy.flatten())
            tw = np.tan(cw.flatten())
            tz = np.tan(dd2)
            
            norm = (np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * 
                   np.exp(-0.5 * (tw ** 2)) * (1 + tw ** 2))
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)

            #Fix: Use broadcasting to compute dQ1, dQ2, and dW for all points at once
            dQ1 = tqx[:, np.newaxis] * tx[np.newaxis, :]          # (length, P)
            dQ2 = tqyy[:, np.newaxis] * ty[np.newaxis, :] + tqyx[:, np.newaxis] * tx[np.newaxis, :]
            dW  = tqwx[:, np.newaxis] * tx[np.newaxis, :] + tqwy[:, np.newaxis] * ty[np.newaxis, :] + tqww[:, np.newaxis] * tw[np.newaxis, :]

            for iz in range(len(tz)):

                dQ4 = tqz[:, np.newaxis] * tz[iz]
                H1 = H[:, np.newaxis] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]    # (length, P)
                K1 = K[:, np.newaxis] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                L1 = L[:, np.newaxis] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                W1 = W[:, np.newaxis] + dW   

                #Fix: Ensure proper broadcasting by adding new axes to norm and normz
                inte = sqw(H1.ravel(), K1.ravel(), L1.ravel(), W1.ravel(), p)                  # (modes, length*P)
                inte = inte.reshape(modes, length, -1)                                         # (modes, length, P)
                convs += np.sum(inte * (norm * normz[iz])[np.newaxis, np.newaxis, :], axis=2)  # (modes, length)
            
            conv = np.sum(convs * prefactor, axis=0)
            conv *= step1**3 * step2 / np.sqrt(detM)

            if M[1] == 0:
                conv *= GAUSS_NORM_1D
            if M[0] == 0:
                conv *= GAUSS_NORM_1D ** 3

        elif METHOD == 'mc':
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError(f'ACCURACY must be an int when using Monte Carlo method: {ACCURACY}')
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            N = M * 1000                                                          # total samples per point

            if seed is not None:
                np.random.seed(seed)

            # --- draw ALL samples at once: (4, length, N) ---
            r            = np.random.uniform(-np.pi / 2, np.pi / 2, size=(4, length, N))
            tx, ty, tz, tw = np.tan(r[0]), np.tan(r[1]), np.tan(r[2]), np.tan(r[3])
                                                                                # each (length, N)

            # --- Jacobian-weighted kernel: (length, N) ---
            norm = (np.exp(-0.5 * (tx**2 + ty**2 + tz**2 + tw**2)) *
                    (1 + tx**2) * (1 + ty**2) * (1 + tz**2) * (1 + tw**2))

            # --- φ-space displacements: (length, N) ---
            # tqx, tqyy, … are (length,) → broadcast against N via [:, np.newaxis]
            dQ1 = tqx [:, np.newaxis] * tx
            dQ2 = tqyy[:, np.newaxis] * ty + tqyx[:, np.newaxis] * tx
            dW  = tqwx[:, np.newaxis] * tx + tqwy[:, np.newaxis] * ty + tqww[:, np.newaxis] * tw
            dQ4 = tqz [:, np.newaxis] * tz

            # --- (H,K,L,W) sample points: (length, N) ---
            H1 = H[:, np.newaxis] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
            K1 = K[:, np.newaxis] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
            L1 = L[:, np.newaxis] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
            W1 = W[:, np.newaxis] + dW

            # --- evaluate cross section over flattened (length*N,) arrays ---
            # sqw must accept flat arrays and return (modes, length*N)
            inte = sqw(H1.ravel(), K1.ravel(), L1.ravel(), W1.ravel(), p)        # (modes, length*N)
            inte = inte.reshape(modes, length, N)                                 # (modes, length, N)

            # --- weighted sum over N samples → (modes, length) ---
            # norm is (length, N); inte[j] is (length, N)
            convs = np.sum(inte * norm[np.newaxis], axis=2)                       # (modes, length)

            # --- prefactor is (modes, length); sum over modes → (length,) ---
            conv  = np.sum(convs * prefactor, axis=0)                             # (length,)

            # --- normalisation: π⁴ from 4-D domain ---
            conv = conv / N * np.pi ** 4/  np.sqrt(detM)

        else:
            raise ValueError(f'Unknown METHOD: {METHOD}. Valid options are: "fix", "mc"')

        conv *= R0
        conv += bgr

        return conv


    def ResConvSMA(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        """
        Numerically calculate the convolution of a user-defined single-mode cross-section function 
        with the resolution function for a 3-axis neutron scattering experiment.
        
        Args:
            sqw (func): Cross section function
            pref (func): Prefactor and background function
            nargout (int): Number of output arguments from pref function
            hkle (tuple): H, K, L, W values
            METHOD (str): Integration method ('fix' or 'mc')
            ACCURACY (array or int): Integration accuracy parameters
            p (list): Parameters passed to sqw and pref functions
            seed (int, optional): Random seed for Monte Carlo method
            
        Returns:
            array: conv - Calculated value of the cross section, folded with the resolution function
            
        Based on ResLib 3.4c, originally authored by A. Zheludev
        """
        GAUSS_NORM_1D = np.sqrt(2.0 / np.pi)   # ≈ 0.79788, normalization for M=0 (collapsed) dimension
        self.CalcResMatHKL(hkle)
        R0, RMS = np.copy(self.R0), self.RMS.copy()

        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
            
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")

        xvec, yvec, zvec = self._StandardSystem()[:3]

        # Extract matrix elements for single-mode approximation
        #20261010 Bug #1: RMS element order is [Qx, Qy, E, Qz] (0-based), matching
        # ResLib ConvResSMA.m:24 (Mww=RMS(3,3), Mxw=RMS(1,3), Myw=RMS(2,3)) after the
        # MATLAB 1-based -> Python 0-based translation. E/Qz were swapped here.
        Mww = RMS[:, 2, 2]  # Energy diagonal element
        Mxw = RMS[:, 0, 2]  # Energy-x coupling
        Myw = RMS[:, 1, 2]  # Energy-y coupling

        # Calculate Lorentzian parameters
        GammaFactor = np.sqrt(Mww / 2)
        OmegaFactorx = Mxw / np.sqrt(2 * Mww)
        OmegaFactory = Myw / np.sqrt(2 * Mww)

        # Reduced matrix elements (removing energy dependence)
        #20261010 Bug #1: Mzz is the vertical-momentum (Qz) element;
        # ResLib ConvResSMA.m:28 uses Mzz=RMS(4,4) -> RMS[:,3,3] (was RMS[:,2,2]).
        Mzz = RMS[:, 3, 3]
        Mxx = RMS[:, 0, 0] - Mxw ** 2 / Mww
        Myy = RMS[:, 1, 1] - Myw ** 2 / Mww
        Mxy = RMS[:, 0, 1] - Mxw * Myw / Mww

        # Transformation parameters
        detxy = np.sqrt(Mxx * Myy - Mxy ** 2)
        detz = np.sqrt(Mzz)

        tqz = 1. / detz
        tqy = np.sqrt(Mxx) / detxy
        tqxx = 1. / np.sqrt(Mxx)
        tqxy = Mxy / np.sqrt(Mxx) / detxy

        # Get dispersion and intensity from cross section function
        disp, inte = sqw(H, K, L, p)[:2]
        modes, points = disp.shape

        # Handle prefactor function
        if pref is None:
            prefactor = np.ones((modes, points))
            bgr = 0
        else:
            if nargout == 2:
                prefactor, bgr = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Invalid number of output arguments in prefactor function')

        # Choose integration method
        # After (fixed):
        if METHOD == 'mc':
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError(f'ACCURACY must be an int when using Monte Carlo method: {ACCURACY}')
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY   # ← now defined

            if seed is not None:
                np.random.seed(seed)
            N = M * 1000

            # --- draw ALL samples at once: (3, length, N) ---
            r        = np.random.uniform(-np.pi / 2, np.pi / 2, size=(3, length, N))
            tx, ty, tz = np.tan(r[0]), np.tan(r[1]), np.tan(r[2])
                                                                                # each (length, N)

            # --- Jacobian-weighted kernel: (length, N) ---
            norm = (np.exp(-0.5 * (tx**2 + ty**2 + tz**2)) *
                    (1 + tx**2) * (1 + ty**2) * (1 + tz**2))

            # --- φ-space displacements: (length, N) ---
            dQ1 = tqxx[:, np.newaxis] * tx - tqxy[:, np.newaxis] * ty
            dQ2 = tqy [:, np.newaxis] * ty
            dQ4 = tqz [:, np.newaxis] * tz

            # --- (H,K,L) sample points: (length, N) ---
            H1 = H[:, np.newaxis] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
            K1 = K[:, np.newaxis] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
            L1 = L[:, np.newaxis] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]

            # --- evaluate cross section over flattened (length*N,) arrays ---
            # sqw must return (modes, length*N)
            disp, inte, WL = sqw(H1.ravel(), K1.ravel(), L1.ravel(), p)
            modes, _       = disp.shape if disp.ndim == 2 else (1, None)
            disp = disp.reshape(modes, length, N)                                 # (modes, length, N)
            inte = inte.reshape(modes, length, N)
            WL   = WL.reshape(modes, length, N)

            # --- Voigt argument arrays: (modes, length, N) ---
            # GammaFactor, OmegaFactorx, OmegaFactory, detxy, detz are (length,)
            Gamma = WL * GammaFactor[np.newaxis, :, np.newaxis]                   # (modes, length, N)
            Omega = (GammaFactor[np.newaxis, :, np.newaxis]  * (disp - W[np.newaxis, :, np.newaxis]) +
                    OmegaFactorx[np.newaxis, :, np.newaxis] * dQ1[np.newaxis] +
                    OmegaFactory[np.newaxis, :, np.newaxis] * dQ2[np.newaxis])   # (modes, length, N)

            # --- per-mode weighted sum: (modes, length) ---
            weight = (norm[np.newaxis]                                            # (1, length, N)
                    / detxy[np.newaxis, :, np.newaxis]
                    / detz [np.newaxis, :, np.newaxis])

            convs = np.sum(inte * _voigt(Omega, Gamma) * weight, axis=2)          # (modes, length)

            # --- sum over modes → (length,) ---
            conv  = np.sum(convs * prefactor, axis=0)                             # (length,)

            # --- normalisation: π³ from 3-D domain ---
            conv = conv / N * np.pi ** 3


        elif METHOD == 'fix':
            if ACCURACY is None:
                ACCURACY = [7, 0]
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))

            cy, cx = np.meshgrid(dd1, dd1, indexing='ij')
            tx = np.tan(cx.flatten())                                     # (P,)  P=(2M[0]+1)²
            ty = np.tan(cy.flatten())                                     # (P,)
            tz = np.tan(dd2)                                              # (Z,)  Z=(2M[1]+1)

            norm  = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2)  # (P,)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)            # (Z,)

            # --- combined vertical weight: (Z, 1, 1, P) for broadcasting ---
            # axes: (Z, modes, length, P)
            # tqxx, tqxy, tqy, tqz are (length,)

            # dQ1, dQ2: (length, P)
            dQ1 = tqxx[:, np.newaxis] * tx - tqxy[:, np.newaxis] * ty   # (length, P)
            dQ2 = tqy [:, np.newaxis] * ty                               # (length, P)

            # H1, K1, L1: (length, P)  — no tz dependence in-plane
            H1_base = H[:, np.newaxis] + dQ1 * xvec[0] + dQ2 * yvec[0] # (length, P)
            K1_base = K[:, np.newaxis] + dQ1 * xvec[1] + dQ2 * yvec[1]
            L1_base = L[:, np.newaxis] + dQ1 * xvec[2] + dQ2 * yvec[2]

            # --- call sqw once over all (length × P) in-plane points ---
            # sqw returns (modes, length*P); reshape to (modes, length, P)
            disp_base, inte_base, WL_base = sqw(H1_base.ravel(),
                                                K1_base.ravel(),
                                                L1_base.ravel(), p)
            modes, _ = disp_base.shape
            disp_base = disp_base.reshape(modes, length, -1)             # (modes, length, P)
            inte_base = inte_base.reshape(modes, length, -1)             # (modes, length, P)
            WL_base   = WL_base.reshape(modes, length, -1)               # (modes, length, P)

            # --- Voigt arguments: (modes, length, P) ---
            # GammaFactor, OmegaFactorx, OmegaFactory are (length,)
            Gamma = WL_base * GammaFactor[np.newaxis, :, np.newaxis]     # (modes, length, P)
            Omega = (GammaFactor  [np.newaxis, :, np.newaxis] *
                         (disp_base - W[np.newaxis, :, np.newaxis]) +
                     OmegaFactorx [np.newaxis, :, np.newaxis] * dQ1[np.newaxis] +
                     OmegaFactory [np.newaxis, :, np.newaxis] * dQ2[np.newaxis])
                                                                          # (modes, length, P)

            # --- per-point normalisation weight: (modes, length, P) ---
            #weight = (norm[np.newaxis, np.newaxis, :]                    # (1, 1, P)
            #          / detxy[np.newaxis, :, np.newaxis]                 # (1, length, 1)
            #          / detz [np.newaxis, :, np.newaxis])                # (1, length, 1)
            
            weight = (norm[np.newaxis, np.newaxis, np.newaxis, :]       # (1, 1, 1, P)
                    / detxy[np.newaxis, np.newaxis, :, np.newaxis]     # (1, 1, length, 1)
                    / detz [np.newaxis, np.newaxis, :, np.newaxis])    # (1, 1, length, 1)
            
            voigt_norm = (inte_base * _voigt(Omega, Gamma) * weight)     # (modes, length, P)

            # --- sum over in-plane grid P: (modes, length) ---
            #convs_xy = np.sum(voigt_norm, axis=2)                        # (modes, length)

            # --- accumulate over vertical grid Z, weighted by normz ---
            # dQ4 contribution shifts only H1/K1/L1 by zvec — but sqw for SMA
            # does not take W as input, so the vertical displacement only
            # affects the spatial arguments through dQ4 * zvec.
            # Since dQ4 = tqz[i] * tz[iz] varies with both i and iz,
            # we handle the z-loop separately but keep i vectorised.
            convs = np.zeros((modes, length))
            Z = 2*M[1]+1
            # dQ4: (Z, length)
            dQ4 = tqz[np.newaxis, :] * tz[:, np.newaxis]           # (Z, length)

            # H1z: (Z, length, P)
            H1z = H1_base[np.newaxis] + dQ4[:, :, np.newaxis] * zvec[0]
            K1z = K1_base[np.newaxis] + dQ4[:, :, np.newaxis] * zvec[1]
            L1z = L1_base[np.newaxis] + dQ4[:, :, np.newaxis] * zvec[2]

            # flatten to (Z*length*P,) for sqw call
            # the following vectorized code works. Try to understand it before changing.
            disp_z, inte_z, WL_z = sqw(H1z.ravel(), K1z.ravel(), L1z.ravel(), p)
            disp_z = disp_z.reshape(modes, Z, length, -1)           # (modes, Z, length, P)
            inte_z = inte_z.reshape(modes, Z, length, -1)
            WL_z   = WL_z.reshape(modes, Z, length, -1)
            """
            #need z dimension
            Gamma_z = WL_z * GammaFactor[np.newaxis, :, np.newaxis]
            Omega_z = (GammaFactor  [np.newaxis, :, np.newaxis] *
                               (disp_z - W[np.newaxis, :, np.newaxis]) +
                           OmegaFactorx [np.newaxis, :, np.newaxis] * dQ1[np.newaxis] +
                           OmegaFactory [np.newaxis, :, np.newaxis] * dQ2[np.newaxis])
            """
            Gamma_z = WL_z * GammaFactor[np.newaxis, np.newaxis, :, np.newaxis]      # (modes, Z, length, P)
            Omega_z = (GammaFactor  [np.newaxis, np.newaxis, :, np.newaxis] *
               (disp_z - W[np.newaxis, np.newaxis, :, np.newaxis]) +
                OmegaFactorx [np.newaxis, np.newaxis, :, np.newaxis] * dQ1[np.newaxis, np.newaxis] +
                OmegaFactory [np.newaxis, np.newaxis, :, np.newaxis] * dQ2[np.newaxis, np.newaxis])

            #voigt_z = (inte_z * _voigt(Omega_z, Gamma_z) * weight)   # (modes, length, P)
            #20261010 Bug #1.2: voigt_z must NOT include inte_z — the next line already
            # multiplies by inte_z, so the old form squared the cross-section (inte_z**2).
            # ResLib ConvResSMA.m:125 has a single int(j,:) factor:
            #   add = int(j,:) .* voigt(Omega,Gamma) .* norm * normz(iz) / detxy / detz
            # Here weight = norm/detxy/detz (line 1590-1592), so line below now mirrors
            # that MATLAB expression exactly.
            voigt_z = _voigt(Omega_z, Gamma_z) 
            convs = np.sum(inte_z * voigt_z * weight * normz[np.newaxis, :, np.newaxis, np.newaxis], axis=(1, 3))
                                                                    # (modes, length)

            """
            #STALE (kept for reference only): this z-loop predates the vectorized code above and would now BROADCAST-FAIL against the 4-D weight (1,1,1,P) at :1590-1592 -- do not uncomment (proj_review_comments_v4.md §2.4 #23).
            for iz in range(tz.size):
                # dQ4: (length,) — scalar tz[iz] broadcast over length
                dQ4 = tqz * tz[iz]                                        # (length,)

                # H1z, K1z, L1z: (length, P)
                H1z = H1_base + dQ4[:, np.newaxis] * zvec[0]
                K1z = K1_base + dQ4[:, np.newaxis] * zvec[1]
                L1z = L1_base + dQ4[:, np.newaxis] * zvec[2]

                disp_z, inte_z, WL_z = sqw(H1z.ravel(), K1z.ravel(),
                                           L1z.ravel(), p)
                disp_z = disp_z.reshape(modes, length, -1)               # (modes, length, P)
                inte_z = inte_z.reshape(modes, length, -1)
                WL_z   = WL_z.reshape(modes, length, -1)

                Gamma_z = WL_z * GammaFactor[np.newaxis, :, np.newaxis]
                Omega_z = (GammaFactor  [np.newaxis, :, np.newaxis] *
                               (disp_z - W[np.newaxis, :, np.newaxis]) +
                           OmegaFactorx [np.newaxis, :, np.newaxis] * dQ1[np.newaxis] +
                           OmegaFactory [np.newaxis, :, np.newaxis] * dQ2[np.newaxis])

                voigt_z = (inte_z * _voigt(Omega_z, Gamma_z) * weight)   # (modes, length, P)

                # sum over P, weight by normz[iz]: accumulate into convs
                convs += normz[iz] * np.sum(voigt_z, axis=2)             # (modes, length)
            """
            # --- sum over modes → (length,) ---
            conv = np.sum(convs * prefactor, axis=0)                     # (length,)

            # --- normalisation ---
            conv = conv * step1 ** 2 * step2


            if M[1] == 0:
                conv *= GAUSS_NORM_1D
            if M[0] == 0:
                conv *= GAUSS_NORM_1D ** 2
        else:
            raise ValueError(f'Unknown METHOD: {METHOD}. Valid options are: "fix" or "mc".')

        conv = conv * R0
        conv = conv + bgr

        return conv



    def ResolutionPlot(self, hkle=None, SMA=None, SMAp=None):
        #plot the calculated resolution at hkle
        if hkle is None:
            print("no enough parameters for plotting!")
            return

        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
            
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")
   
        #20261010: L1 Fix: match ResPlot.m round(length(H)/2) -- (length-1)//2 is the 0-based equivalent (floor length//2 was off by one for even-length scans)
        center = (length - 1) // 2

        StyleOne = '-'
        StyleTwo = '--'

        XYAxesPos  = [0.1, 0.6, 0.36, 0.36]
        XEAxesPos  = [0.1, 0.1, 0.36, 0.36]
        YEAxesPos  = [0.6, 0.6, 0.36, 0.36]
        TxtAxesPos = [0.55, 0.0, 0.36, 0.36]
        GridPoints = 101

        # [R0,RMS]  =  ResMatS(H, K, L, W, EXP)
        self.CalcResMatHKL(hkle)
        [R0, RMS]  = [np.copy(self.R0), self.RMS.copy()]

        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        #Q = _modvec([H, K, L], rsample)
        #uq = np.vstack((H / Q, K / Q, L / Q)) 
        uq = np.vstack((H , K , L ))
        
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W
        #print("qx:{}:".format(qx))
        #print("qy:{}:".format(qy))

        #20261010 Bug #1.4: orient1/orient2 must be COPIED — they alias the live sample.
        #   WHAT WAS WRONG
        #     * the properties return the stored arrays themselves, with no copy:
        #         :311  return self._sample.u       :320  return self._sample.v
        #       so `o1 = self.orient1` / `o2 = self.orient2` are ALIASES of sample.u/v.
        #       MATLAB's `o1=EXP.orient1` (ResPlot.m:44-45) copies, because MATLAB
        #       assignment has value semantics — hence ResPlot.m is harmless and this
        #       port was not.
        #     * the lines below then write INTO them (o2[0..2] = yvec*pr, plus the six
        #       `abs(...) < 1e-5 -> 0` tests), permanently overwriting sample.v AND
        #       sample.u. ResolutionPlotProj runs from the GUI on every parameter change,
        #       so the sample orientation drifted and every later CalcResMatHKL / ResConv
        #       result changed too — silently degrading agreement with ResLib.
        #   WHY THIS IS THE CORRECT MINIMAL FIX
        #     * restores MATLAB's value semantics: local o1/o2 now behave exactly like
        #       ResPlot.m's copies, and nothing downstream changes.
        #   NOT CHANGED
        #     * `pr`, the projection, the 1e-5 threshold, the zeroing logic. Also note
        #       o1/o2 are local-only here while ResPlot.m additionally uses them at
        #       :77-79/:103-105/:129-134 for axis scale/labels — a separate finding.
        o1    = np.array(self.orient1, copy=True)
        o2    = np.array(self.orient2, copy=True)
        pr    = _scalar(o2,yvec,rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if abs(o2[0])<1e-5: o2[0] = 0
        if abs(o2[1])<1e-5: o2[1] = 0
        if abs(o2[2])<1e-5: o2[2] = 0

        if abs(o1[0])<1e-5: o1[0] = 0
        if abs(o1[1])<1e-5: o1[1] = 0
        if abs(o1[2])<1e-5: o1[2] = 0

        #========================================================================================================
        #determine the plot range
        XWidth  =  np.max(fproject(RMS,0))
        YWidth  =  np.max(fproject(RMS,1))
        WWidth  =  np.max(fproject(RMS,2))
        XMax    =  np.max(qx)+XWidth*1.5
        XMin    =  np.min(qx)-XWidth*1.5
        YMax    =  np.max(qy)+YWidth*1.5
        YMin    =  np.min(qy)-YWidth*1.5
        WMax    =  np.max(qw)+WWidth*1.5
        WMin    =  np.min(qw)-WWidth*1.5

        #========================================================================================================
        # plot XE projection
        fig = plt.figure()
        ((ax_xe, ax_ye), (ax_xy, ax_txt)) = fig.subplots(2, 2)
        ax_xe.set_position(XEAxesPos)
        ax_ye.set_position(YEAxesPos)
        ax_xy.set_position(XYAxesPos)
        ax_txt.set_position(TxtAxesPos)
        #xe_plot

        ax_xe.set_xlim(XMin, XMax)
        ax_xe.set_ylim(WMin, WMax)
    
        ax_xe.set_xlabel('Qx[A-1]')
        ax_xe.set_ylabel('E [meV]')
    
        [proj,sec] = project(RMS,1)
        PlotEllipse(ax_xe, proj, qx, qw, StyleOne)
        PlotEllipse(ax_xe, sec,  qx, qw, StyleTwo)
    
        Qxgrid = np.linspace(XMin, XMax, GridPoints)
        #Qygrid = np.linspace(YMin, YMax, GridPoints)        #np.ones(Qxgrid.shape)*qy[center]
        Qygrid = np.full_like(Qxgrid, qy[center])
    
        Hgrid  = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid  = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid  = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
             [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
             [modes,points] =  dispersion.shape
             for mode in range(modes):
                 ax_xe.plot(Qxgrid,dispersion[mode,:],StyleOne)
   
        #ye plot
        #20261010: M1 Fix: QyE panel must span YMin..YMax on x-axis (ResPlot.m:101), not XMin..XMax
        ax_ye.set_xlim(YMin, YMax)
        ax_ye.set_ylim(WMin, WMax)
    
        ax_ye.set_xlabel('Qy[A-1]')
        ax_ye.set_ylabel('E [meV]')
    
        [proj,sec]=project(RMS,0)
        PlotEllipse(ax_ye, proj, qy, qw, StyleOne)
        PlotEllipse(ax_ye, sec,  qy, qw, StyleTwo)
        
        Qygrid = np.linspace(YMin, YMax, GridPoints)
        Qxgrid = np.full_like(Qygrid, qx[center])
        

        Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
            [dispersion, intensity, gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            [modes,points]                  =  dispersion.shape
            for mode in range(modes):
                #20261010: M2 Fix: dispersion runs along the varying Qygrid (ResPlot.m:120); Qxgrid here is constant
                ax_ye.plot(Qygrid,dispersion[mode,:], StyleOne)

        #xy plot
        ax_xy.set_xlim(XMin, XMax)
        ax_xy.set_ylim(YMin, YMax)

        ax_xy.set_xlabel('Qx[A-1]')
        ax_xy.set_ylabel('Qy[A-1]')
    
        [proj,sec] = project(RMS,2)
        PlotEllipse(ax_xy, proj, qx, qy, StyleOne)
        PlotEllipse(ax_xy, sec,  qx, qy, StyleTwo)

        XWidth   =   fproject(RMS, 0)
        YWidth   =   fproject(RMS, 1)
        WWidth   =   fproject(RMS, 2)
        ZWidth   =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 3,3])

        XBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 0,0])
        YBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 1,1])
        WBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 2,2])

        matdet   =   np.linalg.det(RMS[int(center),:,:])
        
        ResVol   =   (2*np.pi)**2/np.sqrt(matdet)
        
        ax_txt.set_axis_off()
        #20261010: L1 Fix: print the 1-based point # like ResPlot.m:162 int2str(center)
        #20261010: L2 Fix: MATLAB num2str(x, N) is N SIGNIFICANT digits -> use .Ng, not .2f/.4f
        ax_txt.text(0, 1.0, 'Scan center (point # {}):'.format(center + 1))
        ax_txt.text(0, 0.88, 'H={0:.4g}  K={1:.4g}  L={2:.4g}  E={3:.4g} meV'.format(H[center], K[center], L[center], W[center]))
        ax_txt.text(0, 0.76, 'Projections on principal axes (FWHM):' )
        ax_txt.text(0, 0.64, 'Qx: {0:.4g} A-1 Qy: {1:.4g} A-1 Qz: {2:.4g} A-1'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2))
        ax_txt.text(0, 0.52, 'Bragg widths (FWHM):' )
        #20261010 Bug #2.2: third Bragg item is the ENERGY width, not Qz.
        #   WBWidth = sqrt(2*ln2)/sqrt(RMS[:,2,2]) (line 1862) = MATLAB RMS(3,3) = E in
        #   meV, but it was printed as "Qz: ... A-1". ResPlot.m:169 prints
        #   'Qx: ... A-1   Qy: ... A-1   E: ... meV'. Label/unit (Bug #2.2) and .4f->.4g precision (L2) changed.
        ax_txt.text(0, 0.40, 'Qx: {0:.4g} A-1 Qy: {1:.4g} A-1 E: {2:.4g} meV'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2))
        ax_txt.text(0, 0.28, 'Resolution volume:{0:.4e}  meV/A3'.format(ResVol*2))
        ax_txt.text(0, 0.16, 'Intensity prefactor:{0:.5g}'.format(R0[center]))
        #plt.autoscale()
        plt.show()



    def PlotlyResPlot(self, hkle=None, SMA=None, SMAp=None, output_type = 'file'):
        #This function is to plot the resolution ellipses using plotly
        if hkle is None:
            raise ValueError("hkle is Empty!")

        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")

        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")
        
        #20261010: L1 Fix: match ResPlot.m round(length(H)/2) -- (length-1)//2 is the 0-based equivalent (floor length//2 was off by one for even-length scans)
        center = (length - 1) // 2
        
        StyleOne = '-'
        StyleTwo = '--'

        XYAxesPos  = [0.1, 0.6, 0.36, 0.36]
        XEAxesPos  = [0.1, 0.1, 0.36, 0.36]
        YEAxesPos  = [0.6, 0.6, 0.36, 0.36]
        TxtAxesPos = [0.55, 0.0, 0.36, 0.36]
        GridPoints = 101
    
        self.CalcResMatHKL(hkle)
        [R0, RMS]  = [np.copy(self.R0), self.RMS.copy()]

        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        
        uq = np.vstack((H , K , L ))
        
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W

        #20261010 Bug #1.4: COPY orient1/orient2 — they alias the live sample arrays
        #   (`return self._sample.u` / `.v` at :311/:320), so the writes below used to
        #   overwrite sample.u/sample.v in place. MATLAB's `o1=EXP.orient1`
        #   (ResPlot.m:44-45) copies by value. Full explanation in ResolutionPlot (~:1731).
        #   NOT CHANGED: `pr`, the projection, the 1e-5 threshold, the zeroing logic.
        o1    = np.array(self.orient1, copy=True)
        o2    = np.array(self.orient2, copy=True)
        pr    = _scalar(o2,yvec,rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if abs(o2[0])<1e-5: o2[0] = 0
        if abs(o2[1])<1e-5: o2[1] = 0
        if abs(o2[2])<1e-5: o2[2] = 0

        if abs(o1[0])<1e-5: o1[0] = 0
        if abs(o1[1])<1e-5: o1[1] = 0
        if abs(o1[2])<1e-5: o1[2] = 0

        #========================================================================================================
        #determine the plot range
        XWidth  =  np.max(fproject(RMS,0))
        YWidth  =  np.max(fproject(RMS,1))
        WWidth  =  np.max(fproject(RMS,2))
        XMax    =  np.max(qx)+XWidth*1.5
        XMin    =  np.min(qx)-XWidth*1.5
        YMax    =  np.max(qy)+YWidth*1.5
        YMin    =  np.min(qy)-YWidth*1.5
        WMax    =  np.max(qw)+WWidth*1.5
        WMin    =  np.min(qw)-WWidth*1.5
        #========================================================================================================
       
        plotlyfig = plysub.make_subplots(rows=2, cols=2)
               
        #XE_Plot
        [proj, sec] = project(RMS, 1)
        [r1, x1, y1]=ProduceEllipse(proj, qx, qw)
        [r2, x2, y2]=ProduceEllipse(sec, qx, qw)
        
        for index in range (length):
            plotlyfig.add_trace(go.Scatter(x=x1[:,index], y=y1[:,index], line=dict(width=2)), row=1, col=1)
            plotlyfig.add_trace(go.Scatter(x=x2[:,index], y=y2[:,index], line=dict(width=2, dash='dash')), row=1, col=1)

    
        Qxgrid = np.linspace(XMin, XMax, GridPoints)
        Qygrid = np.full_like(Qxgrid, qy[center])
    
        Hgrid  = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid  = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid  = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
             [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
             [modes,points] =  dispersion.shape
             for mode in range(modes):
                 plotlyfig.add_trace(go.Scatter(Qxgrid, dispersion[mode,:], line=dict(width=2,  dash='dot'), row=1, col=1))
   
        #YE_Plot
        [proj,sec]=project(RMS,0)
        [r3,x3,y3]=ProduceEllipse(proj, qy, qw)
        [r4,x4,y4]=ProduceEllipse(sec, qy, qw)

        for index in range (length):
            plotlyfig.add_trace(go.Scatter(x=x3[:,index], y=y3[:,index], line=dict(width=2)), row=1, col=2)
            plotlyfig.add_trace(go.Scatter(x=x4[:,index], y=y4[:,index], line=dict(width=2, dash='dash')), row=1, col=2)
        
        Qygrid = np.linspace(YMin, YMax, GridPoints)
        Qxgrid = np.full_like(Qygrid, qx[center])
        
        Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
        Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
        Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
    
        if SMA is not None:
            [dispersion, intensity, gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            [modes,points]                  =  dispersion.shape
            for mode in range(modes):
                #20261010: Similar to M2 Fix: QyE panel -- dispersion runs along the varying
                #   Qygrid (ResPlot.m:120); Qxgrid here is the constant qx[center]
                plotlyfig.add_trace(go.Scatter(Qygrid, dispersion[mode,:], line=dict(width=2,  dash='dot'), row=1, col=2))
        #XY_Plot
        [proj,sec] = project(RMS,2)
        [r5,x5,y5]=ProduceEllipse(proj, qx, qy)
        [r6,x6,y6]=ProduceEllipse(sec,  qx, qy)

        for index in range (length):
            plotlyfig.add_trace(go.Scatter(x=x5[:,index], y=y5[:,index], line=dict(width=2)), row=2, col=1)
            plotlyfig.add_trace(go.Scatter(x=x6[:,index], y=y6[:,index], line=dict(width=2, dash='dash')), row=2, col=1)
        
        # Update xaxis properties
        plotlyfig.update_xaxes(title_text="Qx[A-1]", range=[XMin, XMax], row=1, col=1)
        plotlyfig.update_xaxes(title_text="Qy[A-1]", range=[YMin, YMax], row=1, col=2)
        plotlyfig.update_xaxes(title_text="Qx[A-1]", range=[XMin, XMax], row=2, col=1)
        
        # Update yaxis properties
        plotlyfig.update_yaxes(title_text="E [meV]", range=[WMin, WMax], row=1, col=1)
        plotlyfig.update_yaxes(title_text="E [meV]", range=[WMin, WMax], row=1, col=2)
        plotlyfig.update_yaxes(title_text="Qy[A-1]", range=[YMin, YMax], row=2, col=1)
       
        XWidth   =   fproject(RMS, 0)
        YWidth   =   fproject(RMS, 1)
        WWidth   =   fproject(RMS, 2)
        ZWidth   =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 3,3])

        XBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 0,0])
        YBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 1,1])
        WBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 2,2])

        matdet   =   np.linalg.det(RMS[int(center),:,:])
        
        ResVol   =   (2*np.pi)**2/np.sqrt(matdet)
        
        #20261010: L1 Fix: print the 1-based point # like ResPlot.m:162 int2str(center)
        #20261010: L2 Fix: MATLAB num2str(x, N) is N SIGNIFICANT digits -> use .Ng, not .2f/.4f
        text     ='            Scan center (point # {}):<br>'.format(center + 1)
        text=text+'            H={0:.4g}  K={1:.4g}  L={2:.4g}  E={3:.4g} meV<br>'.format(H[center], K[center], L[center], W[center])
        text=text+'            Projections on principal axes (FWHM):<br>'
        text=text+'            Qx: {0:.4g} A-1 Qy: {1:.4g} A-1 Qz: {2:.4g} A-1<br>'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2)
        text=text+'            Bragg widths (FWHM):<br>'
        #20261010 Bug #2.2: third Bragg item is the ENERGY width, not Qz.
        #   WBWidth = sqrt(2*ln2)/sqrt(RMS[:,2,2]) (line 2031) = MATLAB RMS(3,3) = E in
        #   meV, but it was printed as "Qz: ... A-1". ResPlot.m:169 prints
        #   'Qx: ... A-1   Qy: ... A-1   E: ... meV'. Label/unit (Bug #2.2) and .4f->.4g precision (L2) changed.
        text=text+'            Qx: {0:.4g} A-1 Qy: {1:.4g} A-1 E: {2:.4g} meV<br>'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2)
        text=text+'            Resolution volume:{0:.4e}  meV/A3<br>'.format(ResVol*2)
        text=text+'            Intensity prefactor:{0:.5g} <br>'.format(R0[center])

        plotlyfig.add_annotation(text=text, xref="paper", yref="paper", x=0.66, y=0.20,  align="left", showarrow=False)
      
        plotlyfig.update_layout(autosize=True, width=800, height=700, margin = dict( l=20, r=20, b=20, t=20, pad=4 ), paper_bgcolor="LightSteelBlue", showlegend = False )

        plot_div = plyplt.plot(plotlyfig, output_type = output_type)

        return plot_div


    def ResolutionPlotProj(self, ax=None, qslice='QxQy',hkle=None, SMA=None, SMAp=None):
        #This function is to plot the resolution ellipse using matplotlib
        if hkle is None:
            raise ValueError("Empty hkle!")

        #20261010: L6 Fix: validate BEFORE unpacking -- the old order (unpack first) made
        #   the len==4 check unreachable: a wrong-length hkle died at the unpack with a
        #   raw ValueError instead of the intended message. NOTE: local fix only -- 7 other methods still unpack before validating (v3 §9).
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
        H, K, L, W = hkle

        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")
        
        #20261010: L1 Fix: match ResPlot.m round(length(H)/2) -- (length-1)//2 is the 0-based equivalent (floor length//2 was off by one for even-length scans)
        center = (length - 1) // 2

        StyleOne   = '-'
        StyleTwo   = '--'
        GridPoints = 101
    
    
        # [R0,RMS]  =  ResMatS(H, K, L, W, EXP)
        self.CalcResMatHKL(hkle)
        [R0, RMS] = [np.copy(self.R0), self.RMS.copy()]

        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        uq = np.vstack((H , K , L ))

        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W

        #20261010 Bug #1.4: COPY orient1/orient2 — they alias the live sample arrays
        #   (`return self._sample.u` / `.v` at :311/:320), so the writes below used to
        #   overwrite sample.u/sample.v in place. MATLAB's `o1=EXP.orient1`
        #   (ResPlot.m:44-45) copies by value. Full explanation in ResolutionPlot (~:1731).
        #   NOT CHANGED: `pr`, the projection, the 1e-5 threshold, the zeroing logic.
        o1    = np.array(self.orient1, copy=True)
        o2    = np.array(self.orient2, copy=True)
        pr    = _scalar(o2,yvec,rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if abs(o2[0])<1e-5: o2[0] = 0
        if abs(o2[1])<1e-5: o2[1] = 0
        if abs(o2[2])<1e-5: o2[2] = 0

        if abs(o1[0])<1e-5: o1[0] = 0
        if abs(o1[1])<1e-5: o1[1] = 0
        if abs(o1[2])<1e-5: o1[2] = 0
        
        #========================================================================================================
        #determine the plot range
        XWidth  =  np.max(fproject(RMS,0))
        YWidth  =  np.max(fproject(RMS,1))
        WWidth  =  np.max(fproject(RMS,2))
        XMax    =  np.max(qx)+XWidth*1.5
        XMin    =  np.min(qx)-XWidth*1.5
        YMax    =  np.max(qy)+YWidth*1.5
        YMin    =  np.min(qy)-YWidth*1.5
        WMax    =  np.max(qw)+WWidth*1.5
        WMin    =  np.min(qw)-WWidth*1.5

        #========================================================================================================
        # plot XE projection
        if qslice == "QxE":
            #XE_Plot
            ax.set_xlim(XMin, XMax)
            ax.set_ylim(WMin, WMax)

            ax.set_xlabel('Qx[A-1]')
            ax.set_ylabel('E [meV]')
        
            [proj,sec]=project(RMS,1)
            PlotEllipse(ax, proj, qx, qw, StyleOne)
            PlotEllipse(ax, sec,  qx, qw, StyleTwo)
        
            Qxgrid = np.linspace(XMin, XMax, GridPoints)
            Qygrid = np.full_like(Qxgrid, qy[center])      #####
        
            Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
            Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
            Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
        
            if SMA is not None:
                 [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
                 [modes,points] =  dispersion.shape
                 for mode in range(modes):
                     ax.plot(Qxgrid,dispersion[mode,:],StyleOne)
   
        elif qslice == "QyE": 
            #YE plot
            #20261010: M1 Fix: QyE panel must span YMin..YMax on x-axis (ResPlot.m:101), not XMin..XMax
            ax.set_xlim(YMin, YMax)
            ax.set_ylim(WMin, WMax)

            ax.set_xlabel('Qy[A-1]')
            ax.set_ylabel('E [meV]')
        
            [proj,sec]=project(RMS,0)
            PlotEllipse(ax, proj, qy, qw, StyleOne)
            PlotEllipse(ax, sec,  qy, qw, StyleTwo)

            Qygrid = np.linspace(YMin, YMax, GridPoints)
            Qxgrid = np.full_like(Qygrid, qx[center])
            
            Hgrid = Qxgrid*xvec[0] + Qygrid*yvec[0]
            Kgrid = Qxgrid*xvec[1] + Qygrid*yvec[1]
            Lgrid = Qxgrid*xvec[2] + Qygrid*yvec[2]
        
            if SMA is not None:
                [dispersion,intensity,gamma]  =  SMA(Hgrid, Kgrid, Lgrid, SMAp)
                [modes,points]                =  dispersion.shape
                for mode in range(modes):
                    #20261010: M2 Fix: dispersion runs along the varying Qygrid (ResPlot.m:120); Qxgrid here is constant
                    ax.plot(Qygrid,dispersion[mode,:], StyleOne)

        elif qslice == "QxQy": ##### there is a bug here: when change crystals, it can cause: ValueError: Axis limits cannot be NaN or Inf, maybe its related close triangle.
            #XY plot
            ax.set_xlim(XMin, XMax)
            ax.set_ylim(YMin, YMax)
        
            ax.set_xlabel('Qx[A-1]')
            ax.set_ylabel('Qy[A-1]')
        
            [proj,sec]=project(RMS,2)
            PlotEllipse(ax, proj, qx, qy, StyleOne)
            PlotEllipse(ax, sec,  qx, qy, StyleTwo)

        elif qslice == "Txt": 

            XWidth   =   fproject(RMS, 0)
            YWidth   =   fproject(RMS, 1)
            WWidth   =   fproject(RMS, 2)
            ZWidth   =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 3,3])
    
            XBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 0,0])
            YBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 1,1])
            WBWidth  =   np.sqrt(2*np.log(2))/np.sqrt(RMS[:, 2,2])
    
            #print(center)
            matdet   =   np.linalg.det(RMS[int(center),:,:])
            
            ResVol   =   (2*np.pi)**2/np.sqrt(matdet)

            ax.set_axis_off()
            #20261010: L1 Fix: print the 1-based point # like ResPlot.m:162 int2str(center)
            #20261010: L2 Fix: MATLAB num2str(x, N) is N SIGNIFICANT digits -> use .Ng, not .2f/.4f
            strtemp='Scan center (point # {}):'.format(center + 1)
            ax.text(0, 1.0, strtemp)
            strtemp='H={0:.4g}  K={1:.4g}  L={2:.4g}  E={3:.4g} meV'.format(H[center], K[center], L[center], W[center])
            ax.text(0, 0.88, strtemp)
            strtemp='Projections on principal axes (FWHM):'
            ax.text(0, 0.76, strtemp)
            strtemp='Qx: {0:.4g} A-1 Qy: {1:.4g} A-1 Qz: {2:.4g} A-1'.format(XWidth[center]*2,YWidth[center]*2, ZWidth[center]*2)
            ax.text(0, 0.64, strtemp)
            strtemp='Bragg widths (FWHM):'
            ax.text(0, 0.52,  strtemp)
            #20261010 Bug #2.2: third Bragg item is the ENERGY width, not Qz.
            #   WBWidth = sqrt(2*ln2)/sqrt(RMS[:,2,2]) (line 2203) = MATLAB RMS(3,3) = E
            #   in meV, but it was printed as "Qz: ... A-1". ResPlot.m:169 prints
            #   'Qx: ... A-1   Qy: ... A-1   E: ... meV'. Label/unit (Bug #2.2) and .4f->.4g precision (L2) changed.
            strtemp='Qx: {0:.4g} A-1 Qy: {1:.4g} A-1 E: {2:.4g} meV'.format(XBWidth[center]*2,YBWidth[center]*2, WBWidth[center]*2)
            ax.text(0, 0.40, strtemp)
            strtemp='Resolution volume:{0:.4e}  meV/A3'.format(ResVol*2)
            ax.text(0, 0.28, strtemp)
            strtemp='Intensity prefactor:{0:.5g}'.format(R0[center])
            ax.text(0, 0.16, strtemp)     

        else:
            print("unknown qslice, please give only QxE, QyE, QxQy, or Txt")
        
        #20261010: M6 Fix: self.method may be 0/1 or a string ('popovici') -- list-index
        #   raised TypeError for strings (only AttributeError was caught); label Popovici
        #   exactly when the calculation uses it, mirroring `== 1 or == 'popovici'` elsewhere
        _m = getattr(self, "_method", 0)
        method = 'Popovici' if (_m == 1 or _m == 'popovici') else 'Cooper-Nathans'
        frame = '[Q1,Q2,Qz,E]'

        # Compute angles
        FX = 2 if getattr(self, "infin", None) != 1 else 1  

        if self.RMS.shape[0] == 1:
            NP  = self.RMS[0]
            R0  = float(self.R0[0])
        else:
            NP  = self.RMS[center]
            R0  = float(self.R0[center])
            #20261010: L3 Fix: text below must describe the same scan point as NP/R0/
            #   bragg widths (center), not point 0
            hkle = [H[center], K[center], L[center], W[center]]

        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP))
        bragg_widths = get_bragg_widths(NP)

        #20261010: M5 Fix: default text_format + getattr default so non-TAS instruments
        #   don't hit UnboundLocalError at the description_string join below
        text_format = []
        if getattr(self, "instr_type", None) == "tas":
            angles = self.get_angles_and_Q(hkle)[0]

            text_format = ['Method: {0}'.format(method),
                           'Position HKLE [{0}]\n'.format(dt.datetime.now().strftime('%d-%b-%Y %H:%M:%S')),
                           #20261010: L3 Fix: self.HKLE holds the whole scan's arrays (or a
                           #   stale default) -- print the same center point used for NP/R0
                           ' [Q_H, Q_K, Q_L, E] = {0} \n'.format(hkle),
                           'Resolution Matrix M in {0} (M/10^4):'.format(frame),
                           '[[{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]'.format(*NP[:, 0] / 1.0e4),
                           ' [{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]'.format(*NP[:, 1] / 1.0e4),
                           ' [{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]'.format(*NP[:, 2] / 1.0e4),
                           ' [{0:.4f}   {1:.4f}   {2:.4f}   {3:.4f}]]\n'.format(*NP[:, 3] / 1.0e4),
                           'Resolution volume:   V_0={0:.6f} meV/A^3'.format(2 * ResVol),
                           'Intensity prefactor: R_0={0:.3f}'.format(R0),
                            #20261010 Bug #2.2: dE and dQ_z were fed EACH OTHER'S values.
                            #   WHAT WAS WRONG
                            #     * get_bragg_widths() returns [Q1, Q2, E, phonon, Qz]
                            #       (tools.py:310-344, from RM[0,0], RM[1,1], RM[2,2],
                            #       phonon, RM[3,3] at :338-342), hence
                            #           bragg_widths[2] = E   (meV)
                            #           bragg_widths[4] = Qz  (A^-1)
                            #     * the dE slot received bragg_widths[4] -> a momentum
                            #       value printed under "dE ... [meV]";
                            #       the dQ_z slot received bragg_widths[2] -> an energy value
                            #       printed as a bare number under "dQ_z=..." (no unit on it).
                            #     * expanding *bragg_widths[2:4] into (dQ_z, V) put [2]
                            #       into dQ_z; V correctly kept [3].
                            #   WHY THIS IS THE CORRECT MINIMAL FIX
                            #     * ResPlot.m:169 puts the energy width on the energy axis
                            #       ('E: ... meV'); the axis order is [Q1,Q2,E,Qz]
                            #       (see report §9.1), so index 2 is E and index 4 is Qz.
                            #     * Only the two arguments were swapped: the format
                            #       strings, labels, units, :.3f precision and indices
                            #       [0], [1] are untouched, and the *slice was replaced
                            #       by explicit indices so the swap is visible.
                            #   NOT CHANGED
                            #     * bragg_widths[3] / the "Vanadium width V" slot and its
                            #       [meV] label — left exactly as found (not part of §2.2).
                            'Bragg width in [Q_1,Q_2,E] (FWHM):',
                            ' dQ_1={0:.3f} dQ_2={1:.3f} [A-1] dE={2:.3f} [meV]'.format(bragg_widths[0], bragg_widths[1],
                                                                                        bragg_widths[2]),
                            ' dQ_z={0:.3f} Vanadium width V={1:.3f} [meV]'.format(bragg_widths[4], bragg_widths[3]),
                           'Instrument parameters:',
                           ' DM  =  {0:.3f} ETAM= {1:.3f} SM={2}'.format(self.mono.d, self.mono.mosaic, self.mono.dir),
                           ' KFIX=  {0:.3f} FX  = {1} SS={2}'.format(Energy(energy=self.efixed).wavevector, FX,
                                                                       self.sample.dir),
                           ' DA  =  {0:.3f} ETAA= {1:.3f} SA={2}'.format(self.ana.d, self.ana.mosaic, self.ana.dir),
                           ' A1= {0:.2f} A2={1:.2f} A3={2:.2f} A4={3:.2f} A5={4:.2f} A6={5:.2f} [deg]'.format(*angles),
                           'Collimation [arcmin]:',
                           ' Horizontal: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.hcol),
                           ' Vertical: [{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}]'.format(*self.vcol),
                           'Sample:',
                           ' a, b, c  =  [{0}, {1}, {2}] [Angs]'.format(self.sample.a, self.sample.b, self.sample.c),
                           ' Alpha, Beta, Gamma  =  [{0}, {1}, {2}] [deg]'.format(self.sample.alpha, self.sample.beta,
                                                                                  self.sample.gamma),
                           ' U  =  {0} [rlu]   V  =  {1} [rlu]'.format(self.orient1, self.orient2)]        
        
        self.description_string = '\n'.join(text_format)
        
        plt.show()



    def ResolutionPlot3D(self, hkle=None, RANGE=None, EllipStyle=None, XYStyle=None, XEStyle=None, YEStyle=None, SMA=None, SMAp=None, SXg=None, SYg=None):
        #This function is to plot the resolution function ellipsoid in Q-enery space with plotly
        if hkle is None:
            print("no enough parameters for plotting!")
            return
        
        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")

        [length, H, K, L, W] = _cleanargs(H, K, L, W)  
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).") 

        #20261010: L1 Fix: match ResPlot.m round(length(H)/2) -- (length-1)//2 is the 0-based equivalent (floor length//2 was off by one for even-length scans)
        center = (length - 1) // 2     

        plotly_clr_list = ['Bluered', 'Greens','Hot', 'Jet', 'Earth' ]
        matplot_clrs    = ['r','b','g','y','c'] 
        cmap_list       = ['Spectral', 'PuOr', 'PiYG', 'BrBG', 'BuPu']
        
        SMAGridPoints   = 21
        EllipGridPoints = 21
        bPlotInQSpace   = 1
        
        #20261010: M8 Fix: raise like ResPlot3D.m:35 instead of print-and-continue
        #   (RANGE=None previously died on len(), short RANGE later on RANGE[5])
        if RANGE is None or len(RANGE) < 6:
            raise ValueError('Range must have the form [Xmin Xmax Ymin Ymax Emin Emax]')
        if EllipStyle is None: EllipStyle = 'g'
        if XYStyle    is None: XYStyle    = '-p'
        #20261010: M7 Fix: was assigning XYStyle (copy-paste), leaving XEStyle/YEStyle as None
        if XEStyle    is None: XEStyle    = '-c'
        if YEStyle    is None: YEStyle    = '-b'
        if SMA        is None: print('SMA is not provided')

        if SMA is not None and (SXg is None or SYg is None):
            SX    =    np.linspace(RANGE[0],RANGE[1],SMAGridPoints)
            SY    =    np.linspace(RANGE[2],RANGE[3],SMAGridPoints)
            [SXg, SYg] = np.meshgrid(SX,SY)

        self.CalcResMatHKL(hkle)
        RMS   =   self.RMS.copy()

        #xvec y are orient1 and orient2 vector zvec is vertical on sample
        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        
        #convert HKL into Qx(A-1) in scattering plane x and y
        uq = np.vstack([H, K, L])
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W


        plotlyfig=go.Figure()                #plotly method
        
        #plot ellipsoids
        wx=fproject(RMS,0)
        wy=fproject(RMS,1)
        ww=fproject(RMS,2)

        for point in range(length):

            xlow  = -wx[point]*1.5 + qx[point]
            xhigh =  wx[point]*1.5 + qx[point]
            ylow  = -wy[point]*1.5 + qy[point]
            yhigh =  wy[point]*1.5 + qy[point]
            xcenter  =  qx[point]
            ycenter  =  qy[point]
            zcenter  =  qw[point]
            
            if bPlotInQSpace == 0: #plot in HKL

                Hlow = xlow*xvec[0]+ylow*yvec[0]
                Klow = xlow*xvec[1]+ylow*yvec[1]
                Llow = xlow*xvec[2]+ylow*yvec[2]
                
                xlow = Hlow*self.orient1[0]+Klow*self.orient1[1]+Llow*self.orient1[2]
                ylow = Hlow*self.orient2[0]+Klow*self.orient2[1]+Llow*self.orient2[2]
                
                Hhigh=xhigh*xvec[0]+yhigh*yvec[0]
                Khigh=xhigh*xvec[1]+yhigh*yvec[1]
                Lhigh=xhigh*xvec[2]+yhigh*yvec[2]
                
                xhigh=Hhigh*self.orient1[0]+Khigh*self.orient1[1]+Lhigh*self.orient1[2]
                yhigh=Hhigh*self.orient2[0]+Khigh*self.orient2[1]+Lhigh*self.orient2[2]

                Hcenter=qx[point]*xvec[0]+qy[point]*yvec[0]
                Kcenter=qx[point]*xvec[1]+qy[point]*yvec[1]
                Lcenter=qx[point]*xvec[2]+qy[point]*yvec[2]
                
                xcenter=Hcenter*self.orient1[0]+Kcenter*self.orient1[1]+Lcenter*self.orient1[2]
                ycenter=Hcenter*self.orient2[0]+Kcenter*self.orient2[1]+Lcenter*self.orient2[2]               
            
            x = np.linspace(xlow,  xhigh, EllipGridPoints)
            y = np.linspace(ylow,  yhigh, EllipGridPoints)
            z = np.linspace(-ww[point]*1.5, ww[point]*1.5, EllipGridPoints) + zcenter
            [xg,yg,zg] = np.meshgrid(x,y,z)
            
            ee = ( RMS[point,0,0]*(xg-xcenter)**2 + 
                   RMS[point,1,1]*(yg-ycenter)**2 +
                   RMS[point,2,2]*(zg-zcenter)**2 + 
                   2*RMS[point,0,1]*(xg-xcenter)*(yg-ycenter) +
                   2*RMS[point,0,2]*(xg-xcenter)*(zg-zcenter) +
                   2*RMS[point,2,1]*(zg-zcenter)*(yg-ycenter))


            #********************use plotly to plot into web browser*****************
            plotlyfig.add_trace(go.Isosurface(
                    x=xg.flatten(), y=yg.flatten(), z=zg.flatten(), 
                    value=ee.flatten(),colorscale=plotly_clr_list[point], isomin=2*np.log(2), isomax=2*np.log(2)))
            
            plotlyfig.update_layout(
                scene=dict(
                    xaxis_title='Qx [Å⁻¹]',
                    yaxis_title='Qy [Å⁻¹]',
                    zaxis_title='E [meV]'
                )
            )

            pts = np.empty(xg.shape + (3,), dtype=float)
            pts[..., 0] = xg
            pts[..., 1] = yg/5
            pts[..., 2] = zg/100
            #print(pts[...,0].shape)
            pts = pts.transpose(0, 1, 2, 3).copy()
            pts.shape = [int(pts.size / 3), 3]
            ee = ee.T.copy()


        #plot dispersion surfaces  need to be tested
        if SMA is not None:
            Hgrid = SXg*xvec[0]+SYg*yvec[0]
            Kgrid = SXg*xvec[1]+SYg*yvec[1]
            Lgrid = SXg*xvec[2]+SYg*yvec[2]
            
            Agrid=Hgrid*self.orient1[0]+Kgrid*self.orient1[1]+Lgrid*self.orient1[2]
            Bgrid=Hgrid*self.orient2[0]+Kgrid*self.orient2[1]+Lgrid*self.orient2[2]
    
    
            [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)
            #print(Hgrid.shape)

            #20261010: M9 Fix: dispersion is indexed [x, y, mode] (SMADemo2D's .T) while the
            #   SXg/SYg meshes are (len(SY), len(SX)); go.Surface needs z == (len(y), len(x)),
            #   so transpose. Untransposed z only ran for the square 11x11 test grid and then
            #   mirrored the dispersion along the diagonal.
            [xpoints, ypoints, modes] = dispersion.shape
            for mode in range(modes):
                SZ=dispersion[:,:,mode].T
                #print(SZ)
                #if  SZ > RANGE[5]: SZ = RANGE[5]
                #if  SZ < RANGE[4]: SZ = RANGE[4]
                if bPlotInQSpace ==1:

                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis', opacity= 0.5))
                else: 

                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis', opacity= 0.5))

        #plot projections
        [proj3,sec]  =  project(RMS,2)
        [proj2,sec]  =  project(RMS,1)
        [proj1,sec]  =  project(RMS,0)
        phi =np.linspace(0.1, 2*np.pi+0.1,1001) # 1001 samples; MATLAB-style colon would give 3001 (0.1:2*pi/3000:2*pi+0.1)
        
        for i in range(length):
            r3  =  np.sqrt(2*np.log(2)/(proj3[i,0,0]*np.cos(phi)**2+proj3[i,1,1]*np.sin(phi)**2+2*proj3[i,0,1]*np.cos(phi)*np.sin(phi)))
            r2  =  np.sqrt(2*np.log(2)/(proj2[i,0,0]*np.cos(phi)**2+proj2[i,1,1]*np.sin(phi)**2+2*proj2[i,0,1]*np.cos(phi)*np.sin(phi)))
            r1  =  np.sqrt(2*np.log(2)/(proj1[i,0,0]*np.cos(phi)**2+proj1[i,1,1]*np.sin(phi)**2+2*proj1[i,0,1]*np.cos(phi)*np.sin(phi)))
            xproj3 = r3*np.cos(phi)+qx[i]
            yproj3 = r3*np.sin(phi)+qy[i]
            zproj3 = np.ones(xproj3.shape)*RANGE[4]
            xproj2 = r2*np.cos(phi)+qx[i]
            zproj2 = r2*np.sin(phi)+qw[i]
            yproj2 = np.ones(xproj2.shape)*RANGE[3]
            yproj1 = r1*np.cos(phi)+qy[i]
            zproj1 = r1*np.sin(phi)+qw[i]
            xproj1 = np.ones(yproj1.shape)*RANGE[0]

            #NOTE: the branch BELOW plots in Q (A-1); the else-branch converts to HKL
            if bPlotInQSpace == 1: #plot in Q(A-1)
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=xproj1,y=yproj1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj2,y=yproj2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj3,y=yproj3,z=zproj3, mode='lines'))
                #************************************************************************

            else: #plot in HKL
                H3 = xproj3*xvec[0]+yproj3*yvec[0]
                K3 = xproj3*xvec[1]+yproj3*yvec[1]
                L3 = xproj3*xvec[2]+yproj3*yvec[2]
                A3 = H3*self.orient1[0]+K3*self.orient1[1]+L3*self.orient1[2]
                B3 = H3*self.orient2[0]+K3*self.orient2[1]+L3*self.orient2[2]
    
                H2 = xproj2*xvec[0]+yproj2*yvec[0]
                K2 = xproj2*xvec[1]+yproj2*yvec[1]
                L2 = xproj2*xvec[2]+yproj2*yvec[2]
                A2 = H2*self.orient1[0]+K2*self.orient1[1]+L2*self.orient1[2]
                B2 = H2*self.orient2[0]+K2*self.orient2[1]+L2*self.orient2[2]
    
                H1 = xproj1*xvec[0]+yproj1*yvec[0]
                K1 = xproj1*xvec[1]+yproj1*yvec[1]
                L1 = xproj1*xvec[2]+yproj1*yvec[2] 
                A1 = H1*self.orient1[0]+K1*self.orient1[1]+L1*self.orient1[2]
                B1 = H1*self.orient2[0]+K1*self.orient2[1]+L1*self.orient2[2]
           
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=A1,y=B1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A2,y=B2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A3,y=B3,z=zproj3, mode='lines'))
                #************************************************************************

        plyplt.plot(plotlyfig)

    def PlotlyResPlot3D(self, hkle=None, RANGE=None, EllipStyle=None, XYStyle=None, XEStyle=None, YEStyle=None, SMA=None, SMAp=None, SXg=None, SYg=None, output_type='file'):
        # this function use plotly to plot the resolution of a tas instrument
        if hkle is None:
            print("no enough parameters for plotting!")
            return

        H, K, L, W = hkle
        if not (isinstance(hkle, (list, tuple, np.ndarray)) and len(hkle) == 4):
            raise ValueError("hkle must be a sequence of exactly 4 elements (H, K, L, W).")
            
        [length, H, K, L, W] = _cleanargs(H, K, L, W)
        if length == 0:
            raise ValueError("Invalid hkle: after cleaning, length is 0 (empty or invalid inputs).")
        
        #20261010: L1 Fix: match ResPlot.m round(length(H)/2) -- (length-1)//2 is the 0-based equivalent (floor length//2 was off by one for even-length scans)
        center = (length - 1) // 2

        plotly_clr_list = ['Bluered', 'Greens','Hot', 'Jet', 'Earth' ]

        SMAGridPoints   = 21
        EllipGridPoints = 21
        
        bPlotInQSpace   = 1
        
        #20261010: M8 Fix: raise like ResPlot3D.m:35 instead of print-and-continue
        #   (RANGE=None previously died on len(), short RANGE later on RANGE[5])
        if RANGE is None or len(RANGE) < 6:
            raise ValueError('Range must have the form [Xmin Xmax Ymin Ymax Emin Emax]')
        if EllipStyle is None: EllipStyle = 'g'
        if XYStyle    is None: XYStyle    = '-p'
        #20261010: M7 Fix: was assigning XYStyle (copy-paste), leaving XEStyle/YEStyle as None
        if XEStyle    is None: XEStyle    = '-c'
        if YEStyle    is None: YEStyle    = '-b'
        if SMA        is None: print('SMA is not provided')

        if SMA is not None and (SXg is None or SYg is None):
            SX = np.linspace(RANGE[0], RANGE[1], SMAGridPoints)
            SY = np.linspace(RANGE[2], RANGE[3], SMAGridPoints)
            [SXg, SYg] = np.meshgrid(SX,SY)


        self.CalcResMatHKL(hkle)
        RMS   =   self.RMS.copy()

        #xvec y are orient1 and orient2 vector zvec is vertical on sample
        [xvec,yvec,zvec,sample,rsample]  =  self._StandardSystem()
        
        #convert HKL into Qx(A-1) in scattering plane x and y
        uq = np.vstack([H, K, L])
        qx = _scalar(xvec, uq, rsample)
        qy = _scalar(yvec, uq, rsample)
        qw = W

        plotlyfig=go.Figure()  #plotly method
        
        #plot ellipsoids
        wx=fproject(RMS,0)
        wy=fproject(RMS,1)
        ww=fproject(RMS,2)

        for point in range(length):

            xlow  = -wx[point]*1.5 + qx[point]
            xhigh =  wx[point]*1.5 + qx[point]
            ylow  = -wy[point]*1.5 + qy[point]
            yhigh =  wy[point]*1.5 + qy[point]
            xcenter  =  qx[point]
            ycenter  =  qy[point]
            zcenter  =  qw[point]
            
            if bPlotInQSpace == 0:             #plot in HKL

                Hlow = xlow*xvec[0]+ylow*yvec[0]
                Klow = xlow*xvec[1]+ylow*yvec[1]
                Llow = xlow*xvec[2]+ylow*yvec[2]
                
                xlow = Hlow*self.orient1[0]+Klow*self.orient1[1]+Llow*self.orient1[2]
                ylow = Hlow*self.orient2[0]+Klow*self.orient2[1]+Llow*self.orient2[2]
                
                Hhigh=xhigh*xvec[0]+yhigh*yvec[0]
                Khigh=xhigh*xvec[1]+yhigh*yvec[1]
                Lhigh=xhigh*xvec[2]+yhigh*yvec[2]
                
                xhigh=Hhigh*self.orient1[0]+Khigh*self.orient1[1]+Lhigh*self.orient1[2]
                yhigh=Hhigh*self.orient2[0]+Khigh*self.orient2[1]+Lhigh*self.orient2[2]

                Hcenter=qx[point]*xvec[0]+qy[point]*yvec[0]
                Kcenter=qx[point]*xvec[1]+qy[point]*yvec[1]
                Lcenter=qx[point]*xvec[2]+qy[point]*yvec[2]
                
                xcenter=Hcenter*self.orient1[0]+Kcenter*self.orient1[1]+Lcenter*self.orient1[2]
                ycenter=Hcenter*self.orient2[0]+Kcenter*self.orient2[1]+Lcenter*self.orient2[2]               
            
            x = np.linspace(xlow,  xhigh, EllipGridPoints)
            y = np.linspace(ylow,  yhigh, EllipGridPoints)
            z = np.linspace(-ww[point]*1.5, ww[point]*1.5, EllipGridPoints) + zcenter
            [xg,yg,zg] = np.meshgrid(x,y,z)

            
            ee = ( RMS[point,0,0]*(xg-xcenter)**2 + 
                   RMS[point,1,1]*(yg-ycenter)**2 +
                   RMS[point,2,2]*(zg-zcenter)**2 + 
                   2*RMS[point,0,1]*(xg-xcenter)*(yg-ycenter) +
                   2*RMS[point,0,2]*(xg-xcenter)*(zg-zcenter) +
                   2*RMS[point,2,1]*(zg-zcenter)*(yg-ycenter))

            #********************use plotly to plot into web browser*****************
            plotlyfig.add_trace(go.Isosurface(
                    x=xg.flatten(), y=yg.flatten(), z=zg.flatten(), 
                    value=ee.flatten(),colorscale=plotly_clr_list[point], isomin=2*np.log(2), isomax=2*np.log(2)))
            
            plotlyfig.update_layout(xaxis_title='Qx [A-1]')
            plotlyfig.update_yaxes(title_text='Qy [A-1]')
            #plotlyfig.update_zaxes(title_text='E [meV]')
            #************************************************************************            
        
        #plot dispersion surfaces  need to be tested
        if SMA is not None:
            Hgrid = SXg*xvec[0]+SYg*yvec[0]
            Kgrid = SXg*xvec[1]+SYg*yvec[1]
            Lgrid = SXg*xvec[2]+SYg*yvec[2]
            
            Agrid=Hgrid*self.orient1[0]+Kgrid*self.orient1[1]+Lgrid*self.orient1[2]
            Bgrid=Hgrid*self.orient2[0]+Kgrid*self.orient2[1]+Lgrid*self.orient2[2]
    
            [dispersion,intensity,gamma]  =  SMA(Hgrid,Kgrid,Lgrid,SMAp)

            #20261010: M9 Fix: dispersion is indexed [x, y, mode] (SMADemo2D's .T) while the
            #   SXg/SYg meshes are (len(SY), len(SX)); go.Surface needs z == (len(y), len(x)),
            #   so transpose. Untransposed z only ran for the square 11x11 test grid and then
            #   mirrored the dispersion along the diagonal.
            [xpoints, ypoints, modes] = dispersion.shape
            for mode in range(modes):
                SZ=dispersion[:,:,mode].T

                if bPlotInQSpace ==1:
                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis'))
                else: 
                    plotlyfig.add_trace(go.Surface(x=SXg, y=SYg, z=SZ, colorscale = 'Viridis'))

        #plot projections
        [proj3,sec]  =  project(RMS,2)
        [proj2,sec]  =  project(RMS,1)
        [proj1,sec]  =  project(RMS,0)
        phi =np.linspace(0.1, 2*np.pi+0.1,1001)                  # 1001 samples; MATLAB-style colon would give 3001 (0.1:2*pi/3000:2*pi+0.1)
        
        for i in range(length):
            r3     =  np.sqrt(2*np.log(2)/(proj3[i,0,0]*np.cos(phi)**2+proj3[i,1,1]*np.sin(phi)**2+2*proj3[i,0,1]*np.cos(phi)*np.sin(phi)))
            r2     =  np.sqrt(2*np.log(2)/(proj2[i,0,0]*np.cos(phi)**2+proj2[i,1,1]*np.sin(phi)**2+2*proj2[i,0,1]*np.cos(phi)*np.sin(phi)))
            r1     =  np.sqrt(2*np.log(2)/(proj1[i,0,0]*np.cos(phi)**2+proj1[i,1,1]*np.sin(phi)**2+2*proj1[i,0,1]*np.cos(phi)*np.sin(phi)))
            xproj3 = r3*np.cos(phi)+qx[i]
            yproj3 = r3*np.sin(phi)+qy[i]
            zproj3 = np.ones(xproj3.shape)*RANGE[4]
            xproj2 = r2*np.cos(phi)+qx[i]
            zproj2 = r2*np.sin(phi)+qw[i]
            yproj2 = np.ones(xproj2.shape)*RANGE[3]
            yproj1 = r1*np.cos(phi)+qy[i]
            zproj1 = r1*np.sin(phi)+qw[i]
            xproj1 = np.ones(yproj1.shape)*RANGE[0]

            #NOTE: the branch BELOW plots in Q (A-1); the else-branch converts to HKL
            if bPlotInQSpace == 1: #plot in Q(A-1)
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=xproj1,y=yproj1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj2,y=yproj2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=xproj3,y=yproj3,z=zproj3, mode='lines'))
                #************************************************************************

            else: #plot in HKL
                H3 = xproj3*xvec[0]+yproj3*yvec[0]
                K3 = xproj3*xvec[1]+yproj3*yvec[1]
                L3 = xproj3*xvec[2]+yproj3*yvec[2]
                A3 = H3*self.orient1[0]+K3*self.orient1[1]+L3*self.orient1[2]
                B3 = H3*self.orient2[0]+K3*self.orient2[1]+L3*self.orient2[2]
    
                H2 = xproj2*xvec[0]+yproj2*yvec[0]
                K2 = xproj2*xvec[1]+yproj2*yvec[1]
                L2 = xproj2*xvec[2]+yproj2*yvec[2]
                A2 = H2*self.orient1[0]+K2*self.orient1[1]+L2*self.orient1[2]
                B2 = H2*self.orient2[0]+K2*self.orient2[1]+L2*self.orient2[2]
    
                H1 = xproj1*xvec[0]+yproj1*yvec[0]
                K1 = xproj1*xvec[1]+yproj1*yvec[1]
                L1 = xproj1*xvec[2]+yproj1*yvec[2] 
                A1 = H1*self.orient1[0]+K1*self.orient1[1]+L1*self.orient1[2]
                B1 = H1*self.orient2[0]+K1*self.orient2[1]+L1*self.orient2[2]
           
                #********************use plotly to plot into web browser*****************
                plotlyfig.add_trace(go.Scatter3d(x=A1,y=B1,z=zproj1, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A2,y=B2,z=zproj2, mode='lines'))
                plotlyfig.add_trace(go.Scatter3d(x=A3,y=B3,z=zproj3, mode='lines'))
                #************************************************************************
        plot_div = plyplt.plot(plotlyfig, output_type = output_type)
        return plot_div
  
        
    def S2R(self, qx, qy, qz):
        #===================================================================================
        # Given cartesian coordinates of a vector in the S-system, calculate its
        # reciprocal-space coordinates (Miller indexes).
        [xvec,yvec,zvec, sample, rsample]=self._StandardSystem()
        H=qx*xvec[0] + qy*yvec[0]+qz*zvec[0]
        K=qx*xvec[1] + qy*yvec[1]+qz*zvec[1]
        L=qx*xvec[2] + qy*yvec[2]+qz*zvec[2]
        q = np.sqrt(qx**2 + qy**2 + qz**2)
        
        return [H, K, L, q]
    
    
    def R2S(self, H,K,L):
        # Given reciprocal-space coordinates of a vector, calculates its cartesian
        #coordinates in the S-System.
        uq = np.vstack([H, K, L])    
        [xvec,yvec,zvec,sample,rsample]=self._StandardSystem()
        qx = _scalar(uq,xvec,rsample)
        qy = _scalar(uq,yvec,rsample)
        qz = _scalar(uq,zvec,rsample)
        q  = _modvec(uq,rsample)
        
        return [qx, qy, qz, q]
    
 
#########################################################################################
# the following are functions which do not belong to the class

def  fproject(mat, i):
    
    if i==0:
        v,j=2,1
    elif i==1:
        v,j=0,2
    elif i==2:
        v,j=0,1
    else:
        print("Error: improper i value!, it should be 0 or 1 or 2")
        return
    if not isinstance(mat, np.ndarray):
        mat=np.array(mat)
    if mat.ndim<3:
        mat=np.reshape(mat, [-1, mat.shape[0],mat.shape[1]])    
    
    [a,b,c]=mat.shape

    proj=np.zeros([a,2,2])

    proj[:,0,0] = mat[:,i,i] - mat[:,i,v]**2/mat[:,v,v]
    proj[:,0,1] = mat[:,i,j] - mat[:,i,v]*mat[:,j,v]/mat[:,v,v]
    proj[:,1,0] = mat[:,j,i] - mat[:,j,v]*mat[:,i,v]/mat[:,v,v]
    proj[:,1,1] = mat[:,j,j] - mat[:,j,v]**2/mat[:,v,v]
    hwhm = proj[:,0,0]-proj[:,0,1]**2/proj[:,1,1]

    hwhm = np.sqrt(2*np.log(2))/np.sqrt(hwhm)

    return hwhm
    
def  project(mat, v):
    #RMS shape in neutronpy is (n,4,4) different from the matlab Reslib34.
    #thus, we have some changes here in order to keep this 
    if v == 2:
        i,j=0,1
    elif v == 0:
        i,j=1,2
    elif v == 1:
        i,j=0,2
    else:
        print("Error: improper i value!, it should be 0 or 1 or 2")
        return
    if not isinstance(mat, np.ndarray):
        mat=np.array(mat)
    if mat.ndim<3:
        mat=np.reshape(mat, [-1, mat.shape[0],mat.shape[1]])
 
    [a,b,c]=mat.shape

    proj=np.zeros([a,2,2])
    sec =np.zeros([a,2,2])
    proj[:,0,0] = mat[:,i,i] - mat[:,i,v]**2/mat[:,v,v]
    proj[:,0,1] = mat[:,i,j] - mat[:,i,v]*mat[:,j,v]/mat[:,v,v]
    proj[:,1,0] = mat[:,j,i] - mat[:,j,v]*mat[:,i,v]/mat[:,v,v]
    proj[:,1,1] = mat[:,j,j] - mat[:,j,v]**2/mat[:,v,v]
    sec[:,0,0]  = mat[:,i,i]
    sec[:,0,1]  = mat[:,i,j]
    sec[:,1,0]  = mat[:,j,i]
    sec[:,1,1]  = mat[:,j,j]

    return [proj, sec]


def PlotEllipse(ax,mat,x0,y0,style):
    
    if ax is None:
        print("ax is not valid")
        fig=plt.figure()
        ax=fig.add_subplot(111)
        
    [a,b,c]=mat.shape
    
    phi=np.linspace(0, 2*np.pi,1000)
    for i in range(a):
        r = np.sqrt(2*np.log(2)/(mat[i,0,0]*np.cos(phi)**2+mat[i,1,1]*np.sin(phi)**2+2*mat[i,0,1]*np.cos(phi)*np.sin(phi)))
        x = r*np.cos(phi)+x0[i]
        y = r*np.sin(phi)+y0[i]

        ax.plot(x,y,style)
        ax.autoscale()
        #plt.show()


def ProduceEllipse(mat,x0,y0):
            
    [a,b,c]=mat.shape
    r=np.zeros(shape=(1000, a))
    x=np.zeros(shape=(1000, a))
    y=np.zeros(shape=(1000, a))
    
    phi=np.linspace(0, 2*np.pi,1000)
    for i in range(a):
        r[:,i] = np.sqrt(2*np.log(2)/(mat[i,0,0]*np.cos(phi)**2+mat[i,1,1]*np.sin(phi)**2+2*mat[i,0,1]*np.cos(phi)*np.sin(phi)))
        x[:,i]  = r[:,i]*np.cos(phi)+x0[i]
        y[:,i]  = r[:,i]*np.sin(phi)+y0[i]

    return [r, x, y]