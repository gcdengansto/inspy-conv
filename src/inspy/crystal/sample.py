# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

#Sample class (lattice + orientation/mosaic) used by TripleAxisSpectr

import numpy as np
from .lattice import Lattice

def gram_schmidt(in_vecs, row_vecs=True, normalize=True):
    # Orthonormalises a set of row vectors (Gram-Schmidt); callers take the normal via np.cross.
    if not row_vecs:
        in_vecs = in_vecs.T

    out_vecs = in_vecs[0:1, :]

    for i in range(1, in_vecs.shape[0]):
        proj = np.diag((in_vecs[i, :].dot(out_vecs.T) / np.linalg.norm(out_vecs, axis=1) ** 2).flat).dot(out_vecs)
        out_vecs = np.vstack((out_vecs, in_vecs[i, :] - proj.sum(0)))

    if normalize:
        out_vecs = np.diag(1 / np.linalg.norm(out_vecs, axis=1)).dot(out_vecs)

    if not row_vecs:
        out_vecs = out_vecs.T

    return out_vecs


    

class Sample(Lattice):
    #Sample class 

    def __init__(self, a, b, c, alpha, beta, gamma, u=None, v=None, mosaic=None, vmosaic=None, direct=1,
                 width=None, height=None, depth=None, shape='rectangular', distance=None):
        super(Sample, self).__init__(a, b, c, alpha, beta, gamma)
        if u is not None:
            self._u = np.array(u)
        else:
            self._u = np.array([1,0,0])
        if v is not None:
            self._v = np.array(v)
        else:
            self._v = np.array([0,0,1])
            

        if mosaic is not None:
            self.mosaic = mosaic
        if vmosaic is not None:
            self.vmosaic = vmosaic

        self.dir = direct
        self.shape_type = shape

        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        if depth is not None:
            self.depth = depth
        if distance is not None:
            self.distance = distance

    def __repr__(self):
        args = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
        kwargs = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in
                            ['u', 'v', 'mosaic', 'vmosaic', 'direct', 'width', 'height', 'depth', 'shape'] if
                            getattr(self, key, None) is not None])
        return "Sample({0}, {1})".format(args, kwargs)

    def __eq__(self, right):
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if not np.all(self_parent_keys == right_parent_keys):
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            if not np.all(value == right_parent_val):
                print(value, right_parent_val)
                return False

        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    @property
    def u(self):
        r"""First orientation vector
        """
        return self._u

    @u.setter
    def u(self, vec):
        self._u = np.array(vec)

    @property
    def v(self):
        r"""Second orientation vector
        """
        return self._v

    @v.setter
    def v(self, vec):
        self._v = np.array(vec)

    @property
    def direct(self):
        return self._dir

    @direct.setter
    def direct(self, value):
        self._dir = value

    @property
    def Umatrix(self):
        r"""Rotation matrix that rotates the sample's reference frame into the spectrometer's
        """
        ortho_basis = gram_schmidt(np.vstack((self.u, self.v)))

        return np.vstack((ortho_basis, np.cross(ortho_basis[0], ortho_basis[1])))

    @property
    def UBmatrix(self):
        r"""Orientation matrix of the sample
        """
        return self.Umatrix @ self.Bmatrix

    def get_phi(self, Q):
        u"""Get out-of-plane scattering angle.
        Parameters
        ----------
        Q : array_like
            Scattering vector (same frame as u, v); no wavelength is used.
        Returns
        -------
        phi : float
            The out-of-plane angle
        """
        return self.get_angle_between_planes(Q, np.cross(self.u, self.v))
