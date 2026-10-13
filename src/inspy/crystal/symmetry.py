# -*- coding: utf-8 -*-
# This file is adapted from neutronpy

# Symmetry operations

import numpy as np
from ..constants import symmetry

space_groups = symmetry()['space_groups']


class SpaceGroup(object):
    # Spacegroup class for sample symmetry
    def __init__(self, symbol='P1'):
        if isinstance(symbol, int):
            for key, value in space_groups.items():
                if value['number'] == symbol:
                    self._symbol = key
        elif isinstance(symbol, str):
            if symbol in space_groups:
                self._symbol = symbol
            else:
                for key, value in space_groups.items():
                    if value['hermann-manguin_symbol'] == symbol or value['full_name'] == symbol:
                        self._symbol = key
        else:
            raise KeyError('{0} is not a valid International symbol, Hermann–Mauguin symbol, or space group number'.format(symbol))

        self.point_group = space_groups[self.symbol]['point_group']
        self.full_name = space_groups[self.symbol]['full_name']
        self._generators_str = space_groups[self.symbol]['generators']
        self.lattice_type = space_groups[self.symbol]['type']
        self.group_number = space_groups[self.symbol]['number']
        self.hm_symbol = space_groups[self.symbol]['hermann-manguin_symbol']
        self._generators_mat = get_generator_from_str(self._generators_str)
        self.total_operations = space_groups[self.symbol]['total_operations']
        self.symmetry_operations = self._symmetry_operations_from_generators()

    def __repr__(self):
        return "SpaceGroup({0})".format(self.group_number)

    @property
    def symbol(self):
        #Space group symbol
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self.__init__(symbol)

    @property
    def string_generators(self):
        #Space group generators
        return self._generators_str

    @property
    def generators(self):
        #Space group generators in matrix format
        return self._generators_mat

    def _symmetry_operations_from_generators(self):
        symm_ops, new_ops = [np.copy(self.generators)] * 2
        while len(new_ops) > 0 and len(symm_ops) < self.total_operations:
            gen_ops = []
            for g in new_ops:
                test_ops = np.einsum('ijk,kl', symm_ops, g)
                for op in test_ops:
                    op[:3, 3] = np.mod(get_translation(op), 1)
                    op[np.where(np.abs(1 - get_translation(op)) < 1e-15), 3] = 0
                    if not (np.abs(symm_ops - op) < 1e-15).all(axis=(1, 2)).any():
                        gen_ops.append(op)
                        symm_ops = np.append(symm_ops, [op], axis=0)
            new_ops = gen_ops
        assert len(symm_ops) == self.total_operations
        return symm_ops

    def symmetrize_position(self, vector):
        #Applies symmetry operations to a vector
        positions = []
        for op in self.symmetry_operations:
            positions.append(np.dot(get_rotation(op), np.array(vector)) + get_translation(op))

        #20261010 Bug #2.4: fold each image into [0,1)^3 and drop coincident duplicates.
        #   WHAT WAS WRONG
        #     * the loop above pushed the raw `R.v + t` for EVERY one of the group's
        #       `total_operations` (48 for a primitive cubic group, 192 for an F-centred
        #       one), with
        #         - no `mod 1`, so images landed at e.g. -0.25 or 1.75 instead of in
        #           [0,1)^3, and
        #         - no de-duplication, so every operation in the site's stabiliser
        #           (order = |G| / multiplicity) returned the SAME coordinate again.
        #     * `Material.__init__` (material.py:89-97) and `MagneticUnitCell.__init__`
        #       (material.py:29-34) append one `Atom` per returned position, so a
        #       multiplicity-4 site in a 192-operation group produced 192 `Atom`s where
        #       4 are unique -- a 48x inflation (measured total cross-section 670.08 b
        #       instead of 13.96 b).
        #     * that sum is exactly what `total_scattering_cross_section`,
        #       `calc_optimal_thickness`, `calc_incoh_elas_xs` and
        #       `NuclearStructureFactor.calc_nuc_str_fac` (structure_factors.py:29-32)
        #       iterate over, so structure factors and every cross-section / absorption
        #       number were wrong whenever `space_group` was supplied.
        #   WHY THIS IS THE CORRECT MINIMAL FIX
        #     * `np.mod(x, 1)` + snapping `1 -> 0` is the SAME convention already used
        #       for the translations of the operations themselves at lines 68-69, so
        #       operations and positions now fold consistently.
        #     * `np.round(..., 8)` first quantises away FP noise (~1e-16) from the
        #       `R.v + t` arithmetic, so the `== 1.0` test afterwards is exact and two
        #       images of one site compare equal bit-for-bit.
        #     * dedup uses `np.allclose(atol=1e-8, rtol=0)`. That is comfortably above
        #       the ~1e-16 the stabiliser can introduce (no catastrophic cancellation is
        #       possible for fractional coords of order 1) and comfortably below any
        #       physically distinct pair of sites, so nothing real is ever merged.
        #     * FIRST occurrence wins, so the order follows `self.symmetry_operations`
        #       and the result is deterministic -- no set/hash iteration order.
        #     * `mod 1` is mathematically a no-op for the structure-factor phase term
        #       `exp(1j*2*pi*(h*x + k*y + l*z))` (structure_factors.py:30), which is
        #       periodic under x -> x+n. It only canonicalises the coordinates and
        #       makes the space_group path agree with the no-space_group path
        #       (material.py:100-113), where user-supplied positions are used as-is.
        #   NOT CHANGED
        #     * the `R.v + t` arithmetic, the return type (a list of `np.ndarray`),
        #       `symmetry_operations`, `_symmetry_operations_from_generators`, the
        #       generator/rotation/translation helpers, and both call sites
        #       (material.py:29 and :89) -- they still append one `Atom` per position
        #       they are handed, which is correct once only unique images are returned.
        #   NOT FIXED HERE (pre-existing, outside §2.4)
        #     * structure_factors.py:31-32 and material.py:225 pass `atom.pos` (a
        #       fractional POSITION) to `get_two_theta(hkl, ...)` as though it were a
        #       Miller index, then apply a DW term (sf: Uaniso; material: Uiso only).
        #       Folding `pos` would change those terms' inputs, but both were already meaningless --
        #       that is a separate defect, not this one.
        unique = []
        for pos in positions:
            pos = np.round(np.mod(pos, 1.0), 8)
            pos[pos == 1.0] = 0.0
            if not any(np.allclose(pos, other, atol=1e-8, rtol=0) for other in unique):
                unique.append(pos)

        return unique


def get_formatted_operations(operations):
    #Returns operations formatted in a list for easy parsing

    if (isinstance(operations, list) and isinstance(operations[0], str)) or isinstance(operations, str):
        operations = get_generator_from_str(operations)

    if isinstance(operations, np.ndarray):
        operations = [operations]

    return operations


def get_rotation(operations):
    #Returns rotational part of operator
    rotations = []
    for operation in get_formatted_operations(operations):
        rotations.append(operation[:3, :3])

    if len(rotations) == 1:
        rotations = rotations[0]

    return rotations


def get_translation(operations):
    #Returns translation part of operator (3rd column [:3,3])
    translations = []
    for operation in get_formatted_operations(operations):
        translations.append(operation[:3, 3])

    if len(translations) == 1:
        translations = translations[0]

    return translations


def get_generator_from_str(operations):
    #Returns generator arrays

    if isinstance(operations, str):
        operations = [operations]

    operators = []
    for gen in operations:
        components = gen.split(',')
        if len(components) > 3:
            raise ValueError('Generator string {0} is in wrong format'.format(gen))

        rotation = np.zeros((3, 3))
        translation = np.zeros(3)
        for i, comp in enumerate(components):
            elements = comp.split('+')
            if len(elements) > 1:
                from fractions import Fraction
                translation[i] = float(Fraction(elements[-1]))

            if '-x' in elements[0]:
                rotation[i, 0] = -1
            elif 'x' in elements[0]:
                rotation[i, 0] = 1

            if '-y' in elements[0]:
                rotation[i, 1] = -1
            elif 'y' in elements[0]:
                rotation[i, 1] = 1

            if '-z' in elements[0]:
                rotation[i, 2] = -1
            elif 'z' in elements[0]:
                rotation[i, 2] = 1

        out = np.zeros((4, 4), dtype=float)
        out[0:3, 0:3] = rotation
        out[0:3, 3] = translation
        out[3, 3] = 1.

        operators.append(out)

    if len(operators) == 1:
        operators = operators

    return operators


def get_str_from_generator(operations):
    #Returns strings of generators from arrays

    if isinstance(operations, np.ndarray) and len(operations.shape) < 3:
        operations = [operations]

    syms = ['x', 'y', 'z']
    signs = {-1: '-', 1: '+', 0: ''}
    generators = []
    for operation in operations:
        line = []
        for row in operation[:3, :]:
            element = ''
            for col, sym in zip(row[:3], syms):
                element += signs[int(col)] + np.abs(int(col)) * sym

            if row[3] == 0:
                translate = ''
            elif np.round(1. / row[3], 1) == 1.5:
                translate = '2/3'
            elif row[3] == 0.75:
                translate = '3/4'
            elif np.round(row[3], 3) == 0.833:
                translate = '5/6'
            else:
                denominator = int(np.round(1. / row[3]))
                translate = '1/{0}'.format(denominator)

            if translate != '':
                element += '+{0}'.format(translate)

            if len(element) >= 1 and element[0] == '+':
                element = element[1:]

            line.append(element)

        generators.append(','.join(line))

    return generators
