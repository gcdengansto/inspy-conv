# -*- coding: utf-8 -*-
"""Crystal-structure tests: Sample, Lattice, Material, SpaceGroup, structure factors."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import inspy.constants as C
from inspy.crystal import (Atom, Lattice, Material, NuclearStructureFactor, Sample,
                           SpaceGroup)


class TestLattice:
    def test_cubic_reciprocal_is_2pi_over_a(self):
        lat = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        assert lat.astar == pytest.approx(2 * np.pi / 5.0)
        assert lat.volume == pytest.approx(125.0)
        assert lat.reciprocal_volume == pytest.approx((2 * np.pi) ** 3 / 125.0)

    def test_astd_divides_out_the_2pi(self):
        """astar is 2*pi-scaled (a wavevector); astar_std is the 1/d convention."""
        lat = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        assert lat.astar == pytest.approx(2 * np.pi * lat.astar_std)

    def test_bmatrix_reproduces_the_metric_tensor(self):
        """B.T @ B == Gstar is the defining property of B (item 74)."""
        lat = Lattice(3.0, 4.0, 5.0, 80, 90, 100)
        assert np.allclose(lat.Bmatrix.T @ lat.Bmatrix, lat.Gstar, atol=1e-12)

    def test_get_d_spacing_removes_the_2pi(self):
        lat = Lattice(3.0, 4.0, 5.0, 90, 90, 90)
        assert lat.get_d_spacing([1, 0, 0]) == pytest.approx(3.0)
        assert lat.get_d_spacing([0, 0, 2]) == pytest.approx(2.5)

    @pytest.mark.parametrize('abc,abg,expected', [
        ((5, 5, 5), (90, 90, 90), 'cubic'),
        ((5, 5, 8), (90, 90, 90), 'tetragonal'),      # used to fall through to ValueError
        ((5, 6, 7), (90, 90, 90), 'orthorhombic'),
        ((5, 5, 8), (90, 90, 120), 'hexagonal'),
        ((5, 6, 7), (90, 100, 90), 'monoclinic'),     # a-unique and c-unique both work
        ((5, 6, 7), (80, 100, 110), 'triclinic'),
        ((5, 5, 5), (60, 60, 60), 'rhombohedral'),
    ])
    def test_lattice_type_classification(self, abc, abg, expected):
        """Every Bravais system must be reachable, including the ordinary tetragonal
        cell (5, 5, 8) that the old exact-equality test misclassified (item 172)."""
        assert Lattice(*abc, *abg).lattice_type == expected


class TestSample:
    def test_plotting_does_not_corrupt_the_orientation_vectors(self, tas):
        """The plotting code zeroes components of the orientation vectors.

        It used to do that on the *live* sample.u/sample.v arrays, so every call to
        load_instrument in a GUI progressively destroyed the sample orientation
        (item 1). The properties still return the live array by design; the callers
        must copy, so that is what is checked here.
        """
        before_u = np.array(tas.orient1, dtype=float)
        before_v = np.array(tas.orient2, dtype=float)
        for qslice in ('QxE', 'QyE', 'QxQy'):
            ax = plt.figure().add_subplot(111)
            tas.ResolutionPlotProj(ax=ax, qslice=qslice, hkle=[2.0, 0.0, 0.0, 0.0])
            plt.close('all')
        assert np.array_equal(tas.orient1, before_u)
        assert np.array_equal(tas.orient2, before_v)

    def test_sample_u_defaults_to_a_valid_orientation(self):
        s = Sample(3, 4, 5, 90, 90, 90)
        assert len(s.u) == 3 and len(s.v) == 3


class TestSpaceGroup:
    def test_every_space_group_has_consistent_operations(self):
        """The count of unique operations must match the declared total."""
        db = C.symmetry()['space_groups']
        assert len(db) == 230
        for name in sorted(db):
            sg = SpaceGroup(name)
            assert len(sg.symmetry_operations) == sg.total_operations, name

    def test_symmetry_operations_are_unique_including_translations(self):
        """The 4x4 matrices must be distinct once the translation column counts.

        Fd-3m has 192 operations, of which only 48 are point-group rotations; the
        other 144 are the same rotations combined with the face-centring translations.
        Comparing only the 3x3 rotation block would therefore report 192 - 48 = 144
        "duplicates" that are not duplicates at all.
        """
        sg = SpaceGroup('Fd-3m')
        mats = [tuple(np.round(np.asarray(op).ravel(), 9)) for op in sg.symmetry_operations]
        assert len(mats) == len(set(mats)) == 192

    def test_symmetry_operations_are_closed_under_composition(self):
        """The operation set must be a group: multiplying any two gives another.

        Translation components are reduced modulo 1 before comparison, because that
        is how they are stored (0.5 + 0.5 is the same lattice translation as 0.0).
        """
        sg = SpaceGroup('Fm-3m')
        ops = [np.asarray(op, dtype=float) for op in sg.symmetry_operations]

        def key(mat):
            m = mat.copy()
            m[:3, 3] = np.mod(m[:3, 3], 1.0)
            return tuple(np.round(m.ravel(), 9))

        keys = {key(op) for op in ops}
        for a in ops:
            for b in ops:
                assert key(a @ b) in keys

    def test_unknown_space_group_raises(self):
        with pytest.raises(Exception):
            SpaceGroup('Not-A-Space-Group')


class TestMaterial:
    def test_cell_volume_matches_the_lattice(self):
        mat = Material({'name': 'Si', 'lattice': {'abc': [5.43] * 3, 'abg': [90] * 3},
                        'composition': [{'ion': 'Si', 'pos': [0.125] * 3}]})
        assert mat.volume == pytest.approx(5.43 ** 3, rel=1e-9)

    def test_cell_mass_includes_the_space_group_multiplicity(self):
        """Fd-3m Si has 8 formula units per cell, so Mcell is 8x the formula mass."""
        mat = Material({'name': 'Si', 'lattice': {'abc': [5.43] * 3, 'abg': [90] * 3},
                        'space_group': 'Fd-3m',
                        'composition': [{'ion': 'Si', 'pos': [0.125] * 3}]})
        si_mass = C.periodic_table()['Si']['mass']
        assert mat.Mcell == pytest.approx(8 * si_mass, rel=1e-6)

    def test_missing_lattice_raises_a_clear_keyerror(self):
        with pytest.raises(KeyError, match='lattice'):
            Material({'name': 'Si', 'composition': [{'ion': 'Si', 'pos': [0.125] * 3}]})

    def test_composition_must_be_a_list_not_a_bare_dict(self):
        """A single dict was accepted and iterated key-by-key (item 6)."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            Material({'name': 'Si', 'lattice': {'abc': [5.43] * 3, 'abg': [90] * 3},
                      'composition': {'ion': 'Si', 'pos': [0.125] * 3}})

    def test_nuclear_structure_factor_is_zero_for_the_bcc_extinction(self):
        """Si in Fd-3m: (110) is systematically absent."""
        mat = Material({'name': 'Si', 'lattice': {'abc': [5.43] * 3, 'abg': [90] * 3},
                        'space_group': 'Fd-3m',
                        'composition': [{'ion': 'Si', 'pos': [0.125] * 3}]})
        assert np.abs(mat.calc_nuc_str_fac([1, 1, 0])) < 1e-8

    def test_structure_factor_is_non_zero_for_an_allowed_reflection(self):
        mat = Material({'name': 'Si', 'lattice': {'abc': [5.43] * 3, 'abg': [90] * 3},
                        'space_group': 'Fd-3m',
                        'composition': [{'ion': 'Si', 'pos': [0.125] * 3}]})
        assert np.abs(mat.calc_nuc_str_fac([2, 2, 0])) > 1.0

    def test_many_atoms_build_quickly(self):
        """A 50-atom cell used to take ~0.27 s from repeated JSON parsing (item 156)."""
        import time
        atoms = [{'ion': 'Si', 'pos': [i * 0.01, 0, 0]} for i in range(50)]
        start = time.perf_counter()
        Material({'name': 'Si', 'lattice': {'abc': [5.43] * 3, 'abg': [90] * 3},
                  'composition': atoms, 'Uiso': 0.01})
        assert time.perf_counter() - start < 0.1


class TestDatabases:
    def test_loaders_return_the_same_object(self):
        """The memoised loaders must not re-parse on every access (item 156)."""
        assert C.periodic_table() is C.periodic_table()
        assert C.symmetry() is C.symmetry()
        assert C.scattering_lengths() is C.scattering_lengths()
        assert C.magnetic_ion_j() is C.magnetic_ion_j()

    def test_clear_cache_forces_a_reread(self):
        first = C.periodic_table()
        C.clear_database_cache()
        assert C.periodic_table() is not first
        # ...and the content is unchanged.
        assert C.periodic_table() == first

    @pytest.mark.parametrize('table', [
        'periodic_table', 'symmetry', 'scattering_lengths', 'magnetic_ion_j',
    ])
    def test_loaded_tables_are_non_empty_dicts(self, table):
        data = getattr(C, table)()
        assert isinstance(data, dict)
        assert data


class TestAtom:
    def test_scattering_length_and_debye_waller_are_finite(self):
        atom = Atom('Si', [0, 0, 0], Uiso=0.5)
        assert np.isfinite(atom.b)
        assert np.isfinite(atom.Uiso)

    def test_occupancy_defaults_to_one(self):
        assert Atom('Si', [0, 0, 0]).occupancy == pytest.approx(1.0)

    def test_unknown_element_raises(self):
        with pytest.raises(Exception):
            Atom('NotAnElement', [0, 0, 0], Uiso=0.5)


class TestMagneticFormFactor:
    def test_coefficients_come_from_the_database(self):
        from inspy.crystal import MagneticFormFactor
        mff = MagneticFormFactor('Mn')
        table = C.magnetic_ion_j()['Mn']
        assert mff.j0 == pytest.approx(table['j0'])
        assert mff.j2 == pytest.approx(table['j2'])
        assert mff.j4 == pytest.approx(table['j4'])

    def test_form_factor_is_finite_and_decays_with_q(self):
        from inspy.crystal import MagneticFormFactor
        mff = MagneticFormFactor('Mn')
        # q is |Q| in A^-1 (a scalar or an array of them), not the 4-component hkle.
        # g must be passed explicitly: the old default of 2.0 zeroed the j2 term.
        # Returns (ff, q, j0, j2, j4).
        q = np.array([0.0, 1.0, 2.0, 4.0])
        ff_low = np.asarray(mff.calc_mag_form_fac(q, g=2.0)[0], dtype=float)
        ff_high = np.asarray(mff.calc_mag_form_fac(q + 10.0, g=2.0)[0], dtype=float)
        assert np.all(np.isfinite(ff_low))
        assert np.all(np.isfinite(ff_high))
        assert np.all(np.abs(ff_high) <= np.abs(ff_low) + 1e-12)
        # F(0) is the total moment: at x = 0 the exponentials are all 1 and the
        # j2/j4 terms vanish, so F(0) = j0[0] + j0[2] + j0[4] + j0[6].
        expected_f0 = mff.j0[0] + mff.j0[2] + mff.j0[4] + mff.j0[6]
        assert ff_low[0] == pytest.approx(expected_f0, rel=1e-9)

    def test_g_zero_is_rejected(self):
        """g = 0 would divide by zero in the (g-1)/(g+1) term."""
        from inspy.crystal import MagneticFormFactor
        mff = MagneticFormFactor('Mn')
        with pytest.raises(ValueError, match='g = 0'):
            mff.calc_mag_form_fac(np.array([1.0]), g=0.0)

    def test_default_q_grid_spans_zero_to_two(self):
        """With q omitted the function returns its own 0..2 A^-1 grid."""
        from inspy.crystal import MagneticFormFactor
        mff = MagneticFormFactor('Mn')
        _, q, _, _, _ = mff.calc_mag_form_fac(g=2.0)
        assert q[0] == pytest.approx(0.0)
        assert q[-1] == pytest.approx(2.0)
        assert np.allclose(np.diff(q), 0.025)

    def test_missing_g_is_rejected_rather_than_silently_dropping_j2(self):
        from inspy.crystal import MagneticFormFactor
        mff = MagneticFormFactor('Mn')
        with pytest.raises(ValueError, match='Land'):
            mff.calc_mag_form_fac(np.array([1.0]))

    def test_unknown_ion_raises(self):
        from inspy.crystal import MagneticFormFactor
        with pytest.raises(ValueError, match='No such ion'):
            MagneticFormFactor('NotAnIon')
