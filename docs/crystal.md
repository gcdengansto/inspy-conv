# Crystallography

All classes below are importable from `inspy` directly or from `inspy.crystal`.

```python
from inspy.crystal import (Lattice, Sample, Atom, MagneticAtom,
                           Material, SpaceGroup,
                           NuclearStructureFactor, MagneticFormFactor,
                           MagneticStructureFactor)
```

!!! note "Units"
    Lattice constants are in **Å**, angles (`alpha`, `beta`, `gamma`) in
    **degrees**. Radian variants are derived internally where needed.

## Lattice

Defined by six parameters; provides metric tensors, reciprocal lattice,
d-spacing, and volume.

```python
lat = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
lat.volume                    # 125.0 Å³
lat.get_d_spacing([1, 0, 0])  # 5.0 Å
lat.get_q([1, 0, 0])          # |Q| = 2π/d for that reflection
lat.G, lat.Gstar              # metric tensors (direct / reciprocal)
lat.Bmatrix                   # Cartesian basis matrix (B.T @ B = Gstar)
lat.get_two_theta([1, 1, 1], wavelength=1.5)   # Bragg angle (deg)
```

Useful properties: `abc`, `abg` (degrees), `abg_rad`, `astar/bstar/cstar`,
`alphastar/betastar/gammastar`, `lattice_type` (auto-classified:
triclinic/monoclinic/orthorhombic/…).

## Sample

`Sample` extends `Lattice` with the physical/orientational properties needed by
a spectrometer calculation: orientation vectors **u**, **v**, mosaic spread,
and sample shape.

```python
s = Sample(3, 4, 5, 90, 90, 90,
           u=[1, 0, 0], v=[0, 1, 0],
           mosaic=30, vmosaic=30,
           width=20, height=20, depth=0.2)

s.Umatrix        # orientation matrix (u, v)
s.UBmatrix       # full UB matrix — hkl -> Q in the lab frame
s.get_phi(Q)     # rotation phi aligning Q into the scattering plane
```

`Sample` is what `TripleAxisSpectr.sample` expects.

## Atom / MagneticAtom

```python
atom = Atom('Mn', pos=[0, 0, 0], occupancy=1.0, Uiso=0.005)
atom.b            # coherent scattering length (from scattering_lengths.json)
atom.mass         # atomic mass (from periodic_table.json)
```

- `Atom` looks up coherent/incoherent scattering lengths, cross sections,
  mass, and number density from the bundled databases.
- `MagneticAtom` adds magnetic moment / form-factor lookup.

## Material

`Material` = `Sample` + structure factors, built from a dictionary-based
description of a crystal:

```python
crystal = {
    'name': 'MnO',
    'lattice': {'abc': [4.445, 4.445, 4.445], 'abg': [90, 90, 90]},
    'space_group': 'Fm-3m',
    'massNorm': False,
    'formulaUnits': 1,            # optional, default 1
    'composition': [
        {'ion': 'Mn', 'pos': [0, 0, 0], 'occupancy': 1.0},
        {'ion': 'O',  'pos': [0.5, 0.5, 0.5], 'occupancy': 1.0},
    ],
}
mat = Material(crystal)
mat.calc_nuc_str_fac([1, 1, 1])   # nuclear structure factor
```

Notes on the dictionary keys:

- `'lattice'` is a **nested dict** with `'abc'` (Å) and `'abg'` (degrees) lists.
- `'massNorm'` is read whenever `'space_group'` is present — include it.
- Per-composition `'occupancy'`, `'Uiso'`, `'Uaniso'` default to 1 / 0 / zeros.
- With a space group, each listed site is expanded through
  `symmetrize_position` to the full set of equivalent positions.

It inherits from `Sample`, `NuclearStructureFactor`, and
`MagneticStructureFactor`.

## SpaceGroup

Any of the 230 groups, addressed by international symbol, Hermann–Mauguin
symbol, or number:

```python
sg = SpaceGroup('Fm-3m')       # or SpaceGroup(225), SpaceGroup('P 21 21 21')
sg.group_number
sg.symmetry_operations         # list of operations
sg.symmetrize_position([0, 0, 0])   # symmetry-equivalent positions
```

Backed by `database/symmetry.json`.

## Structure factors

| Class / method | Purpose |
|----------------|---------|
| `NuclearStructureFactor.calc_nuc_str_fac(hkl)` | Nuclear structure factor F(hkl) with Debye–Waller factors |
| `MagneticFormFactor(ion)` | Magnetic form-factor coefficients (j0, j2, j4) for an ion, from `magnetic_form_factors.json` |
| `MagneticStructureFactor` | Magnetic structure factor (partially implemented — magnetic interaction vector components exist; full polarization analysis is stubbed) |

## Databases

Four JSON files ship with the package and are loaded on demand:

| File | Loader | Contents |
|------|--------|----------|
| `periodic_table.json` | `periodic_table()` | mass, Z, density |
| `scattering_lengths.json` | `scattering_lengths()` | coherent/incoherent b, cross sections |
| `magnetic_form_factors.json` | `magnetic_ion_j()` | j0, j2, j4 coefficients |
| `symmetry.json` | `symmetry()` | the 230 space groups |

Loader functions live in `inspy.constants`.
