# Energy

`inspy.Energy` converts between the common units of a neutron's state.
Adapted from *neutronpy* conventions.

## Constructor

```python
Energy(energy=None, wavelength=None, velocity=None,
       wavevector=None, temperature=None, frequency=None)
```

Pass **exactly one** quantity; the others are derived from it.

| Keyword | Unit | Meaning |
|---------|------|---------|
| `energy` | meV | Neutron energy E |
| `wavelength` | Å | de Broglie wavelength λ |
| `wavevector` | 1/Å | k = 2π/λ |
| `velocity` | m/s | Group velocity v |
| `temperature` | K | Equivalent temperature (E = k_B·T) |
| `frequency` | THz | E = ħ·ω |

## Properties

After construction, all of `energy`, `wavelength`, `wavevector`, `velocity`,
`temperature`, `frequency` are available as attributes (read-only conversions).

## Examples

```python
from inspy import Energy

e = Energy(energy=14.7)
print(e.wavelength)     # ~2.3739 Å
print(e.wavevector)     # ~2.6473 1/Å

w = Energy(wavelength=2.5)
print(w.energy)         # ~13.06 meV

# equality compares energies (within 1e-6 meV)
Energy(energy=14.7) == Energy(energy=14.7)   # True
```

## Typical use in resolution work

The fixed energy of the spectrometer is specified in meV:

```python
tas = inspy.TripleAxisSpectr(efixed=14.7)
```

and the Bose factor used by demo S(Q,ω) models converts energies to temperature
units internally (`W / (k_B·T)`).
