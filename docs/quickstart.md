# Quick start

This walkthrough computes a resolution matrix and draws the resolution ellipsoid
for a simple instrument.

## 1. Energy units

```python
import inspy

en = inspy.Energy(energy=14.7)     # meV
print(en.wavelength)               # Å
print(en.wavevector)               # 1/Å
print(en.velocity)                 # m/s
```

`Energy` accepts any one of `energy`, `wavelength`, `wavevector`, `velocity`,
`temperature`, or `frequency` and converts to the rest — see [Energy](energy.md).

## 2. Define the sample

```python
sample = inspy.Sample(
    a=3.0, b=4.0, c=5.0,          # lattice constants (Å)
    alpha=90, beta=90, gamma=90,  # angles — in DEGREES
    u=[1, 0, 0], v=[0, 1, 0],     # orientation vectors
    mosaic=30, vmosaic=30,        # mosaic spread (arcmin)
)
print(sample.UBmatrix)             # UB orientation matrix
```

!!! note
    All lattice angles throughout the package are in **degrees**. The reciprocal
    quantities (`abg_rad`, etc.) are derived internally where radians are needed.

## 3. Configure the spectrometer

```python
from inspy.instrument import TripleAxisSpectr, Mono, Ana

mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30,
            height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1)
ana  = Ana(tau='PG(002)', mosaic=30, vmosaic=30,
           height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1,
           thickness=0.2, horifoc=-1)

tas = TripleAxisSpectr(
    efixed=14.7,          # fixed energy (meV) — fixed-Ef if infin=-1 (default)
    method=1,             # 0 = Cooper-Nathans, 1 = Popovici (default)
    mono=mono, ana=ana, sample=sample,
)
tas.hcol = [40, 40, 40, 120]      # horizontal collimation (arcmin): L0,L1,L2,L3
tas.arms = [200, 100, 100, 200]   # distances (cm)
```

`tau` accepts a crystal name from the built-in table
(`'PG(002)'`, `'Si(111)'`, `'Ge(111)'`, `'Cu(111)'`, …) or a numeric value in Å⁻¹.

## 4. Calculate the resolution

```python
tas.CalcResMatHKL([2, 0, 0, 0])   # (H, K, L, E)

tas.R0        # prefactor            -> shape (N,)
tas.RMS       # 4x4 matrix (HKL-E)   -> shape (N, 4, 4)
tas.RM        # 4x4 matrix (Q frame) -> shape (N, 4, 4)
```

!!! important
    `CalcResMatHKL` **returns `None`** and stores the results on the instrument
    (`self.R0`, `self.RMS`, `self.RM`). Earlier README revisions showed a
    tuple unpack — that was wrong.

Batch calculation works by passing a (4, N) array:

```python
import numpy as np
hkle = np.array([[2, 0, 0, 0],
                 [2, 0, 0, 2],
                 [3, 0, 0, 0]])
tas.CalcResMatHKL(hkle.T)     # N = 3 points
print(tas.RMS.shape)          # (3, 4, 4)
```

## 5. Plot

```python
tas.ResolutionPlot(hkle=[2, 0, 0, 0])                       # 4-panel plot
tas.ResolutionPlotProj(qslice='QxQy', hkle=[2, 0, 0, 0])    # single panel
tas.ResolutionPlot3D(hkle=[2, 0, 0, 0])                     # 3D ellipsoid
```

## 6. Angles both ways

```python
tas.get_angles_and_Q([2, 0, 0, 0])          # HKL-E -> spectrometer angles
tas.get_hkl_and_Q(M2, S1, S2, A2)           # motor angles -> HKL-E
```

## Next steps

- Full configuration and theory: [Resolution](resolution.md)
- Fit a model to data: [Convolution fitting](fitting.md)
- Interactive use: [GUIs](gui.md)
