# Resolution calculation

The core of the package is **`inspy.TripleAxisSpectr`**
(`src/inspy/instrument/tas_spectr.py`, ~2 900 lines) — a faithful Python port of
MATLAB **ResLib 3.4c** by A. Zheludev.

## Theory in one page

A triple-axis spectrometer measures a **convolution** of the sample cross section
with the instrument resolution function. Near a point in (Q, E) the resolution
is approximated as a 4-dimensional Gaussian characterised by:

- **R₀** — a dimensionless prefactor (normalisation of the Gaussian), and
- **RMS** — the 4×4 inverse covariance matrix ("resolution matrix") in the
  measurement frame.

Two formalisms are implemented, selectable per instrument via `method`:

| `method` | Formalism | Notes |
|----------|-----------|-------|
| `0` | **Cooper–Nathans** | Analytic Gaussian approximation; older, simpler |
| `1` | **Popovici** (default) | Improves Cooper–Nathans, closer to Monte-Carlo |

Both follow the classic chain: project source/beam-monitor/monochromator/sample/
analyzer/detector phasespaces through the spectrometer arms (`arms`, `hcol`,
`vcol`), rotate into the sample frame, and (optionally) apply resolution
smoothing (`Smooth`) or sample-shape correction.

Key intermediate quantities:

| Symbol in code | Meaning |
|----------------|---------|
| `R0` | Prefactor, shape `(N,)` |
| `RMS` | 4×4 matrix in the (H,K,L,E) frame, shape `(N,4,4)` |
| `RM` | 4×4 matrix in the Q frame (before HKL rotation), shape `(N,4,4)` |
| `self.method` | 0 = Cooper–Nathans, 1 = Popovici |
| `self.infin` | −1 fixed E_f, +1 fixed E_i (default −1) |

## Instrument configuration

```python
from inspy.instrument import TripleAxisSpectr, Mono, Ana

tas = TripleAxisSpectr(
    efixed=14.7,          # fixed energy (meV)
    method=1,             # Popovici (default)
    mono=mono, ana=ana, sample=sample,
    hcol=[40, 40, 40, 120],       # arcmin: before mono, before sample,
                                  #         before ana, after ana
    vcol=[120, 120, 120, 120],    # vertical collimation (or default)
)
tas.arms = [200, 100, 100, 200]   # cm: L0 L1 L2 L3 (L1mon extra, optional)
```

Sub-components can also be passed piecewise as keyword arguments
(`mono_tau='PG(002)'`, `sample_a=3.0`, …); the constructor builds the
components for you.

### Monochromator / Analyzer

```python
mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30,
            height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1)
ana  = Ana(tau='PG(002)', mosaic=30, vmosaic=30, ..., thickness=0.2, horifoc=-1)
```

- `tau` — crystal name from the built-in table (`get_tau`), or a numeric value.
  Table entries include PG(002), Si(111), Ge(111/220/311/511/533),
  Cu(111/002/220), …
- `direct` — −1/+1 focusing direction; `rh`, `rv` — horizontal/vertical
  focusing radii (cm).

## Main methods

| Method | Description |
|--------|-------------|
| `CalcResMatHKL(hkle)` | Resolution at (H,K,L,E); **stores `self.R0`, `self.RMS`, `self.RM`, returns `None`**. Accepts `(4,)` or `(4,N)`. |
| `CalcResMatQ(Q, W)` | Same, in Q–E frame (internal entry point used by the HKL version) |
| `ResConv(sqw, pref, nargout, hkle, ...)` | Full 4-D convolution of an S(Q,ω) model with the resolution |
| `get_angles_and_Q(hkle)` | HKL-E → spectrometer angles (A3, mono/ana 2θ, …) |
| `get_hkl_and_Q(M2, S1, S2, A2)` | Motor angles → HKL-E |
| `ResolutionPlot(hkle=...)` | 4-panel figure: QxQy, QxE, QyE + Bragg widths |
| `ResolutionPlotProj(ax, qslice, ...)` | Single panel; `qslice` in `{'QxQy','QxE','QyE','QxQz',...}` |
| `ResolutionPlot3D(hkle, ...)` | 3-D resolution ellipsoid, optionally with dispersion |

```python
tas.CalcResMatHKL([2, 0, 0, 0])
print(tas.R0, tas.RMS.shape)

tas.ResolutionPlot(hkle=[2, 0, 0, 0])
```

## Utilities — `inspy.instrument.tools`

| Function | Purpose |
|----------|---------|
| `get_tau(name)` | Look up a monochromator/analyzer crystal by name and return its τ value (table entries are d in Å or precomputed 2π/d — the returned magnitude is 2π/d) |
| `get_bragg_widths(RM)` | Bragg peak FWHM in (Q1, Q2, E) from the resolution matrix |
| `get_phonon_width(r0, M, C)` | Phonon FWHM projected along a dispersion direction |
| `calc_proj_hwhm(MP)` | 2-D projection HWHM + rotation angle |
| `project_into_plane(...)` | Integrate an out-of-plane Gaussian direction |
| `_voigt(x, a)` | Voigt profile (Faddeeva approximation) |
| `chop(...)` | Scalar-path helper for scalar inputs |

## Time-of-flight (preview)

`inspy.TimeOfFlightSpectr` (`tof_spectr.py`) with `Chopper`, `Detector`, and
`Guide` components exists but is **stub-level** (`CalcResMat*` not yet
implemented). The `Energy`/`Sample` infrastructure is shared.

## Relationship to ResLib 3.4c

The MATLAB sources are bundled in `ResLib3.4c/` (including `manual.pdf`) for
reference. Port correspondence (short):

| ResLib | inspy |
|--------|-------|
| `ResMat.m` / `ResMatS.m` | `TripleAxisSpectr.CalcResMatQ` |
| `ConvRes.m` | `TripleAxisSpectr.ResConv` |
| `ResPlot.m` | `ResolutionPlot`, `ResolutionPlotProj` |
| `ResPlot3D.m` | `ResolutionPlot3D` |
| `GetTau.m` | `instrument.tools.get_tau` |
| `FitConv.m` / `FitConvSMA.m` | `insfit.FitConv`, `insfit.UltraFastFitConv` |
