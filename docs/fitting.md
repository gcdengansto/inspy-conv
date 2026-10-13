# Convolution fitting

Measured scan = S(Q,ω) convolved with the instrument resolution. Instead of
fitting analytic peak shapes, **inspy-conv** fits the *model* by numerically
convolving it at every iteration.

## The pieces

| Piece | Role |
|-------|------|
| `TripleAxisSpectr` | Resolution engine (see [Resolution](resolution.md)) |
| `sqw(H, K, L, W, p)` | S(Q,ω) model — a callable returning intensity on the scan grid |
| `pref(H, K, L, W, EXP, p)` | Prefactor model — polarization, form factor, overall scale |
| `FitConv` / `UltraFastFitConv` | Optimiser driving the loop |

Demo models ship in `inspy.gui.tools`:

- `SqwDemo` — two-peak E-scan model (Lorentzians in W, Bose factor, ratio `p[2]`
  multiplies **peak 1**) — used by the **E-scan GUI**.
- `SqwQScanTwoPeaks` — two Gaussians in Q, width `p[3]`/`p[4]` are the
  **FWHM in Q**, ratio `p[2]` multiplies **peak 2** — used by the **Q-scan GUI**.
- `PrefDemoFF` — polarization ⊗ magnetic form factor ⊗ overall scale `p[5]`.

!!! warning "Note the ratio convention"
  The two demo models differ: `SqwDemo` applies the intensity ratio to
  peak 1, `SqwQScanTwoPeaks` applies it to peak 2. See
  [GUIs → printed output](gui.md#printed-output).

## FitConv

```python
from inspy import FitConv

fitter = FitConv(exp=tas, sqw=sqw_func, prefactor=pref_func,
                 hkle=hkle, Iobs=Iobs, dIobs=dIobs,
                 param=param, paramfixed=param_fixed_mask)
fitter.fitwithconv(...)
```

- Optimiser: `scipy.optimize.least_squares` with **`method='trf'`**
  (Trust-Region Reflective; `lm` would be Levenberg–Marquardt).
- `param` is a flat numpy array of model parameters; `paramfixed` a 0/1 mask.
- Bounds: see the note in the source — `bounds` are **not currently forwarded**
  to `least_squares` (known limitation, documented at the call site).

## UltraFastFitConv

```python
from inspy import UltraFastFitConv

fitter = UltraFastFitConv(tas, sqw_func, pref_func, hkle, Iobs, dIobs,
                          cache_size=128, precompute_data=True)
result = fitter.fit_ultrafast(param_initial, param_fixed_mask)
```

Optimisations over `FitConv`:

- **Pre-computed data** — hkle grids and static factors are prepared once.
- **LRU caching** of repeated resolution evaluations
  (`cache_size`; the value is currently reserved — see source note).
- **Adaptive Jacobian steps** — central differences with parameter-aware step
  sizes (two `ResConv` calls per free parameter).

Both classes follow the ResLib `FitConv`/`FitConvSMA` structure.

## Writing your own model

```python
def my_sqw(H, K, L, W, p):
    """Gaussian in H centred at p[0], width p[1], scale p[2]."""
    import numpy as np
    return p[2] * np.exp(-(H - p[0])**2 / (2 * p[1]**2))

def my_pref(H, K, L, W, EXP, p):
    """Flat prefactor."""
    return np.full_like(np.asarray(W, dtype=float), 1.0)
```

Conventions the fitter expects:

- The model must broadcast over the scan grid (same shape as `hkle` slices).
- Temperature/energy conversions inside Bose factors should use meV
  (`W / (k_B T)` with `k_B = 0.08617` meV/K, as in the demo models).

## Fitting from the GUI

The Q-scan and E-scan GUIs wrap the fitters with data loading, plotting, and
parameter tables — see [GUIs](gui.md).
