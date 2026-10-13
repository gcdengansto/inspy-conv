# GUIs

Three Qt (PySide6 via QtPy) interfaces ship with the package. They are imported
**lazily** — see [Installation](installation.md#lazy-imports-and-optionally-guis).

## Launching

```bash
inspy-conv            # resolution calculator (default)
inspy-conv -escan     # energy-scan fit
inspy-conv -qscan     # Q-scan fit
```

or from Python:

```python
import inspy
inspy.main()                    # resolution calculator

from inspy.gui import gui_convfit_escan_uf, gui_convfit_qscan_uf
gui_convfit_escan_uf.main()     # E-scan fit
gui_convfit_qscan_uf.main()     # Q-scan fit
```

Each module also has a `MainWindow` class if you want to embed it.

## 1. Resolution calculator — `main_gui.py`

The default mode. Three linked resolution panels (**QxQy, QxE, QyE**) with the
full instrument parameter set (collimation, arms, mono/ana, sample, energy)
editable in side docks. Recalculates `CalcResMatHKL` + `ResolutionPlotProj`
interactively.

UI definition: `src/inspy/gui/ui/resolution.ui` (Qt Designer).

## 2. Q-scan fit — `gui_convfit_qscan_uf.py`

Fit a constant-Q scan (intensity vs QH/QK/QL) with the resolution convolved in.

- Model: `SqwQScanTwoPeaks` — two Gaussians in Q
  (widths `p[3]`, `p[4]` are **FWHM in Q**), ratio `p[2]` scales **peak 2**,
  `p[5]` overall scale (via `PrefDemoFF`).
- Data: load a text/table file, select the H K L W Iobs dIobs columns.
- Uses `UltraFastFitConv` with `lmfit` for parameter handling/reporting.

## 3. E-scan fit — `gui_convfit_escan_uf.py`

Fit a constant-E scan (intensity vs energy transfer).

- Model: `SqwDemo` — two Lorentzians in W with Bose factor
  (Γ₁ = `p[3]`, Γ₂ = `p[4]` are **HWHM**), ratio `p[2]` scales **peak 1**.
- Same data-loading and fitting flow as the Q-scan GUI.

## Printed output

After a fit both fitting dialogs print a parameter table. Conventions:

| Label | Meaning |
|-------|---------|
| `QH1/QH2` (or `EN1/EN2` for E-scan) | peak centres, error |
| `Int1/Int2` | **peak integrals** — Int1 belongs to peak 1 (under the first centre label), Int2 to peak 2 |
| `FWHM1/FWHM2` (Q-scan) | Gaussian FWHM in Q — equals the model width `p[3]`/`p[4]` |
| `HWHM1/HWHM2` (E-scan) | Lorentzian HWHM — equals `p[3]`/`p[4]` |
| `ratio` | intensity-ratio parameter `p[2]` |
| `bg`, `temp` | background `p[6]`, temperature `p[7]` |

!!! note "Fixed in the 2026-10 review"
    - The Q-scan dialog's widths were briefly mislabelled `HWHM` during the
      port; they are Gaussian **FWHM** and are printed as `FWHM1/FWHM2`.
    - Q-scan `Int1/Int2` were swapped relative to the model (the ratio
      multiplies peak **2** there); they are now printed in peak order with
      first-order error propagation on `Int2 = ratio·p5`.

## Qt / headless notes

- The GUIs need a working Qt binding (`PySide6` by default through `QtPy`).
- On headless machines the calculation core (`TripleAxisSpectr`, fitting
  classes) still works — only the GUIs refuse to start.
