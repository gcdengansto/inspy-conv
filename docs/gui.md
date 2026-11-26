# GUI

The unified Qt (PySide6 via QtPy) fitting interface ships with the package. It
is imported **lazily** — see
[Installation](installation.md#lazy-imports-and-optional-guis).

## Launching

```bash
inspy-conv            # unified fit GUI (default)
inspy-conv -fit       # unified energy- and Q-scan fit
```

or from Python:

```python
import inspy

from inspy.gui.gui_convfit_uf import main as fit_main
fit_main()                      # unified E/Q-scan fit
```

`inspy.main()` launches the same unified fit GUI. The `MainWindow` class can be
embedded in another Qt application.

## Unified scan fit — `gui_convfit_uf.py`

The unified fit GUI detects the scan type after data import:

- If W varies and H/K/L are constant, it uses the energy-transfer model.
- If one or more HKL coordinates vary and W is constant, it uses the Q-scan
  model.
- Changes of at most 0.005 r.l.u. in H/K/L and 0.05 meV in W are treated as
  motor tolerance. If multiple coordinates exceed their tolerance, the GUI
  warns and asks which coordinate is the intended scan axis.
- If no coordinate varies beyond its tolerance, it warns and does not load the
  data.

The Q model (`SqwQScanTwoPeaks`) fits two Gaussians in the selected Q
coordinate (FWHM); the energy model (`SqwDemo`) fits two Lorentzians in W
(HWHM). Both use `UltraFastFitConv`.

## Printed output

After a fit the unified GUI prints a parameter table. Conventions:

| Label | Meaning |
|-------|---------|
| `QH1/QH2` (or `EN1/EN2` for an energy scan) | peak centres, error |
| `Int1/Int2` | **peak integrals** — Int1 belongs to peak 1 (under the first centre label), Int2 to peak 2 |
| `FWHM1/FWHM2` (Q-scan) | Gaussian FWHM in Q — equals the model width `p[3]`/`p[4]` |
| `HWHM1/HWHM2` (E-scan) | Lorentzian HWHM — equals `p[3]`/`p[4]` |
| `ratio` | intensity-ratio parameter `p[2]` |
| `bg`, `temp` | background `p[6]`, temperature `p[7]` |

!!! note "Fixed in the 2026-10 review"
    - Q-scan widths are Gaussian **FWHM** and are printed as `FWHM1/FWHM2`.
    - The Q-model ratio multiplies peak **2**; `Int1/Int2` are printed in
      peak order with first-order error propagation on `Int2 = ratio·p5`.

## Qt / headless notes

- The GUI needs a working Qt binding (`PySide6` by default through `QtPy`).
- On headless machines the calculation core (`TripleAxisSpectr`, fitting
  classes) still works — only the GUI refuses to start.
