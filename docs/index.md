# inspy-conv

**inspy-conv** is a Python toolkit for inelastic neutron scattering data analysis:
triple-axis spectrometer **resolution calculation**, resolution-convolution
**fitting**, **crystallography** helpers, and Qt **GUIs**.

- Install name: `inspy-conv` — import name: `inspy`
- Version: 0.3.5 — Python ≥ 3.8 — MIT license


## What can it do?

| Area | Highlights |
|------|-----------|
| Resolution | Cooper–Nathans and Popovici formalisms; `R0` prefactor and 4×4 `RMS`/`RM` matrices in Q–E and HKL–E frames |
| Convolution fitting | Fit S(Q,ω) model parameters against measured scans with the resolution convolved in (`FitConv`, `UltraFastFitConv`) |
| Crystallography | Lattice metrics, UB orientation matrix, 230 space groups, nuclear/magnetic structure factors, built-in databases |
| GUIs | Resolution calculator, Q-scan fit, E-scan fit — launched from one console command |

## Where to start

1. [Installation](installation.md) — install and verify.
2. [Quick start](quickstart.md) — first resolution calculation in a few lines.
3. [CLI reference](cli.md) — the `inspy-conv` command.
4. User guide: [Energy](energy.md) · [Crystallography](crystal.md) ·
   [Resolution](resolution.md) · [Fitting](fitting.md) · [GUIs](gui.md)
5. Developer guide: [Architecture](architecture.md) · [Development](development.md)

## Minimal example

```python
import inspy

tas = inspy.TripleAxisSpectr(efixed=14.7)
tas.sample = inspy.Sample(3, 4, 5, 90, 90, 90)
tas.CalcResMatHKL([2, 0, 0, 0])   # stores tas.R0, tas.RMS, tas.RM
print(tas.R0)
```
