# Installation

## Requirements

- Python **≥ 3.8**
- Core dependencies (installed automatically):
  `numpy`, `scipy`, `matplotlib`, `pandas`, `lmfit`, `h5py`, `plotly`
- GUI dependencies: `QtPy` + `PySide6` (also in the default dependency list)

## From PyPI

```bash
pip install inspy-conv
```

## Editable (development) install

```bash
git clone https://github.com/gcdengansto/inspy-conv
cd inspy-conv
pip install -e .
```

For running the test suite:

```bash
pip install -e ".[dev]"   # adds pytest, flake8
```

## Verify the install

```python
import inspy
print(inspy.__version__)          # e.g. "0.3.3"

en = inspy.Energy(energy=14.7)
print(en.wavelength)              # ~2.37 Å
```

The console command should also be on `PATH`:

```bash
inspy-conv -h
```

## Lazy imports and optional GUIs

The package uses **lazy loading**:

- `import inspy` loads only the core scientific stack
  (constants, crystal, instrument, insfit) — **no Qt, no GUI modules**.
- `inspy.main` is resolved on first access (PEP 562 `__getattr__`), which then
  imports the resolution-calculation GUI.
- Each CLI mode imports its own GUI module only when its flag is selected, so a
  missing fit-dialog dependency (Qt backend, `lmfit`) breaks only that mode —
  not `import inspy` and not the other modes.

Practical consequence: on a headless/batch analysis machine you can install and
use the calculation core without a working Qt stack, as long as you do not launch
the GUIs.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `inspy.main` raises `ImportError`, warning `QtPy not found...` | Qt bindings missing | `pip install PySide6` (or another QtPy-supported binding) |
| `-qscan` / `-escan` fail with `lmfit` import error | `lmfit` missing | `pip install lmfit` |
| `pip install` succeeds but `import inspy` fails | Broken environment / path | Reinstall with `pip install --force-reinstall inspy-conv` |
| Package data (`database/*.json`, `gui/ui/*.ui`) missing | Old install tooling | Reinstall from PyPI or use an editable install; data files are declared in `pyproject.toml` |

## Upgrading

```bash
pip install --upgrade inspy-conv
```
