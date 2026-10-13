# Architecture

## Layout

```
inspy-conv/
├── pyproject.toml          # setuptools build, deps, console script, package data
├── README.md               # project overview (points here)
├── mkdocs.yml              # this documentation site
├── docs/                   # documentation sources (Markdown)
├── src/inspy/              # the package (import name: inspy)
│   ├── __main__.py         # CLI dispatch (lazy per-mode imports)
│   ├── constants.py        # physical constants + JSON database loaders
│   ├── energy.py           # Energy conversions (from neutronpy conventions)
│   ├── crystal/            # lattice, sample, atoms, symmetry, structure factors
│   ├── instrument/         # TAS/TOF spectrometers, mono/ana, projection tools
│   ├── insfit/             # resolution-convolution fitters
│   ├── gui/                # Qt dialogs + shared tools + ui/*.ui files
│   └── database/*.json     # packaged data files
├── tests/                  # pytest suite + GUI scratch scripts
└── ResLib3.4c/             # upstream MATLAB reference (not installed)
```

## Module dependency graph (simplified)

```
            constants
                │
        ┌───────┴────────┐
      energy           crystal ◄──── database/*.json
        │                │
        └───────┬────────┘
            instrument ──────────► ResLib3.4c (algorithm reference)
                │
             insfit
                │
              gui ────────────────► lmfit, QtPy/PySide6 (fitting dialogs)
                │
           __main__ (CLI)          inspy/__init__ (public API)
```

Rules of thumb visible in the code:

1. **Lower layers never import upper layers.** `crystal` does not know about
   `instrument`; `insfit` does not know about `gui`.
2. **GUI modules are leaves.** Only `__main__` (or the user) imports them.
3. **Databases load lazily** through `constants.py` on first use.

## Lazy import design

Two mechanisms keep `import inspy` cheap and robust:

1. **`inspy/gui/__init__.py` imports nothing eagerly.**
   The three GUI modules (and their `lmfit`/Qt dependencies) are pulled in by
   `__main__`'s `PROGRAMS` dispatch only when the corresponding CLI flag is
   selected. Importing a concrete module directly
   (`from inspy.gui.gui_convfit_qscan_uf import main`) still works — Python
   imports submodules on demand. (Older docs suggested
   `from inspy.gui import main_qscan`; that re-export was removed in 0.3.x —
   use the concrete module path above.)

2. **`inspy.main` uses a PEP 562 module `__getattr__`.**
   The attribute is resolved on first access: `inspy.main()` imports
   `gui.main_gui` at call time. A missing Qt stack therefore surfaces exactly
   when the GUI is requested (with the historical warning), not at
   `import inspy`.

```python
# inspy/__init__.py (simplified)
def __getattr__(name):
    if name == 'main':
        try:
            from .gui.main_gui import main
        except ImportError:
            warnings.warn('QtPy not found, cannot run Resolution GUI.')
            raise
        return main
    raise AttributeError(name)
```

## The resolution engine

`instrument/tas_spectr.py` implements the ResLib 3.4c port:

1. `_StandardSystem()` — build the sample/reciprocal frames from `Sample`.
2. `CalcResMatQ(Q, W)` — Cooper–Nathans or Popovici in the Q frame → `R0`, `RM`.
3. `CalcResMatHKL(hkle)` — rotate into the (H,K,L,E) frame → `self.R0/RMS/RM`
   (stored on the instance; returns `None`).
4. `ResConv(...)` — convolve an S(Q,ω) model on a quadrature grid.
5. Plot methods render the ellipses/ellipsoids from the same matrices.

`insfit` calls `ResConv` repeatedly through `least_squares`
(`method='trf'`), with `UltraFastFitConv` adding caching and precomputation.

## Packaging

- Build backend: `setuptools>=77` (`pyproject.toml`).
- Package discovery: `src/` layout (`[tool.setuptools.packages.find]`).
- Package data: `database/*.json`, `gui/ui/*.ui` (explicitly declared).
- Console script: `inspy-conv = "inspy.__main__:main"`.
- Optional dev extra: `.[dev]` → pytest, flake8.

## File encodings

- Most files: UTF-8, **LF** line endings, no BOM.
- `tas_spectr.py` in particular is LF-only with no trailing newline — keep
  edits raw-text to avoid introducing CRLF (the MATLAB port was converted
  wholesale).
