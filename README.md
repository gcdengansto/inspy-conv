# inspy-conv

**inspy-conv** (install name: `inspy-conv`, import name: `inspy`) is a Python toolkit
for **inelastic neutron scattering data analysis**. It provides triple-axis spectrometer
resolution calculation (Cooper–Nathans / Popovici), resolution-convolution fitting of
S(Q,ω) models, crystal lattice and space-group tools, and Qt-based interactive GUIs.

The resolution algorithms are the same as **ResLib**, with known issues tracked and fixed incrementally.

---

## Features

- **Resolution calculation** — Cooper–Nathans (`method=0`) and Popovici (`method=1`,
  default) formalisms in Q–E and HKL–E frames: `R0` prefactor, `RMS`/`RM` matrices.
- **Resolution convolution fitting** — fit S(Q,ω) parameters to measured scans by
  numerically convolving the model with the instrument resolution
  (`FitConv`, `UltraFastFitConv`).
- **Crystallography** — lattices, UB orientation matrices, the 230 space groups,
  nuclear/magnetic structure factors, built-in scattering and form-factor databases.
- **Interactive GUIs** — resolution calculator, Q-scan fit, E-scan fit.
- **CLI** — one console command launches any of the three GUIs.

---

## Installation

```bash
pip install inspy-conv
```

Editable development install:

```bash
git clone https://github.com/gcdengansto/inspy-conv
cd inspy-conv
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `lmfit`, `h5py`, `plotly`,
`QtPy`, `PySide6` (Python ≥ 3.8).

Because GUI modules are imported lazily, `import inspy` itself only needs the core
scientific stack — Qt/lmfit are required when a fitting GUI is actually launched.
See [docs/installation.md](docs/installation.md).

---

## Command-line usage

Installing the package provides the `inspy-conv` console command:

```bash
inspy-conv            # resolution calculator GUI (default)
inspy-conv -rescalc   # same as default
inspy-conv -escan     # energy-scan convolution-fit GUI
inspy-conv -qscan     # Q-scan convolution-fit GUI
inspy-conv -h         # help
```

The three flags are mutually exclusive; without a flag the resolution calculator
starts. `python -m inspy [flag]` is equivalent.
Full reference: [docs/cli.md](docs/cli.md).

---

## Quick start

```python
import inspy

# Neutron beam: 14.7 meV
en = inspy.Energy(energy=14.7)
print(en.wavelength)      # Angstrom
print(en.wavevector)      # 1/Angstrom

# Cubic sample, 30' mosaic
sample = inspy.Sample(a=3.0, b=4.0, c=5.0,
                      alpha=90, beta=90, gamma=90,
                      u=[1, 0, 0], v=[0, 1, 0],
                      mosaic=30, vmosaic=30)

# Triple-axis spectrometer, fixed Ef = 14.7 meV (Popovici by default)
tas = inspy.TripleAxisSpectr(efixed=14.7, method=1)
tas.sample = sample
tas.hcol = [40, 40, 40, 120]     # collimation in arcmin
tas.arms = [200, 100, 100, 200]  # distances in cm

# Resolution at (H, K, L, E) = (2, 0, 0, 0)
tas.CalcResMatHKL([2, 0, 0, 0])
print(tas.R0)    # prefactor, shape (N,)
print(tas.RMS)   # 4x4 resolution matrix, shape (N, 4, 4)
print(tas.RM)    # resolution matrix in the Q frame
```

> `CalcResMatHKL` stores its results on the instrument object (`R0`, `RMS`, `RM`)
> and returns `None`.

More in [docs/quickstart.md](docs/quickstart.md).

---

## Package structure

```
src/inspy/
├── __main__.py            # CLI dispatch (-rescalc / -escan / -qscan)
├── constants.py           # Physical constants + database loaders
├── energy.py              # Energy / wavelength / wavevector conversion
├── crystal/               # Crystallography
│   ├── lattice.py         #   Lattice: metric tensors, d-spacing, volume
│   ├── sample.py          #   Sample: UB orientation matrix, mosaic
│   ├── atom.py            #   Atom / MagneticAtom
│   ├── symmetry.py        #   SpaceGroup: 230 groups, symmetry operations
│   ├── structure_factors.py  # Nuclear / magnetic structure factors
│   └── material.py        #   Material = Sample + structure factors
├── instrument/            # Instrument models
│   ├── mono.py            #   Monochromator (+ built-in crystal tau table)
│   ├── ana.py             #   Analyzer
│   ├── components.py      #   Chopper, Detector, Guide (Time-of-Flight)
│   ├── tas_spectr.py      #   TripleAxisSpectr — core resolution engine
│   ├── tof_spectr.py      #   TimeOfFlightSpectr (stub)
│   └── tools.py           #   Projection utilities, get_tau, Bragg/phonon widths
├── insfit/                # Resolution-convolution fitting
│   ├── fitconv.py         #   FitConv (scipy.optimize.least_squares, method='trf')
│   └── uffitconv.py       #   UltraFastFitConv (caching, precompute, adaptive steps)
├── gui/                   # Qt graphical interfaces (imported lazily)
│   ├── main_gui.py        #   Resolution calculator GUI
│   ├── gui_convfit_qscan_uf.py  # Q-scan convolution-fit GUI
│   ├── gui_convfit_escan_uf.py  # E-scan convolution-fit GUI
│   ├── tools.py           #   Shared widgets + demo S(Q,w)/prefactor models
│   └── ui/                #   Qt Designer .ui files
└── database/              # JSON data files (packaged via pyproject.toml)
    ├── magnetic_form_factors.json
    ├── periodic_table.json
    ├── scattering_lengths.json
    └── symmetry.json
```

---

## Documentation

Detailed documentation lives in [`docs/`](docs/index.md) and is written to be
publishable as a static site with [MkDocs](https://www.mkdocs.org/)
(config provided in `mkdocs.yml`):

```bash
pip install mkdocs
mkdocs serve     # http://127.0.0.1:8000
mkdocs build     # static site in site/
```

| Document | Contents |
|----------|----------|
| [Installation](docs/installation.md) | Install, dependencies, verification, troubleshooting |
| [Quick start](docs/quickstart.md) | End-to-end first calculation |
| [CLI reference](docs/cli.md) | `inspy-conv` flags and behaviour |
| [Energy](docs/energy.md) | Neutron energy conversions |
| [Crystallography](docs/crystal.md) | Lattice, Sample, Atom, Material, SpaceGroup, structure factors |
| [Resolution](docs/resolution.md) | TripleAxisSpectr, theory notes, plotting, utilities |
| [Convolution fitting](docs/fitting.md) | FitConv, UltraFastFitConv, S(Q,ω) models |
| [GUIs](docs/gui.md) | The three graphical interfaces |
| [Architecture](docs/architecture.md) | Package layout, lazy imports, packaging |
| [Development](docs/development.md) | Setup, tests, conventions, review history |

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## References

- A. Zheludev, *ResLib 3.4c* — MATLAB resolution library the TAS engine.
- neutronpy conventions — several modules are adapted from *neutronpy*.

## Author

Guochu Deng — gc.deng.ansto@gmail.com

## License

MIT — see [LICENSE](LICENSE).
