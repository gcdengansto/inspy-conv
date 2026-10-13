# CLI reference

Installing the package adds one console command, **`inspy-conv`**
(defined in `pyproject.toml` as `inspy.__main__:main`).

## Synopsis

```bash
inspy-conv [-h] [-rescalc | -escan | -qscan]
```

## Options

| Flag | Effect |
|------|--------|
| *(none)* | Launch the **resolution calculation GUI** (default) |
| `-rescalc` | Same as default — resolution calculation GUI |
| `-escan` | Launch the **energy-scan convolution-fit GUI** (`gui_convfit_escan_uf`) |
| `-qscan` | Launch the **Q-scan convolution-fit GUI** (`gui_convfit_qscan_uf`) |
| `-h`, `--help` | Show usage and exit |

The three mode flags are **mutually exclusive** (argparse mutually-exclusive
group); combining them is an error.

## Examples

```bash
inspy-conv              # resolution calculator
inspy-conv -escan       # E-scan fitting
inspy-conv -qscan       # Q-scan fitting
python -m inspy -qscan  # equivalent module form
```

## Behaviour

- On start the command prints `Running inspy in the <mode> mode.` and then runs
  the selected GUI's `main()`.
- The selected GUI module is imported **lazily**, only after the flag is parsed.
  If the import fails (missing Qt binding, missing `lmfit`, …), the command
  prints `Error: cannot start <mode> (<description>): <detail>` to **stderr**
  and exits with status **1** — other modes are unaffected.
- A missing dependency therefore affects only the mode that needs it;
  see [Installation → Lazy imports](installation.md#lazy-imports-and-optionally-guis).

## Programmatic equivalent

```python
from inspy.__main__ import main

main([])          # resolution calculator
main(['-escan'])  # energy-scan GUI
main(['-qscan'])  # Q-scan GUI
```

GUI modules can also be started directly (the package `__init__` deliberately
re-exports nothing GUI-related — import the concrete module):

```python
import inspy
inspy.main()                                   # resolution calculator (lazy attribute)
from inspy.gui.gui_convfit_qscan_uf import main
main()                                         # Q-scan GUI
```
