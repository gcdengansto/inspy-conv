# CLI reference

Installing the package adds one console command, **`inspy-conv`**
(defined in `pyproject.toml` as `inspy.__main__:main`).

## Synopsis

```bash
inspy-conv [-h] [-fit]
```

## Options

| Flag | Effect |
|------|--------|
| *(none)* | Launch the unified energy- and Q-scan fitting GUI (default) |
| `-fit` | Same as default — unified fitting GUI (`gui_convfit_uf`) |
| `-h`, `--help` | Show usage and exit |

## Examples

```bash
inspy-conv              # unified E/Q-scan fitting
inspy-conv -fit         # same GUI, explicit mode
python -m inspy         # equivalent module form
```

## Behaviour

- The selected GUI module is imported lazily when the command is run.
- If import fails (for example, because Qt or `lmfit` is missing), the command
  prints an error to **stderr** and exits with status **1**.
- Core APIs such as `TripleAxisSpectr` remain importable without launching the
  GUI.

## Programmatic equivalent

```python
import inspy
inspy.main()  # unified energy- and Q-scan fitting GUI
```

The GUI can also be started directly:

```python
from inspy.gui.gui_convfit_uf import main
main()
```
