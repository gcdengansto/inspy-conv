# Development

## Setup

```bash
git clone https://github.com/gcdengansto/inspy-conv
cd inspy-conv
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest tests/
```

> The suite is small (`test_tas.py`, `test_mc.py`, `inspy_test.py`) — it
> exercises the TAS engine and conversions, not the GUIs. The files
> `tests/main_gui.py`, `tests/res_3D_plot.py` are scratch scripts for manual
> GUI runs, not pytest modules.

## Conventions

### Fix markers

Every behavioural fix since 2026-10-10 carries a marker comment:

```python
#20261010: <tag> Fix: <what was wrong and what changed>
```

Tags used so far: `L1`…`L11` (low-priority cleanups), `M1`…`M11` (medium),
`H1` (high), `Bug #2.x` (ResLib-port divergences), `§n.m` (review-report items).
When editing a marked block, keep the marker with the code it describes.

### Comment style

- Corrections to the ported MATLAB comments use `#20261010:` markers with the
  reasoning inline (why, not just what).
- Known-but-unfixed issues are marked `KNOWN ISSUE (...)` with a pointer to the
  review report section.
- Original-author credit: where a file is adapted from ResLib 3.4c or
  neutronpy, the header says so — do not remove attribution.

### Line references

Review comments cite `file.py:NNN`. Insertions inside cited ranges shift the
numbers: prefer **line-neutral edits** (fold a note into an existing comment
line) or update the inbound references in the same pass.

### Line endings

Keep files **LF**. `tas_spectr.py` is LF-only, no BOM, no trailing newline.

## Review reports

The repository root carries the static-review trail (kept as historical
records, not part of the package):

| File | Contents |
|------|----------|
| `proj_potential_problems.md` | Initial problem inventory |
| `proj_review_output.md` | First full static review |
| `proj_review_output_v2.md` | Second pass + fixes |
| `proj_review_output_v3.md` | Findings + §9 fix log (code fixes) |
| `proj_review_comments_v4.md` | Comment/docstring audit + status notes + recommended actions |

Fix work follows: read the report section → verify against the current tree →
apply → append a **Status note** under the section.

## Documentation

Sources live in `docs/` (Markdown, MkDocs-compatible):

```bash
pip install mkdocs
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site/ folder (deploy anywhere)
```

To publish online, push `mkdocs build` output to GitHub Pages
(`mkdocs gh-deploy`) or any static host. The nav is defined in `mkdocs.yml` —
add new pages there.

When you change an API:

1. Update the docstring in the source.
2. Update the matching page in `docs/`.
3. Check the examples in `README.md`.

## Known limitations (tracked, not forgotten)

| Item | Where documented |
|------|------------------|
| `bounds` not forwarded to `least_squares` | `fitconv.py` `fitwithconv` note |
| `cache_size` reserved but unused | `uffitconv.py` note |
| Magnetic structure factor partially stubbed | `crystal/structure_factors.py` |
| `TimeOfFlightSpectr` stub-level | `instrument/tof_spectr.py` |
| Dead 2.6/3.3 version check | `inspy/__init__.py` note |
| Demo S(Q,ω) models are examples, not physics-complete | `gui/tools.py` |

Code-level review status lives in `proj_review_comments_v4.md` §6/§7.
