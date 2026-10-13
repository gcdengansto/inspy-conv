#src/inspy/__main__.py

import argparse
import sys
from importlib import import_module

#20261010: CLI dispatch fix (originally read `arg.func` -> AttributeError, later took a
#   `-func MODE` value): now uses mode-switch flags -rescalc / -escan / -qscan
#   (mutually exclusive). No flag -> rescalc, so bare `inspy-conv` starts the resolution
#   GUI. Loading is genuinely lazy as of the §1.1/§6.3 code fix: inspy/__init__.py no
#   longer imports the GUI stack at package import (its `main` is resolved on first
#   access via a PEP 562 __getattr__), and inspy/gui/__init__.py imports nothing eagerly.
#   The selected GUI module is imported here only when its flag is chosen, so a missing
#   fit-dialog dependency (e.g. lmfit, imported by gui_convfit_*_uf.py) fails only that
#   mode via the per-mode error handler below, not -rescalc (proj_review_comments_v4.md §1.1).

PROGRAMS = {
    'rescalc': ('gui.main_gui',               'Resolution calculation GUI'),
    'escan':   ('gui.gui_convfit_escan_uf',   'energy-scan fitting GUI'),
    'qscan':   ('gui.gui_convfit_qscan_uf',   'Q-scan fitting GUI'),
}


def main(argv=None):
    parser=argparse.ArgumentParser(description="Inspy Triple Axis Spectrometer Resolution Calculate and Convolution")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-rescalc', dest='func', action='store_const', const='rescalc',
                      help="run the resolution calculation GUI (default)")
    mode.add_argument('-escan', dest='func', action='store_const', const='escan',
                      help="run the energy-scan fitting GUI (gui_convfit_escan_uf)")
    mode.add_argument('-qscan', dest='func', action='store_const', const='qscan',
                      help="run the Q-scan fitting GUI (gui_convfit_qscan_uf)")
    parser.set_defaults(func='rescalc')
    args = parser.parse_args(argv)

    print(f"Running inspy in the {args.func} mode.")

    modpath, description = PROGRAMS[args.func]
    try:
        # relative import when executed as a package (__main__ via console script or
        # `python -m inspy`); absolute fallback for plain-file execution
        module = import_module('.' + modpath, package=__package__) if __package__ \
            else import_module('inspy.' + modpath)
        run = module.main
    except (ImportError, AttributeError) as exc:
        print(f"Error: cannot start {args.func} ({description}): {exc}", file=sys.stderr)
        sys.exit(1)

    run()

if __name__ == "__main__":
    main()
