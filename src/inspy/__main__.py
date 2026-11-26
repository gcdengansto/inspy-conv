#src/inspy/__main__.py

import argparse
import sys
from importlib import import_module

# The unified fitting GUI is the default and only GUI launched by the CLI.
# Its module is imported lazily so core package use does not require Qt.

PROGRAMS = {
    'fit':     ('gui.gui_convfit_uf',         'unified energy- and Q-scan fitting GUI'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspy unified energy- and Q-scan convolution fitting")
    parser.add_argument(
        '-fit', dest='func', action='store_const', const='fit',
        help="run the unified energy- and Q-scan fitting GUI (default)")
    parser.set_defaults(func='fit')
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
