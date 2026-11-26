# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:27:29 2020
@author: dgc
"""



import sys
import warnings
from importlib.metadata import version, PackageNotFoundError

from . import constants
from . import instrument
from .crystal import Lattice
from .crystal import Material
from .crystal import Sample
from .crystal import SpaceGroup
from .energy import Energy
from .instrument import TripleAxisSpectr, TimeOfFlightSpectr, tools
from .insfit import FitConv, UltraFastFitConv


def __getattr__(name):
    # Resolve the unified fit GUI on demand so `import inspy` does not load Qt.
    if name == 'main':
        try:
            from .gui.gui_convfit_uf import main
        except ImportError:
            warnings.warn(
                'GUI dependencies are unavailable; cannot run the unified fitting GUI.')
            raise
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    __version__ = version("inspy-conv")   # matches name in pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"

#20261010: §3.3 note: this 2.6/3.3 check is dead (package is 0.3.3) and the promised
#   0.2.7 drop never happened; left as-is (removal is a code change, pending).
if sys.version_info[:2] == (2, 6) or sys.version_info[:2] == (3, 3):
    warnings.warn('Support for Python 2.6 and Python 3.3 is deprecated and will be dropped in inspy 0.2.7', DeprecationWarning)
