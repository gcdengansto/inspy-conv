# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:27:29 2020
@author: dgc
"""

#20261010: §1.1/§6.3 Fix: this package no longer eagerly imports the three GUI modules,
#   their tools (which pull the Qt matplotlib backend), or lmfit. Each GUI is imported
#   on demand by inspy/__main__.py's PROGRAMS dispatch, so a missing fit-dialog
#   dependency (e.g. lmfit, imported by gui_convfit_*_uf.py) no longer breaks -rescalc.
#   Import the concrete modules directly, e.g.
#   `from inspy.gui.gui_convfit_qscan_uf import main`.
