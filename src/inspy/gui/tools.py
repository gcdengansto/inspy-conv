import os
import sys

import numpy as np
from matplotlib.backends.backend_qtagg  import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                            QLabel, QPlainTextEdit, QSpinBox, QVBoxLayout,
                            QComboBox, QTableWidget, QTableWidgetItem, QLineEdit)


from ..energy import Energy
from ..instrument.tools import get_tau, _cleanargs, _star, _modvec
from ..instrument.tas_spectr import TripleAxisSpectr
#from ..insfit import FitConv


def _read_text_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as stream:
            lines = stream.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='cp1252') as stream:
            lines = stream.readlines()
    return lines


def _split_data_line(line, delimiter='whitespace'):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    if delimiter == 'whitespace':
        fields = line.split()
    elif delimiter == 'comma':
        fields = line.split(',')
    elif delimiter == 'tab':
        fields = line.split('\t')
    elif delimiter == 'semicolon':
        fields = line.split(';')
    else:
        fields = line.split(delimiter)
    return [field.strip() for field in fields if field.strip()]


def _parse_hklw_lines(lines, skip_lines=0, delimiter='whitespace',
                      h_col=0, k_col=1, l_col=2, w_col=3, i_col=4):
    column_indexes = [h_col, k_col, l_col, w_col, i_col]
    if len(set(column_indexes)) != 5:
        raise ValueError("H, K, L, W, and intensity must use five different columns")
    if min(column_indexes) < 0:
        raise ValueError("Column indexes must be zero-based and non-negative")

    rows = []
    for line_number, line in enumerate(lines):
        if line_number < skip_lines:
            continue
        fields = _split_data_line(line, delimiter)
        if fields is None:
            continue
        try:
            required = max(column_indexes)
            if len(fields) <= required:
                continue
            row = [float(fields[index]) for index in column_indexes]
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(row)):
            continue
        rows.append(row)

    if not rows:
        raise ValueError("No valid numeric data rows matched the current import settings")

    data = np.asarray(rows, dtype=float).T
    iobs = data[4]
    if np.any(iobs < 0):
        raise ValueError("Intensity values must be non-negative")
    return tuple(data[index] for index in range(5)) + (np.sqrt(iobs),)


def _load_hklw(filepath, skip_lines=0, delimiter='whitespace',
               h_col=0, k_col=1, l_col=2, w_col=3, i_col=4):
    r"""Load HKLW data using configurable text-import settings."""
    lines = _read_text_lines(filepath)
    return _parse_hklw_lines(
        lines, skip_lines=skip_lines, delimiter=delimiter,
        h_col=h_col, k_col=k_col, l_col=l_col, w_col=w_col, i_col=i_col)


class HKLWImportDialog(QDialog):
    r"""Text-data preview and HKLW column mapping dialog."""

    def __init__(self, filepath, parent=None):
        super(HKLWImportDialog, self).__init__(parent)
        self.filepath = filepath
        self.lines = _read_text_lines(filepath)
        self.result_data = None

        self.setWindowTitle("Import data")
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(os.path.basename(filepath)))

        preview_layout = QHBoxLayout()
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumWidth(320)
        self.preview_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.preview_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_layout.addWidget(self.preview_text)

        settings = QFormLayout()
        self.skip_lines = QSpinBox()
        self.skip_lines.setRange(0, 100000)
        try:
            first_fields = _split_data_line(self.lines[0]) if self.lines else None
            if first_fields:
                [float(field) for field in first_fields]
            inferred_skip = 0
        except ValueError:
            inferred_skip = 1
        self.skip_lines.setValue(inferred_skip)
        settings.addRow("Header lines to skip:", self.skip_lines)

        self.delimiter = QComboBox()
        self.delimiter.addItem("Whitespace", "whitespace")
        self.delimiter.addItem("Comma", "comma")
        self.delimiter.addItem("Tab", "tab")
        self.delimiter.addItem("Semicolon", "semicolon")
        self.delimiter.addItem("Custom", "custom")
        settings.addRow("Delimiter:", self.delimiter)

        self.custom_delimiter = QLineEdit()
        self.custom_delimiter.setPlaceholderText("Custom delimiter")
        settings.addRow("Custom delimiter:", self.custom_delimiter)

        self.h_col = QSpinBox()
        self.k_col = QSpinBox()
        self.l_col = QSpinBox()
        self.w_col = QSpinBox()
        self.i_col = QSpinBox()
        for spinbox in (self.h_col, self.k_col, self.l_col, self.w_col, self.i_col):
            spinbox.setRange(0, 100)

        self.k_col.setValue(1)
        self.l_col.setValue(2)
        self.w_col.setValue(3)
        self.i_col.setValue(4)
        settings.addRow("H column (0-based):", self.h_col)
        settings.addRow("K column (0-based):", self.k_col)
        settings.addRow("L column (0-based):", self.l_col)
        settings.addRow("W column (0-based):", self.w_col)
        settings.addRow("Intensity column (0-based):", self.i_col)

        settings_widget = QLabel()
        settings_widget.setLayout(settings)
        preview_layout.addWidget(settings_widget)
        layout.addLayout(preview_layout)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["H", "K", "L", "W", "Intensity"])
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.preview_text.setPlainText("".join(self.lines[:50]))

        self.skip_lines.valueChanged.connect(self.refresh_preview)
        self.delimiter.currentIndexChanged.connect(self.refresh_preview)
        self.custom_delimiter.textChanged.connect(self.refresh_preview)
        for spinbox in (self.h_col, self.k_col, self.l_col, self.w_col, self.i_col):
            spinbox.valueChanged.connect(self.refresh_preview)
        self.refresh_preview()

    def refresh_preview(self):
        try:
            delimiter = self.delimiter.currentData()
            if delimiter == "custom":
                delimiter = self.custom_delimiter.text()
                if not delimiter:
                    raise ValueError("Enter a custom delimiter")
            loaded = _parse_hklw_lines(
                self.lines,
                skip_lines=self.skip_lines.value(),
                delimiter=delimiter,
                h_col=self.h_col.value(),
                k_col=self.k_col.value(),
                l_col=self.l_col.value(),
                w_col=self.w_col.value(),
                i_col=self.i_col.value())
        except ValueError as exc:
            self.result_data = None
            self.status_label.setText(str(exc))
            self.status_label.setStyleSheet("color: red")
            self.table.setRowCount(0)
            return

        self.result_data = loaded
        self.status_label.setText(
            "Valid data: {0} rows; dIobs = sqrt(intensity)".format(
                len(loaded[0])))
        self.status_label.setStyleSheet("color: green")

        preview_rows = min(20, len(loaded[0]))
        self.table.setRowCount(preview_rows)
        for row in range(preview_rows):
            for column in range(5):
                item = QTableWidgetItem("{0:.8g}".format(loaded[column][row]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, column, item)

    def accept(self):
        if self.result_data is None:
            self.status_label.setText(
                "Choose valid header, delimiter, and column settings before importing")
            self.status_label.setStyleSheet("color: red")
            return
        super(HKLWImportDialog, self).accept()


def _parse_param_fixed(text, count, name='parameter fixed mask'):
    r"""Parse and validate a whitespace-separated 0/1 parameter mask."""
    if text is None:
        raise ValueError("{0} is empty".format(name))

    tokens = str(text).strip().split()
    if not tokens:
        raise ValueError("{0} is empty".format(name))
    if len(tokens) != count:
        raise ValueError(
            "{0} must contain exactly {1} values; got {2}".format(
                name, count, len(tokens)))

    mask = []
    for token in tokens:
        if token not in ('0', '1'):
            raise ValueError(
                "{0} contains invalid value '{1}'; use only 0 or 1".format(
                    name, token))
        mask.append(int(token))
    return mask


def _parse_curvature_radius(text, label):
    """Return a curvature radius entered in cm, or infinity for a flat crystal."""
    value = str(text).strip()
    if value.lower() in ('', 'none', 'flat', 'inf', 'infinity'):
        return np.inf

    try:
        radius = float(value)
    except ValueError as exc:
        raise ValueError(
            "{0} curvature must be a positive radius in cm or 'None' for flat".format(
                label)) from exc

    if not np.isfinite(radius) or radius <= 0:
        if np.isposinf(radius):
            return radius
        raise ValueError(
            "{0} curvature must be a positive radius in cm or 'None' for flat".format(
                label))
    return radius


def _detect_qscan_axes(h, k, l):
    """Return HKL axes that vary beyond numerical rounding noise."""
    varying = []
    for axis, values in zip(('QH', 'QK', 'QL'), (h, k, l)):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            continue
        scale = max(1.0, float(np.max(np.abs(values))))
        tolerance = max(1e-10, 1e-8 * scale)
        if float(np.max(values) - np.min(values)) > tolerance:
            varying.append(axis)
    return varying


def angle2(x, y, z, h, k, l, lattice):
    r"""Function necessary for Prefactor functions
    """
    #20261010 Bug #2.7: clip the cosine into [-1, 1] before np.arccos.
    #   np.arccos(1 + 2e-16) is NaN, and the quotient below really does reach
    #   1 +/- round-off: the two vectors are exact multiples of each other when
    #   (x,y,z) and (h,k,l) are collinear -- which for PrefDemoFF's three calls
    #   ((1,0,0), (0,1,0), (0,0,1) against (H,K,L)) happens whenever two of
    #   H,K,L are pinned at 0, i.e. at EVERY point of a 1D scan along a
    #   reciprocal axis: the usual case.  The NaN then lands in polx/poly/polz
    #   and in prefactor, and scipy.optimize.least_squares aborts with
    #   "Residuals are not finite in the initial point." -- a hard fit failure.
    #   Clipping cannot mask a real error: get_lattice() deg2rad's the cell
    #   angles exactly as GetLattice.m:15-17 does, and _scalar/_star mirror
    #   scalar.m/star.m on those radians, so by Cauchy-Schwarz the quotient is
    #   a genuine cosine -- anything outside [-1,1] is round-off alone.
    #   ResLib's angle2.m:17 carries the same latent defect (MATLAB's acos
    #   returns NaN too) but its demo only plots the NaN instead of fitting.
    #   NOT covered here: (h,k,l) == (0,0,0) gives |Q| == 0 -> 0/0 = NaN, which
    #   np.clip passes through -- a separate, pre-existing division by zero.
    latticestar = _star(lattice)[-1]

    return np.arccos(np.clip(
        2 * np.pi * (h * x + k * y + l * z) / _modvec([x, y, z], lattice) / _modvec(
            [h, k, l], latticestar), -1.0, 1.0))

def SelFormFactor(magion=''):
    formfactList = [
        {"magion": "Sc0", "AA": 0.251200, "aa": 90.029600, "BB": 0.329000, "bb": 39.402100, "CC": 0.423500, "cc": 14.322200, "DD": -0.004300 },
        {"magion": "Sc1", "AA": 0.488900, "aa": 51.160300, "BB": 0.520300, "bb": 14.076400, "CC": -0.028600, "cc": 0.179200, "DD": 0.018500 },
        {"magion": "Sc2", "AA": 0.504800, "aa": 31.403500, "BB": 0.518600, "bb": 10.989700, "CC": -0.024100, "cc": 1.183100, "DD": 0.000000 },
        {"magion": "Ti0", "AA": 0.465700, "aa": 33.589800, "BB": 0.549000, "bb": 9.879100, "CC": -0.029100, "cc": 0.323200, "DD": 0.012300 },
        {"magion": "Ti1", "AA": 0.509300, "aa": 36.703300, "BB": 0.503200, "bb": 10.371300, "CC": -0.026300, "cc": 0.310600, "DD": 0.011600 },
        {"magion": "Ti2", "AA": 0.509100, "aa": 24.976300, "BB": 0.516200, "bb": 8.756900, "CC": -0.028100, "cc": 0.916000, "DD": 0.001500 },
        {"magion": "Ti3", "AA": 0.357100, "aa": 22.841300, "BB": 0.668800, "bb": 8.930600, "CC": -0.035400, "cc": 0.483300, "DD": 0.009900 },
        {"magion": "V0", "AA": 0.408600, "aa": 28.810900, "BB": 0.607700, "bb": 8.543700, "CC": -0.029500, "cc": 0.276800, "DD": 0.012300 },
        {"magion": "V1", "AA": 0.444400, "aa": 32.647900, "BB": 0.568300, "bb": 9.097100, "CC": -0.228500, "cc": 0.021800, "DD": 0.215000 },
        {"magion": "V2", "AA": 0.408500, "aa": 23.852600, "BB": 0.609100, "bb": 8.245600, "CC": -0.167600, "cc": 0.041500, "DD": 0.149600 },
        {"magion": "V3", "AA": 0.359800, "aa": 19.336400, "BB": 0.663200, "bb": 7.617200, "CC": -0.306400, "cc": 0.029600, "DD": 0.283500 },
        {"magion": "V4", "AA": 0.310600, "aa": 16.816000, "BB": 0.719800, "bb": 7.048700, "CC": -0.052100, "cc": 0.302000, "DD": 0.022100 },
        {"magion": "Cr0", "AA": 0.113500, "aa": 45.199000, "BB": 0.348100, "bb": 19.493100, "CC": 0.547700, "cc": 7.354200, "DD": -0.009200 },
        {"magion": "Cr1", "AA": -0.097700, "aa": 0.047000, "BB": 0.454400, "bb": 26.005400, "CC": 0.557900, "cc": 7.489200, "DD": 0.083100 },
        {"magion": "Cr2", "AA": 1.202400, "aa": -0.005500, "BB": 0.415800, "bb": 20.547500, "CC": 0.603200, "cc": 6.956000, "DD": -1.221800 },
        {"magion": "Cr3", "AA": -0.309400, "aa": 0.027400, "BB": 0.368000, "bb": 17.035500, "CC": 0.655900, "cc": 6.523600, "DD": 0.285600 },
        {"magion": "Cr4", "AA": -0.232000, "aa": 0.043300, "BB": 0.310100, "bb": 14.951800, "CC": 0.718200, "cc": 6.172600, "DD": 0.204200 },
        {"magion": "Mn0", "AA": 0.243800, "aa": 24.962900, "BB": 0.147200, "bb": 15.672800, "CC": 0.618900, "cc": 6.540300, "DD": -0.010500 },
        {"magion": "Mn1", "AA": -0.013800, "aa": 0.421300, "BB": 0.423100, "bb": 24.668000, "CC": 0.590500, "cc": 6.654500, "DD": -0.001000 },
        {"magion": "Mn2", "AA": 0.422000, "aa": 17.684000, "BB": 0.594800, "bb": 6.005000, "CC": 0.004300, "cc": -0.609000, "DD": -0.021900 },
        {"magion": "Mn3", "AA": 0.419800, "aa": 14.282900, "BB": 0.605400, "bb": 5.468900, "CC": 0.924100, "cc": -0.008800, "DD": -0.949800 },
        {"magion": "Mn4", "AA": 0.376000, "aa": 12.566100, "BB": 0.660200, "bb": 5.132900, "CC": -0.037200, "cc": 0.563000, "DD": 0.001100 },
        {"magion": "Mn5", "AA": 0.292400, "aa": 11.665500, "BB": 0.740500, "bb": 5.074100, "CC": -1.788300, "cc": 0.005900, "DD": 1.755700 },
        {"magion": "Fe0", "AA": 0.070600, "aa": 35.008500, "BB": 0.358900, "bb": 15.358300, "CC": 0.581900, "cc": 5.560600, "DD": -0.011400 },
        {"magion": "Fe1", "AA": 0.125100, "aa": 34.963300, "BB": 0.362900, "bb": 15.514400, "CC": 0.522300, "cc": 5.591400, "DD": -0.010500 },
        {"magion": "Fe2", "AA": 0.026300, "aa": 34.959700, "BB": 0.366800, "bb": 15.943500, "CC": 0.618800, "cc": 5.593500, "DD": -0.011900 },
        {"magion": "Fe3", "AA": 0.397200, "aa": 13.244200, "BB": 0.629500, "bb": 4.903400, "CC": -0.031400, "cc": 0.349600, "DD": 0.004400 },
        {"magion": "Fe4", "AA": 0.378200, "aa": 11.380000, "BB": 0.655600, "bb": 4.592000, "CC": -0.034600, "cc": 0.483300, "DD": 0.000500 },
        {"magion": "Co0", "AA": 0.413900, "aa": 16.161600, "BB": 0.601300, "bb": 4.780500, "CC": -0.151800, "cc": 0.021000, "DD": 0.134500 },
        {"magion": "Co1", "AA": 0.099000, "aa": 33.125200, "BB": 0.364500, "bb": 15.176800, "CC": 0.547000, "cc": 5.008100, "DD": -0.010900 },
        {"magion": "Co2", "AA": 0.433200, "aa": 14.355300, "BB": 0.585700, "bb": 4.607700, "CC": -0.038200, "cc": 0.133800, "DD": 0.017900 },
        {"magion": "Co3", "AA": 0.390200, "aa": 12.507800, "BB": 0.632400, "bb": 4.457400, "CC": -0.150000, "cc": 0.034300, "DD": 0.127200 },
        {"magion": "Co4", "AA": 0.351500, "aa": 10.778500, "BB": 0.677800, "bb": 4.234300, "CC": -0.038900, "cc": 0.240900, "DD": 0.009800 },
        {"magion": "Ni0", "AA": -0.017200, "aa": 35.739200, "BB": 0.317400, "bb": 14.268900, "CC": 0.713600, "cc": 4.566100, "DD": -0.014300 },
        {"magion": "Ni1", "AA": 0.070500, "aa": 35.856100, "BB": 0.398400, "bb": 13.804200, "CC": 0.542700, "cc": 4.396500, "DD": -0.011800 },
        {"magion": "Ni2", "AA": 0.016300, "aa": 35.882600, "BB": 0.391600, "bb": 13.223300, "CC": 0.605200, "cc": 4.338800, "DD": -0.013300 },
        {"magion": "Ni3", "AA": 0.001200, "aa": 34.999800, "BB": 0.346800, "bb": 11.987400, "CC": 0.666700, "cc": 4.251800, "DD": -0.014800 },
        {"magion": "Ni4", "AA": -0.009000, "aa": 35.861400, "BB": 0.277600, "bb": 11.790400, "CC": 0.747400, "cc": 4.201100, "DD": -0.016300 },
        {"magion": "Cu0", "AA": 0.090900, "aa": 34.983800, "BB": 0.408800, "bb": 11.443200, "CC": 0.512800, "cc": 3.824800, "DD": -0.012400 },
        {"magion": "Cu1", "AA": 0.074900, "aa": 34.965600, "BB": 0.414700, "bb": 11.764200, "CC": 0.523800, "cc": 3.849700, "DD": -0.012700 },
        {"magion": "Cu2", "AA": 0.023200, "aa": 34.968600, "BB": 0.402300, "bb": 11.564000, "CC": 0.588200, "cc": 3.842800, "DD": -0.013700 },
        {"magion": "Cu3", "AA": 0.003100, "aa": 34.907400, "BB": 0.358200, "bb": 10.913800, "CC": 0.653100, "cc": 3.827900, "DD": -0.014700 },
        {"magion": "Cu4", "AA": -0.013200, "aa": 30.681700, "BB": 0.280100, "bb": 11.162600, "CC": 0.749000, "cc": 3.817200, "DD": -0.016500 },
        {"magion": "Y0", "AA": 0.591500, "aa": 67.608100, "BB": 1.512300, "bb": 17.900400, "CC": -1.113000, "cc": 14.135900, "DD": 0.008000 },
        {"magion": "Zr0", "AA": 0.410600, "aa": 59.996100, "BB": 1.054300, "bb": 18.647600, "CC": -0.475100, "cc": 10.540000, "DD": 0.010600 },
        {"magion": "Zr1", "AA": 0.453200, "aa": 59.594800, "BB": 0.783400, "bb": 21.435700, "CC": -0.245100, "cc": 9.036000, "DD": 0.009800 },
        {"magion": "Nb0", "AA": 0.394600, "aa": 49.229700, "BB": 1.319700, "bb": 14.821600, "CC": -0.726900, "cc": 9.615600, "DD": 0.012900 },
        {"magion": "Nb1", "AA": 0.457200, "aa": 49.918200, "BB": 1.027400, "bb": 15.725600, "CC": -0.496200, "cc": 9.157300, "DD": 0.011800 },
        {"magion": "Mo0", "AA": 0.180600, "aa": 49.056800, "BB": 1.230600, "bb": 14.785900, "CC": -0.426800, "cc": 6.986600, "DD": 0.017100 },
        {"magion": "Mo1", "AA": 0.350000, "aa": 48.035400, "BB": 1.030500, "bb": 15.060400, "CC": -0.392900, "cc": 7.479000, "DD": 0.013900 },
        {"magion": "Tc0", "AA": 0.129800, "aa": 49.661100, "BB": 1.165600, "bb": 14.130700, "CC": -0.313400, "cc": 5.512900, "DD": 0.019500 },
        {"magion": "Tc1", "AA": 0.267400, "aa": 48.956600, "BB": 0.956900, "bb": 15.141300, "CC": -0.238700, "cc": 5.457800, "DD": 0.016000 },
        {"magion": "Ru0", "AA": 0.106900, "aa": 49.423800, "BB": 1.191200, "bb": 12.741700, "CC": -0.317600, "cc": 4.912500, "DD": 0.021300 },
        {"magion": "Ru1", "AA": 0.441000, "aa": 33.308600, "BB": 1.477500, "bb": 9.553100, "CC": -0.936100, "cc": 6.722000, "DD": 0.017600 },
        {"magion": "Rh0", "AA": 0.097600, "aa": 49.882500, "BB": 1.160100, "bb": 11.830700, "CC": -0.278900, "cc": 4.126600, "DD": 0.023400 },
        {"magion": "Rh1", "AA": 0.334200, "aa": 29.756400, "BB": 1.220900, "bb": 9.438400, "CC": -0.575500, "cc": 5.332000, "DD": 0.021000 },
        {"magion": "Pd0", "AA": 0.200300, "aa": 29.363300, "BB": 1.144600, "bb": 9.599300, "CC": -0.368900, "cc": 4.042300, "DD": 0.025100 },
        {"magion": "Pd1", "AA": 0.503300, "aa": 24.503700, "BB": 1.998200, "bb": 6.908200, "CC": -1.524000, "cc": 5.513300, "DD": 0.021300 },
        {"magion": "Ce2", "AA": 0.295300, "aa": 17.684600, "BB": 0.292300, "bb": 6.732900, "CC": 0.431300, "cc": 5.382700, "DD": -0.019400 },
        {"magion": "Nd2", "AA": 0.164500, "aa": 25.045300, "BB": 0.252200, "bb": 11.978200, "CC": 0.601200, "cc": 4.946100, "DD": -0.018000 },
        {"magion": "Nd3", "AA": 0.054000, "aa": 25.029300, "BB": 0.310100, "bb": 12.102000, "CC": 0.657500, "cc": 4.722300, "DD": -0.021600 },
        {"magion": "Sm2", "AA": 0.090900, "aa": 25.203200, "BB": 0.303700, "bb": 11.856200, "CC": 0.625000, "cc": 4.236600, "DD": -0.020000 },
        {"magion": "Sm3", "AA": 0.028800, "aa": 25.206800, "BB": 0.297300, "bb": 11.831100, "CC": 0.695400, "cc": 4.211700, "DD": -0.021300 },
        {"magion": "Eu2", "AA": 0.075500, "aa": 25.296000, "BB": 0.300100, "bb": 11.599300, "CC": 0.643800, "cc": 4.025200, "DD": -0.019600 },
        {"magion": "Eu3", "AA": 0.020400, "aa": 25.307800, "BB": 0.301000, "bb": 11.474400, "CC": 0.700500, "cc": 3.942000, "DD": -0.022000 },
        {"magion": "Gd2", "AA": 0.063600, "aa": 25.382300, "BB": 0.303300, "bb": 11.212500, "CC": 0.652800, "cc": 3.787700, "DD": -0.019900 },
        {"magion": "Gd3", "AA": 0.018600, "aa": 25.386700, "BB": 0.289500, "bb": 11.142100, "CC": 0.713500, "cc": 3.752000, "DD": -0.021700 },
        {"magion": "Tb2", "AA": 0.054700, "aa": 25.508600, "BB": 0.317100, "bb": 10.591100, "CC": 0.649000, "cc": 3.517100, "DD": -0.021200 },
        {"magion": "Tb3", "AA": 0.017700, "aa": 25.509500, "BB": 0.292100, "bb": 10.576900, "CC": 0.713300, "cc": 3.512200, "DD": -0.023100 },
        {"magion": "Dy2", "AA": 0.130800, "aa": 18.315500, "BB": 0.311800, "bb": 7.664500, "CC": 0.579500, "cc": 3.146900, "DD": -0.022600 },
        {"magion": "Dy3", "AA": 0.115700, "aa": 15.073200, "BB": 0.327000, "bb": 6.799100, "CC": 0.582100, "cc": 3.020200, "DD": -0.024900 },
        {"magion": "Ho2", "AA": 0.099500, "aa": 18.176100, "BB": 0.330500, "bb": 7.855600, "CC": 0.592100, "cc": 2.979900, "DD": -0.023000 },
        {"magion": "Ho3", "AA": 0.056600, "aa": 18.317600, "BB": 0.336500, "bb": 7.688000, "CC": 0.631700, "cc": 2.942700, "DD": -0.024800 },
        {"magion": "Er2", "AA": 0.112200, "aa": 18.122300, "BB": 0.346200, "bb": 6.910600, "CC": 0.564900, "cc": 2.761400, "DD": -0.023500 },
        {"magion": "Er3", "AA": 0.058600, "aa": 17.980200, "BB": 0.354000, "bb": 7.096400, "CC": 0.612600, "cc": 2.748200, "DD": -0.025100 },
        {"magion": "Tm2", "AA": 0.098300, "aa": 18.323600, "BB": 0.338000, "bb": 6.917800, "CC": 0.587500, "cc": 2.662200, "DD": -0.024100 },
        {"magion": "Tm3", "AA": 0.058100, "aa": 15.092200, "BB": 0.278700, "bb": 7.801500, "CC": 0.685400, "cc": 2.793100, "DD": -0.022400 },
        {"magion": "Yb2", "AA": 0.085500, "aa": 18.512300, "BB": 0.294300, "bb": 7.373400, "CC": 0.641200, "cc": 2.677700, "DD": -0.021300 },
        {"magion": "Yb3", "AA": 0.041600, "aa": 16.094900, "BB": 0.284900, "bb": 7.834100, "CC": 0.696100, "cc": 2.672500, "DD": -0.022900 },
        {"magion": "Pr3", "AA": 0.050400, "aa": 24.998900, "BB": 0.257200, "bb": 12.037700, "CC": 0.714200, "cc": 5.003900, "DD": -0.021900 },
        {"magion": "U3", "AA": 0.505800, "aa": 23.288200, "BB": 1.346400, "bb": 7.002800, "CC": -0.872400, "cc": 4.868300, "DD": 0.019200 },
        {"magion": "U4", "AA": 0.329100, "aa": 23.547500, "BB": 1.083600, "bb": 8.454000, "CC": -0.434000, "cc": 4.119600, "DD": 0.021400 },
        {"magion": "U5", "AA": 0.365000, "aa": 19.803800, "BB": 3.219900, "bb": 6.281800, "CC": -2.607700, "cc": 5.301000, "DD": 0.023300 },
        {"magion": "Np3", "AA": 0.515700, "aa": 20.865400, "BB": 2.278400, "bb": 5.893000, "CC": -1.816300, "cc": 4.845700, "DD": 0.021100 },
        {"magion": "Np4", "AA": 0.420600, "aa": 19.804600, "BB": 2.800400, "bb": 5.978300, "CC": -2.243600, "cc": 4.984800, "DD": 0.022800 },
        {"magion": "Np5", "AA": 0.369200, "aa": 18.190000, "BB": 3.151000, "bb": 5.850000, "CC": -2.544600, "cc": 4.916400, "DD": 0.024800 },
        {"magion": "Np6", "AA": 0.292900, "aa": 17.561100, "BB": 3.486600, "bb": 5.784700, "CC": -2.806600, "cc": 4.870700, "DD": 0.026700 },
        {"magion": "Pu3", "AA": 0.384000, "aa": 16.679300, "BB": 3.104900, "bb": 5.421000, "CC": -2.514800, "cc": 4.551200, "DD": 0.026300 },
        {"magion": "Pu4", "AA": 0.493400, "aa": 16.835500, "BB": 1.639400, "bb": 5.638400, "CC": -1.158100, "cc": 4.139900, "DD": 0.024800 },
        {"magion": "Pu5", "AA": 0.388800, "aa": 16.559200, "BB": 2.036200, "bb": 5.656700, "CC": -1.451500, "cc": 4.255200, "DD": 0.026700 },
        {"magion": "Pu6", "AA": 0.317200, "aa": 16.050700, "BB": 3.465400, "bb": 5.350700, "CC": -2.810200, "cc": 4.513300, "DD": 0.028100 },
        {"magion": "Am2", "AA": 0.474300, "aa": 21.776100, "BB": 1.580000, "bb": 5.690200, "CC": -1.077900, "cc": 4.145100, "DD": 0.021800 },
        {"magion": "Am3", "AA": 0.423900, "aa": 19.573900, "BB": 1.457300, "bb": 5.872200, "CC": -0.905200, "cc": 3.968200, "DD": 0.023800 },
        {"magion": "Am4", "AA": 0.373700, "aa": 17.862500, "BB": 1.352100, "bb": 6.042600, "CC": -0.751400, "cc": 3.719900, "DD": 0.025800 },
        {"magion": "Am5", "AA": 0.295600, "aa": 17.372500, "BB": 1.452500, "bb": 6.073400, "CC": -0.775500, "cc": 3.661900, "DD": 0.027700 },
        {"magion": "Am6", "AA": 0.230200, "aa": 16.953300, "BB": 1.486400, "bb": 6.115900, "CC": -0.745700, "cc": 3.542600, "DD": 0.029400 },
        {"magion": "Am7", "AA": 0.360100, "aa": 12.729900, "BB": 1.964000, "bb": 5.120300, "CC": -1.356000, "cc": 3.714200, "DD": 0.031600 }
    ]
    for item in formfactList:
        if magion.upper() == item["magion"].upper():
            print(f"The form factor of {magion} is found.")
            return item
    print(f"The mangetic ion {magion} is not found.")
    return None


def SqwDemo(H, K, L, W, p):
    r"""Example Scattering function for convolution tests
    """
    #print(isinstance(W,np.ndarray))
    #print(np.shape(W))
    en1     =    p[0]   #peak 1
    en2     =    p[1]   #peak 2
    IntRatio=    p[2]
    
    Gamma1 =     p[3]
    Gamma2 =     p[4]  #peak-2 Lorentzian HWHM (used in A2/B2 below)
    #Int= p[5]
    #Bg=p[6]
    temp   =     p[7]
    
    A1=1/((W-en1)**2+Gamma1**2)
    B1=1/((W+en1)**2+Gamma1**2)
    
    A2=1/((W-en2)**2+Gamma2**2)
    B2=1/((W+en2)**2+Gamma2**2)
    
    #BF=1+1/(1-np.exp(W/temp/0.08713))
    # To avoid numerical instability when W is close to zero, we can use np.expm1, which computes exp(x) - 1 accurately even for small x.
    #20261010 Bug #2.6: Bose factor was 1-n; must be 1+n.
    #   Detailed balance for S(Q,W) reads S(Q,-W) = exp(-x)*S(Q,W), x = W/(kB*T).
    #   The Lorentzian difference below, (A1-B1), is ODD in W because
    #   A1(-W) = B1(W), so the factor multiplying it must satisfy
    #       BF(-x) = -exp(-x)*BF(x)
    #   and only the PLUS sign does:
    #       1 + 1/expm1(x)    = e^x/(e^x-1)      BF(-x) = 1/(1-e^x) = -exp(-x)*BF(x)
    #       1 + 1/(-expm1(x)) = (e^x-2)/(e^x-1)  would require e^x = -1, i.e. NEVER
    #   The wrong sign also makes BF NEGATIVE for 0 < x < ln2, i.e. below
    #   ~0.693*kB*T (~0.30 meV at 5 K): a negative scattered intensity.
    #   Minimal change: drop the leading '-' on the last line only.  The np.expm1
    #   stabilisation and the |expm1|<1e-10 guard stay as they were (the guard only
    #   keeps the W==0 product finite -- BF itself diverges there while A1-B1 -> 0).
    expm1 = np.expm1(W / temp / 0.08713)          # numerically stable exp(x)-1
    BF    = np.where(np.abs(expm1) < 1e-10,
                    1.0,                          # BF has no finite limit at W->0; pin it so BF*(A1-B1) stays finite
                    1 + 1 / expm1)
    
    sqw0=IntRatio*Gamma1*BF*(A1-B1) + Gamma2*BF*(A2-B2)
    sqw1=IntRatio*Gamma1*BF*(A1-B1) + Gamma2*BF*(A2-B2)
    #sqw1=np.zeros(sqw0.shape)
    
    sqw = np.vstack((sqw0, sqw1, sqw1))
    #print(np.size(sqw))
    #print(sqw0)
    
    return sqw



def PrefDemoFF(H, K, L, W, EXP, p):
    r"""Prefactor example for convolution tests
    """
    [sample, rsample] = EXP.get_lattice()

    q2 = _modvec([H, K, L], rsample) ** 2
    if len(p) == 15:
        AA =   p[8]
        aa =   p[9]
        BB =   p[10]
        bb =   p[11]
        CC =   p[12]
        cc =   p[13]
        DD =   p[14]
        
        # Jane Brown 4-Gaussian approximation; AA..DD/aa..dd are whichever ion's
        # coefficients the caller passed in p[8:15] (not hardcoded to Co2+)
        sd = q2 / (16 * np.pi ** 2)
        ff = AA*np.exp(-aa*sd) + BB*np.exp(-bb*sd) + CC*np.exp(-cc*sd) + DD
    else:
        ff = np.ones(q2.shape)  # no Form Factor
    
    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)

    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2

    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = ff ** 2.0 * polx * p[5]
    prefactor[1, :] = ff ** 2.0 * poly * p[5]
    prefactor[2, :] = ff ** 2.0 * polz * p[5]

    bgr = np.ones(H.shape) * p[6]
    

    return [prefactor, bgr]



def SqwQScanTwoPeaks(H, K, L, W, p):
    r"""Example Scattering function for convolution tests
    """

    # Extract the parameters contained in "p":
    q1    =   p[0]                    # peak1 q position
    q2    =   p[1]                    # peak2 q position
    ratio =   p[2]                    # Intensity ratio 
    w1    =   p[3] 
    w2    =   p[4]                    # peak width

    temp   =     p[7]

    #20261010 Bug #2.6: same Bose factor as SqwDemo above -- it was 1-n, must be 1+n.
    #   1 - np.exp(x) is exactly -expm1(x), so the old denominator was -(e^x-1) and
    #   BF came out as 1-n = (e^x-2)/(e^x-1) instead of the Bose factor
    #   1+n = 1/(1-e^-x) = e^x/(e^x-1).  1-n is NEGATIVE for 0 < x < ln2, i.e. below
    #   ~0.693*kB*T, which turns the scattered intensity negative in that window.
    #   The commented-out original of SqwDemo (line :170) shows the very form this
    #   site still carried; np.expm1 repairs the sign and inherits the small-x
    #   accuracy that SqwDemo's refactor to expm1 already bought.
    #   Pre-existing and deliberately left alone: the W == 0 division by zero
    #   (np.expm1(0) == 0 -> +inf here, exactly as 1-np.exp() did before this fix).
    BF     = 1 + 1/np.expm1(W/temp/0.08713)

    A1=1/(w1*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(H-q1)**2/w1**2)
    
    A2=1/(w2*np.sqrt(np.pi/(4*np.log(2)))) * np.exp(-4*np.log(2)*(H-q2)**2/w2**2)

    sqw0 = (A1 + ratio*A2)* BF  #   = A1 + ratio*A2
    sqw1 = np.zeros(sqw0.shape)
    sqw2 = np.zeros(sqw0.shape)



    #print(np.size(sqw))
    sqw = np.vstack((sqw0, sqw1, sqw2))
    #print(sqw)
    return sqw