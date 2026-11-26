# -*- coding: utf-8 -*-
r"""Unified energy- and Q-scan two-peak fitting GUI.
"""
import json
import os
import sys

import numpy as np
from matplotlib.backends.backend_qtagg  import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy  import uic
from qtpy.QtWidgets import (QApplication, QFileDialog, QDialog, QMainWindow,
                            QMessageBox, QInputDialog, QSizePolicy, QVBoxLayout,
                            QTextEdit)

from ..energy import Energy
from ..instrument.tools import get_tau, _cleanargs
from ..instrument.tas_spectr import TripleAxisSpectr
from ..insfit import UltraFastFitConv
from .tools import (SqwDemo, SqwQScanTwoPeaks, PrefDemoFF, SelFormFactor,
                    HKLWImportDialog, _parse_curvature_radius,
                    _parse_param_fixed)


_PARAMETER_FIELDS = (
    'fit_param_en1', 'fit_param_en2', 'fit_param_ratio',
    'fit_param_gamma1', 'fit_param_gamma2', 'fit_param_int',
    'fit_param_bg', 'fit_param_temp',
)
_SCAN_PARAMETER_DEFAULTS = {
    'energy': ('1.6', '3.3', '1', '0.13', '0.1', '12000', '0', '1.5'),
    'q': ('0.93', '1.07', '1', '0.02', '0.02', '12000', '10', '1.5'),
}
_SCAN_PARAMETER_LABELS = {
    'energy': ('En1:', 'En2:', 'Ratio:', 'Gamma1:', 'Gamma2:',
               'Intensity:', 'BG:', 'Temp:'),
    'q': ('Q1:', 'Q2:', 'Ratio:', 'FWHM1:', 'FWHM2:',
          'Intensity:', 'BG:', 'Temp:'),
}
_CONFIG_LINE_FIELDS = (
    'energy_input', 'mono_select_input', 'mono_mosaic_input',
    'mono_vmosaic_input', 'mono_height_input', 'mono_width_input',
    'mono_depth_input', 'mono_hcurve_input', 'mono_vcurve_input',
    'ana_select_input', 'ana_mosaic_input', 'ana_vmosaic_input',
    'ana_height_input', 'ana_width_input', 'ana_depth_input',
    'ana_hcurve_input', 'ana_vcurve_input', 'abc_input', 'abg_input',
    'sample_mosaic_input', 'sample_vmosaic_input', 'sample_height_input',
    'sample_width_input', 'sample_depth_input', 'sample_u_input',
    'sample_v_input', 'hcols_input', 'vcols_input', 'arms_input',
    'guide_height_input', 'guide_width_input', 'detector_height_input',
    'detector_width_input', 'q_input', 'w_input', 'mag_ion',
)
_CONFIG_COMBO_FIELDS = (
    'instr_dropdown', 'energy_dropdown', 'mono_select_dropdown',
    'ana_select_dropdown', 'mono_dir_select', 'ana_dir_select',
    'sample_dir_select', 'fixed_kikf_dropdown', 'method_dropdown',
    'moncor_dropdown', 'sample_shape_dropdown',
)
_CONFIG_CHECK_FIELDS = ('chkMagFF',)
_CONFIG_FORMAT = 'inspy-conv-instrument-config'
_CONFIG_VERSION = 1


def _detect_scan_axes(h, k, l, w):
    """Return coordinates varying beyond expected scan-motor tolerances."""
    varying_axes = []
    for axis, values, tolerance in (
            ('QH', h, 0.005), ('QK', k, 0.005), ('QL', l, 0.005),
            ('W', w, 0.05)):
        values = np.asarray(values, dtype=float)
        if values.size and np.all(np.isfinite(values)):
            if float(np.max(values) - np.min(values)) > tolerance:
                varying_axes.append(axis)
    return varying_axes

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=261, height=201, dpi=100, qslice='QxQy'):
        self.fig = Figure(figsize=(width, height), dpi=dpi, edgecolor='k')
        self.fig.patch.set_facecolor('#ECE8E4')
        self.fig.subplots_adjust(bottom=0.25, left=0.25)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_position([0.25,0.25,0.73,0.73])

        #self.compute_initial_figure(self.axes, qslice, projections, u, v)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self, qslice, projections, u, v):
        pass


class MyDataCanvas(FigureCanvas):
    def __init__(self, parent=None, width=261, height=201, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi, edgecolor='k')
        self.fig.patch.set_facecolor('#ECE8E4')
        self.fig.subplots_adjust(bottom=0.25, left=0.25)
        
        self.axes = self.fig.add_subplot(111)
        self.axes.set_position([0.13,0.17,0.81,0.81])
        self.axes.set_xlabel('Energy [meV]', fontsize=10)
        self.axes.set_ylabel('Intensity [a.u]', fontsize=10)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)



class MyStaticMplCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        super(MyStaticMplCanvas, self).__init__(*args, **kwargs)

    def compute_initial_figure(self, axis, qslice, projections, u, v):
        self.plot_slice(axis, qslice, projections, u, v)


class MainWindow(QMainWindow):
    r"""Main Window for energy- and Q-scan fitting.
    """
    def closeEvent(self, event):
        QApplication.quit()
        
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ui', 'ResConFitEScan.ui'), self)
        self.resize(self.width(), self.height() + 50)
        self.setWindowTitle("Resolution Convolution Fitting")

        self.qxqyplot = QVBoxLayout(self.qx_qy_plot_widget)
        self.qxwplot  = QVBoxLayout(self.qx_w_plot_widget)
        self.qywplot  = QVBoxLayout(self.qy_w_plot_widget)
        self.dataplot = QVBoxLayout(self.fit_data_plot)

        self.text_output.setFontPointSize(6)
        self.text_output.setLineWrapMode(QTextEdit.NoWrap)

        self.dir_dict    = {'Clockwise': 1, 'Counter-Clockwise': -1}
        self.infin_dict  = {'ki': 1, 'kf': -1}

        self.method_dict = {'Cooper-Nathans': 0, 'Popovici': 1}
        self.moncor_dict = {'On': 1, 'Off': 0}
        
        self.param       = np.array([1.0,1,1,1,1,1,1,1.5])
        self.param_fixed = np.array([1,1,1,1,0,1,1,1])
        self.data        = []
        self.scan_mode   = 'energy'
        self.scan_axis   = None
        self.data_start  = 0
        self.data_end    = 0
        self.dplot       = None
        self.filepath    = None
        self.filename    = None
        self._last_fit_export = None
        self._scan_parameters = {
            'energy': tuple(getattr(self, name).text() for name in _PARAMETER_FIELDS),
            'q': _SCAN_PARAMETER_DEFAULTS['q'],
        }
        self._scan_masks = {
            'energy': self.fit_param_fixed.text(),
            'q': '1 1 1 1 1 1 1 0',
        }
        self._set_scan_mode('energy')
        
        self.load_instrument()
        self.load_signals()
        

    def _set_scan_parameter_labels(self, scan_mode):
        labels = _SCAN_PARAMETER_LABELS[scan_mode]
        for object_name, text in zip(
                ('label', 'label_3', 'label_4', 'label_5', 'label_6',
                 'label_7', 'label_9', 'label_10'), labels):
            getattr(self, object_name).setText(text)

    def _set_scan_mode(self, scan_mode, scan_axis=None):
        if scan_mode not in _SCAN_PARAMETER_DEFAULTS:
            raise ValueError("Unknown scan mode: {0}".format(scan_mode))

        if scan_mode != self.scan_mode:
            self._scan_parameters[self.scan_mode] = tuple(
                getattr(self, name).text() for name in _PARAMETER_FIELDS)
            self._scan_masks[self.scan_mode] = self.fit_param_fixed.text()
            for name, value in zip(
                    _PARAMETER_FIELDS, self._scan_parameters[scan_mode]):
                getattr(self, name).setText(value)
            self.fit_param_fixed.setText(self._scan_masks[scan_mode])
            self.scan_mode = scan_mode
            self._set_scan_parameter_labels(scan_mode)

        self.scan_axis = scan_axis
        mode_label = 'Q scan' if scan_mode == 'q' else 'Energy scan'
        self.label_14.setText('Resolution Convolution Fitting: ' + mode_label)

    def _sqw_for_scan_axis(self, h, k, l, w, params):
        coordinates = {'QH': h, 'QK': k, 'QL': l}
        if self.scan_axis not in coordinates:
            raise ValueError("Select a valid Q-scan axis before fitting")
        return SqwQScanTwoPeaks(
            coordinates[self.scan_axis], k, l, w, params)

    def load_instrument(self):
        try:
            curvature = (
                _parse_curvature_radius(self.mono_hcurve_input.text(), 'Monochromator horizontal'),
                _parse_curvature_radius(self.mono_vcurve_input.text(), 'Monochromator vertical'),
                _parse_curvature_radius(self.ana_hcurve_input.text(), 'Analyzer horizontal'),
                _parse_curvature_radius(self.ana_vcurve_input.text(), 'Analyzer vertical'),
            )
            fixed_mask = _parse_param_fixed(self.fit_param_fixed.text(), 8)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid fit settings", str(exc))
            return False

        self.param       = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.param_fixed = np.asarray(fixed_mask)
        self.instrument = TripleAxisSpectr(efixed=5)    
            
    
        self.instrument.sample.a, self.instrument.sample.b, self.instrument.sample.c = [float(i) for i in
                                                                                        self.abc_input.text().split(',')]
        self.instrument.sample.alpha, self.instrument.sample.beta, self.instrument.sample.gamma = [float(i) for i in
                                                                                                   self.abg_input.text().split(',')]
        #Deng: the following statement has been moved from __init__ to here, in order to run the calculation after modifying energy
        self.edrop_dict = {'energy (meV)': float(self.energy_input.text()),
                           'wavelength (A)': Energy(wavelength=float(self.energy_input.text())).energy,
                           'wave vector (A-1)': Energy(wavevector=float(self.energy_input.text())).energy}
        self.instrument.sample.height      = float(self.sample_height_input.text())
        self.instrument.sample.width       = float(self.sample_width_input.text())
        self.instrument.sample.depth       = float(self.sample_depth_input.text())
        self.instrument.sample.u           =[float(i) for i in self.sample_u_input.text().split(',')]
        self.instrument.sample.v           =[float(i) for i in self.sample_v_input.text().split(',')]
        self.instrument.sample.dir         = self.dir_dict[self.sample_dir_select.currentText()]
        self.instrument.sample.mosaic      = float(self.sample_mosaic_input.text())
        self.instrument.sample.vmosaic     = float(self.sample_vmosaic_input.text())
        self.instrument.sample.shape_type  = self.sample_shape_dropdown.currentText().lower()
        

        if self.mono_select_dropdown.currentText() == 'Custom':
            self.instrument.mono.tau = 2 * np.pi / float(self.mono_select_input.text())
        else:
            self.instrument.mono.tau = get_tau(self.mono_select_dropdown.currentText())
            self.mono_select_input.setText(
                '{0:.3f}'.format(2. * np.pi / get_tau(self.mono_select_dropdown.currentText())))
        self.instrument.mono.mosaic  = float(self.mono_mosaic_input.text())
        self.instrument.mono.vmosaic = float(self.mono_vmosaic_input.text())
        self.instrument.mono.dir     = self.dir_dict[self.mono_dir_select.currentText()]
        self.instrument.mono.height  = float(self.mono_height_input.text())
        self.instrument.mono.width   = float(self.mono_width_input.text())
        self.instrument.mono.depth   = float(self.mono_depth_input.text())

        if self.ana_select_dropdown.currentText() == 'Custom':
            self.instrument.ana.tau  = 2 * np.pi / float(self.ana_select_input.text())
        else:
            self.instrument.ana.tau  = get_tau(self.ana_select_dropdown.currentText())
            self.ana_select_input.setText('{0:.3f}'.format(2. * np.pi / get_tau(self.ana_select_dropdown.currentText())))
        self.instrument.ana.mosaic   = float(self.ana_mosaic_input.text())
        self.instrument.ana.vmosaic  = float(self.ana_vmosaic_input.text())
        self.instrument.ana.dir      = self.dir_dict[self.ana_dir_select.currentText()]
        self.instrument.ana.height   = float(self.ana_height_input.text())
        self.instrument.ana.width    = float(self.ana_width_input.text())
        self.instrument.ana.depth    = float(self.ana_depth_input.text())
        self.instrument.efixed       = self.edrop_dict[self.energy_dropdown.currentText()]
        self.instrument.infin        = self.infin_dict[self.fixed_kikf_dropdown.currentText()]
        self.instrument.hcol         = [float(i) for i in self.hcols_input.text().split(',')]
        self.instrument.vcol         = [float(i) for i in self.vcols_input.text().split(',')]
        self.instrument.arms         = [float(i) for i in self.arms_input.text().split(',')]

        self.instrument.guide.height = float(self.guide_height_input.text())
        self.instrument.guide.width  = float(self.guide_width_input.text())

        self.instrument.detector.height = float(self.detector_height_input.text())
        self.instrument.detector.width  = float(self.detector_width_input.text())

        self.instrument.moncor = self.moncor_dict[self.moncor_dropdown.currentText()]
        self.instrument.method = self.method_dict[self.method_dropdown.currentText()]

        (self.instrument.mono.rh, self.instrument.mono.rv,
         self.instrument.ana.rh, self.instrument.ana.rv) = curvature

        if self.q_input.text() != 'None':
            tempstr=self.q_input.text()
            self.q  =  [float(x) for x in tempstr.split(',')]
            
        if self.w_input.text() != 'None':
            tempstr=self.w_input.text()
            self.w  =  [float(x) for x in tempstr.split(',')]
       
            
        self.mag_on =self.chkMagFF.isChecked()
        if self.mag_on:
            self.mag_ion.setEnabled(True)
            self.mag_form_factor.setEnabled(True)
            self.mag_ion_name = self.mag_ion.text()
            self.ffactor = SelFormFactor(self.mag_ion_name)
            if self.ffactor is None:
                self.ffactor = SelFormFactor("Mn2")
                print("The given magnetic ion was not found. Instead, Mn2 is used.")

            AA=self.ffactor["AA"]
            aa=self.ffactor["aa"]
            BB=self.ffactor["BB"]
            bb=self.ffactor["bb"]
            CC=self.ffactor["CC"]
            cc=self.ffactor["cc"]
            DD=self.ffactor["DD"]

            self.mag_form_factor.setText( "{}  {}  {}  {}  {}  {}  {}".format(AA, aa, BB, bb, CC, cc, DD))
        else:
            self.mag_ion_name = "NONE"
            self.ffactor = None
            self.mag_ion.setEnabled(False)
            self.mag_form_factor.setText( "------------------------")
            self.mag_form_factor.setEnabled(False)
            print("No magnetic form factor is set.")

        
        [length, temph,tempk, templ, tempW] = _cleanargs(self.q[0],self.q[1],self.q[2],self.w)
        #print(temph, tempk, templ)
        #print(tempW)
        self.hkle= [temph, tempk, templ, tempW]
        #print(self.hkle)

        #self.instrument.calc_resolution(self.q)
        #self.instrument.calc_projections(self.q)
        
        self.param = np.asarray([
            float(getattr(self, name).text()) for name in _PARAMETER_FIELDS
        ])
        self.data_start=int(self.fit_data_start.text())
        self.data_end=int(self.fit_data_end.text())
        self.str_param_fixed=self.fit_param_fixed.text()
        self.param_fixed = np.asarray(fixed_mask)

        # render/fit path (live -- this dialog ships; formerly marked TEST CODE)
        try:
            self.clearLayout(self.qxqyplot)
            self.clearLayout(self.qxwplot)
            self.clearLayout(self.qywplot)
            #self.clearLayout(self.dataplot)
        except Exception:
            pass

        qxqy = MyStaticMplCanvas(self.qx_qy_plot_widget, width=201, height=201, dpi=80, qslice='QxQy')
        self.qxqyplot.addWidget(qxqy)
        self.instrument.ResolutionPlotProj(ax=qxqy.axes, qslice='QxQy',hkle=self.hkle)

        qxw = MyStaticMplCanvas(self.qx_w_plot_widget,   width=201, height=201, dpi=80, qslice='QxW')
        self.qxwplot.addWidget(qxw)
        self.instrument.ResolutionPlotProj(ax=qxw.axes,  qslice='QxE',hkle=self.hkle)

        qyw = MyStaticMplCanvas(self.qy_w_plot_widget,   width=201, height=201, dpi=80, qslice='QyW')
        self.qywplot.addWidget(qyw)
        self.instrument.ResolutionPlotProj(ax=qyw.axes,  qslice='QyE',hkle=self.hkle)
        
        if self.dplot is None:
            self.dplot = MyDataCanvas(self.fit_data_plot, width=201, height=201, dpi=80)
            self.dataplot.addWidget(self.dplot)
        else:
            self.dplot.fig.canvas.draw_idle()
        
        self.text_output.setText(self.instrument.description_string)  #self.instrument.description_string
        return True


    def instrument_changed(self):
        
        if self.instr_dropdown.currentText() == 'SIKA':
            #20261010: H1 Fix: preset energy is in meV -> pin the unit dropdown before load_instrument()
            self.energy_dropdown.setCurrentText('energy (meV)')
            self.energy_input.setText(str(5))
            self.arms_input.setText("230, 203, 179, 39.5, 106")
        elif self.instr_dropdown.currentText() == 'TAIPAN':
            #20261010: H1 Fix: preset energy is in meV -> pin the unit dropdown before load_instrument()
            self.energy_dropdown.setCurrentText('energy (meV)')
            self.energy_input.setText(str(14.87))
            self.arms_input.setText("202, 196, 179, 65, 106")
        else:
            pass
        
        self.load_instrument()

    def load_signals(self):
        self.instr_dropdown.currentIndexChanged.connect(self.instrument_changed)
        self.method_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.mono_dir_select.currentIndexChanged.connect(self.load_instrument)
        self.sample_dir_select.currentIndexChanged.connect(self.load_instrument)
        self.ana_dir_select.currentIndexChanged.connect(self.load_instrument)
        self.mono_select_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.ana_select_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.moncor_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.fixed_kikf_dropdown.currentIndexChanged.connect(self.load_instrument)

        self.energy_input.editingFinished.connect(self.load_instrument)
        self.mono_select_input.editingFinished.connect(self.load_instrument)
        self.mono_mosaic_input.editingFinished.connect(self.load_instrument)
        self.mono_vmosaic_input.editingFinished.connect(self.load_instrument)
        self.mono_height_input.editingFinished.connect(self.load_instrument)
        self.mono_width_input.editingFinished.connect(self.load_instrument)
        self.mono_depth_input.editingFinished.connect(self.load_instrument)
        self.mono_hcurve_input.editingFinished.connect(self.load_instrument)
        self.mono_vcurve_input.editingFinished.connect(self.load_instrument)

        self.ana_select_input.editingFinished.connect(self.load_instrument)
        self.ana_mosaic_input.editingFinished.connect(self.load_instrument)
        self.ana_vmosaic_input.editingFinished.connect(self.load_instrument)
        self.ana_height_input.editingFinished.connect(self.load_instrument)
        self.ana_width_input.editingFinished.connect(self.load_instrument)
        self.ana_depth_input.editingFinished.connect(self.load_instrument)
        self.ana_hcurve_input.editingFinished.connect(self.load_instrument)
        self.ana_vcurve_input.editingFinished.connect(self.load_instrument)

        self.abc_input.editingFinished.connect(self.load_instrument)
        self.abg_input.editingFinished.connect(self.load_instrument)
        self.sample_mosaic_input.editingFinished.connect(self.load_instrument)
        self.sample_vmosaic_input.editingFinished.connect(self.load_instrument)
        self.sample_height_input.editingFinished.connect(self.load_instrument)
        self.sample_width_input.editingFinished.connect(self.load_instrument)
        self.sample_depth_input.editingFinished.connect(self.load_instrument)
        self.sample_u_input.editingFinished.connect(self.load_instrument)
        self.sample_v_input.editingFinished.connect(self.load_instrument)
        self.sample_shape_dropdown.currentIndexChanged.connect(self.load_instrument)

        self.hcols_input.editingFinished.connect(self.load_instrument)
        self.vcols_input.editingFinished.connect(self.load_instrument)
        self.arms_input.editingFinished.connect(self.load_instrument)

        self.guide_height_input.editingFinished.connect(self.load_instrument)
        self.guide_width_input.editingFinished.connect(self.load_instrument)
        self.detector_height_input.editingFinished.connect(self.load_instrument)
        self.detector_width_input.editingFinished.connect(self.load_instrument)
        self.energy_input.returnPressed.connect(self.load_instrument)
        
        #deng: add the following statement in order to recalc resolution when change q and w
        self.q_input.editingFinished.connect(self.load_instrument)
        self.w_input.editingFinished.connect(self.load_instrument)
        self.fit_param_en1.editingFinished.connect(self.load_instrument)
        self.fit_param_en2.editingFinished.connect(self.load_instrument)
        self.fit_param_ratio.editingFinished.connect(self.load_instrument)
        self.fit_param_gamma1.editingFinished.connect(self.load_instrument)
        self.fit_param_gamma2.editingFinished.connect(self.load_instrument)
        self.fit_param_int.editingFinished.connect(self.load_instrument)
        self.fit_param_bg.editingFinished.connect(self.load_instrument)
        self.fit_param_temp.editingFinished.connect(self.load_instrument)
        self.fit_param_fixed.editingFinished.connect(self.load_instrument)
        self.fit_data_start.editingFinished.connect(self.load_instrument)
        self.fit_data_end.editingFinished.connect(self.load_instrument)
        self.fit_btn_init.clicked.connect(self.initData)
        self.fit_btn_fit.clicked.connect(self.fitData)
        self.fit_opendatafile.clicked.connect(self.loadData)
        self.export_config_button.clicked.connect(self.exportInstrumentConfig)
        self.import_config_button.clicked.connect(self.importInstrumentConfig)
        self.export_fit_csv_button.clicked.connect(self.exportFitCsv)
        self.chkMagFF.stateChanged.connect(self.load_instrument)
        self.mag_ion.editingFinished.connect(self.load_instrument)
        self.mag_form_factor.editingFinished.connect(self.load_instrument)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
    
    def loadData(self):
        fname = QFileDialog.getOpenFileName(
            self, "Choose data file", "",
            "Text data (*.txt *.dat *.hklw *.csv);;HKLW data (*.hklw);;All files (*.*)")
        if fname[0]:
            dialog = HKLWImportDialog(fname[0], self)
            if dialog.exec_() != QDialog.Accepted:
                return
            loaded = dialog.result_data
            if loaded is None:
                return

            H, K, L, W, Iobs, _ = loaded
            scan_axes = _detect_scan_axes(H, K, L, W)
            if not scan_axes:
                QMessageBox.warning(
                    self, "Unable to identify scan",
                    "No scan axis exceeds the detection tolerances "
                    "(0.005 r.l.u. for H/K/L and 0.05 meV for W).")
                return

            if len(scan_axes) > 1:
                QMessageBox.warning(
                    self, "Multiple scan axes detected",
                    "More than one coordinate varies beyond the motor "
                    "tolerance. Choose the coordinate that represents the "
                    "intended scan axis.")
                scan_axis, accepted = QInputDialog.getItem(
                    self, "Choose scan axis",
                    "Detected varying coordinates: {0}".format(
                        ", ".join(scan_axes)),
                    scan_axes, 0, False)
                if not accepted:
                    return
            else:
                scan_axis = scan_axes[0]

            scan_mode = 'energy' if scan_axis == 'W' else 'q'
            scan_axis = None if scan_mode == 'energy' else scan_axis
            self._set_scan_mode(scan_mode, scan_axis)
            self.filepath, self.filename = os.path.split(fname[0])
            print('Path:'+self.filepath+';Name:'+self.filename)
            self.fit_filepath.setText(fname[0])

            self.data = np.asarray(loaded)
            self._last_fit_export = None
            self.data_start = 0
            self.data_end=self.data.shape[1]
            self.fit_data_start.setText(str(self.data_start))
            self.fit_data_end.setText(str(self.data_end))
            if self.dplot is None:
                self.dplot = MyDataCanvas(self.fit_data_plot, width=201, height=201, dpi=80)
                self.dataplot.addWidget(self.dplot)
            self.dplot.axes.clear()
            x_values = W if scan_mode == 'energy' else {
                'QH': H, 'QK': K, 'QL': L}[scan_axis]
            x_label = 'Energy [meV]' if scan_mode == 'energy' else scan_axis + ' [rlu]'
            self.dplot.axes.plot(x_values, Iobs, "bo")
            self.dplot.axes.set_xlabel(x_label, fontsize=8)
            self.dplot.axes.set_ylabel('Intensity [a.u]', fontsize=8)
            self.dplot.fig.canvas.draw_idle()
        
    
    def initData(self):
        if len(self.data) == 0:
            QMessageBox.warning(self, "No data", "Load a scan data file before initializing the fit.")
            return
        if not self.load_instrument():
            return
        [H, K, L, W, Iobs, dIobs] = self.data[:, self.data_start:self.data_end]
        if self.chkMagFF.isChecked(): 
            AA=self.ffactor["AA"]
            aa=self.ffactor["aa"]
            BB=self.ffactor["BB"]
            bb=self.ffactor["bb"]
            CC=self.ffactor["CC"]
            cc=self.ffactor["cc"]
            DD=self.ffactor["DD"]
            self.param       = np.append(self.param, np.array([AA, aa, BB, bb, CC, cc, DD]))
            self.param_fixed = np.append(self.param_fixed, np.array([ 0,  0,  0,  0,  0,  0,  0]))

        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)
        model = SqwDemo if self.scan_mode == 'energy' else self._sqw_for_scan_axis
        sim_init = self.instrument.ResConv(
            sqw=model, pref=PrefDemoFF, nargout=2,
            hkle=[newH,newK,newL,newW], METHOD='fix',
            ACCURACY=[5,5] if self.scan_mode == 'energy' else None,
            p=self.param)

        self.dplot.axes.clear()
        if self.scan_mode == 'energy':
            x_data, x_fit = W, newW
            x_label = 'Energy [meV]'
        else:
            hkl = {'QH': H, 'QK': K, 'QL': L}
            new_hkl = {'QH': newH, 'QK': newK, 'QL': newL}
            x_data = hkl[self.scan_axis]
            x_fit = new_hkl[self.scan_axis]
            x_label = self.scan_axis + ' [rlu]'
        self.dplot.axes.plot(x_data, Iobs, "bo", x_fit, sim_init, "g-")
        self.dplot.axes.set_xlabel(x_label, fontsize=8)
        self.dplot.fig.canvas.draw_idle()

    
    def fitData(self):
        if len(self.data) == 0:
            QMessageBox.warning(self, "No data", "Load a scan data file before fitting.")
            return
        if not self.load_instrument():
            return
        H, K, L, W, Iobs, dIobs = self.data[:, self.data_start:self.data_end]

        model = SqwDemo if self.scan_mode == 'energy' else self._sqw_for_scan_axis
        param_initial = np.array(self.param, copy=True)
        param_fixed = np.array(self.param_fixed, copy=True)
        if self.chkMagFF.isChecked():
            form_factor = [self.ffactor[key] for key in
                           ('AA', 'aa', 'BB', 'bb', 'CC', 'cc', 'DD')]
            param_initial = np.append(param_initial, form_factor)
            param_fixed = np.append(param_fixed, np.zeros(7, dtype=int))

        fitter = UltraFastFitConv(
            self.instrument, model, PrefDemoFF, [H, K, L, W], Iobs, dIobs)
        result = fitter.fit_ultrafast(
            param_initial=param_initial, param_fixed_mask=param_fixed,
            maxfev=200, use_analytical_jacobian=True,
            early_stopping=True, verbose=True)
        final_params = result['params']
        param_errors = result['param_errors']

        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)
        final = self.instrument.ResConv(
            sqw=model, pref=PrefDemoFF, nargout=2,
            hkle=[newH, newK, newL, newW], METHOD='fix',
            ACCURACY=None, p=final_params)

        if self.scan_mode == 'energy':
            x_data, x_fit = W, newW
            x_label = 'Energy [meV]'
        else:
            new_hkl = {'QH': newH, 'QK': newK, 'QL': newL}
            x_fit = new_hkl[self.scan_axis]
            x_label = self.scan_axis + ' [rlu]'
        self.dplot.axes.plot(x_fit, final, "r-")
        self.dplot.axes.set_xlabel(x_label, fontsize=8)
        self.dplot.fig.canvas.draw_idle()

        if self.scan_mode == 'energy':
            par_output = (
                "The current energy scan is fitted by using\n"
                "the SqwDemo S(Q, w) function with UltraFastFitConv.\n"
                "The fitted results are:\n")
        else:
            par_output = (
                "The current {0} scan is fitted by using\n"
                "the SqwQScanTwoPeaks S(Q, w) function with UltraFastFitConv.\n".format(self.scan_axis))
        par_output += "The fitted parameters ({0} scan):\n".format(
            self.scan_axis if self.scan_mode == 'q' else 'energy')
        par_output += "{0}1  :\t{1:8f}  \t{2:8f}\n".format(
            'En' if self.scan_mode == 'energy' else self.scan_axis,
            final_params[0], param_errors[0])
        par_output += "{0}2  :\t{1:8f}  \t{2:8f}\n".format(
            'En' if self.scan_mode == 'energy' else self.scan_axis,
            final_params[1], param_errors[1])

        if self.scan_mode == 'energy':
            intensity1_error = np.sqrt(
                (final_params[5] * param_errors[2]) ** 2
                + (final_params[2] * param_errors[5]) ** 2)
            par_output += "Int1 :\t{0:8f}  \t{1:8f}\n".format(
                final_params[2] * final_params[5], intensity1_error)
            par_output += "Int2 :\t{0:8f}  \t{1:8f}\n".format(
                final_params[5], param_errors[5])
            width_label = 'HWHM'
        else:
            intensity2_error = np.sqrt(
                (final_params[5] * param_errors[2]) ** 2
                + (final_params[2] * param_errors[5]) ** 2)
            par_output += "Int1 :\t{0:8f}  \t{1:8f}\n".format(
                final_params[5], param_errors[5])
            par_output += "Int2 :\t{0:8f}  \t{1:8f}\n".format(
                final_params[2] * final_params[5], intensity2_error)
            width_label = 'FWHM'
        par_output += "{0}1:\t{1:8f}  \t{2:8f}\n".format(
            width_label, final_params[3], param_errors[3])
        par_output += "{0}2:\t{1:8f}  \t{2:8f}\n".format(
            width_label, final_params[4], param_errors[4])
        par_output += "bg   :\t{0:8f}  \t{1:8f}\n".format(
            final_params[6], param_errors[6])
        par_output += "temp :\t{0:8f}  \t{1:8f}\n".format(
            final_params[7], param_errors[7])

        self._last_fit_export = {
            'experimental': np.array(
                [H, K, L, W, Iobs], dtype=float, copy=True),
            'fitted': np.array(
                [newH, newK, newL, newW, final], dtype=float, copy=True),
            'parameters': par_output,
        }

        dat_output = "The origin data:\n    H\t    K\t    L\t  W\t  Iobs\t dIobs\t\n"
        original_data = np.column_stack([H, K, L, W, Iobs, dIobs])
        fitted_data = np.column_stack([newH, newK, newL, newW, final])
        for row in original_data:
            dat_output += "\n" + "".join(
                "{0:6f}\t".format(value) for value in row)

        dat_output += "\n\nThe fitted data:\n    H\t    K\t    L\t   W\t  Fitted\t\n"
        for row in fitted_data:
            dat_output += "\n" + "".join(
                "{0:6f}\t".format(value) for value in row)

        self.fit_output_text.clear()
        self.fit_output_text.insertPlainText(par_output + dat_output)
        if self.filepath is not None and self.filename is not None:
            stem = os.path.splitext(self.filename)[0]
            with open(os.path.join(self.filepath, stem + '_par.txt'), "w") as par_file:
                par_file.write(par_output)
            with open(os.path.join(self.filepath, stem + '_fit.txt'), "w") as fit_file:
                fit_file.write(dat_output)

    def _configuration_values(self):
        controls = {}
        for name in _CONFIG_LINE_FIELDS:
            controls[name] = getattr(self, name).text()
        for name in _CONFIG_COMBO_FIELDS:
            controls[name] = getattr(self, name).currentText()
        for name in _CONFIG_CHECK_FIELDS:
            controls[name] = getattr(self, name).isChecked()
        return {
            'format': _CONFIG_FORMAT,
            'version': _CONFIG_VERSION,
            'controls': controls,
        }

    def _set_configuration_controls(self, controls):
        names = (
            _CONFIG_LINE_FIELDS + _CONFIG_COMBO_FIELDS + _CONFIG_CHECK_FIELDS)
        widgets = {name: getattr(self, name) for name in names}
        signals_blocked = {
            name: widget.blockSignals(True)
            for name, widget in widgets.items()
        }
        try:
            for name in _CONFIG_LINE_FIELDS:
                widgets[name].setText(controls[name])
            for name in _CONFIG_COMBO_FIELDS:
                widgets[name].setCurrentText(controls[name])
            for name in _CONFIG_CHECK_FIELDS:
                widgets[name].setChecked(controls[name])
        finally:
            for name, widget in widgets.items():
                widget.blockSignals(signals_blocked[name])

    def exportInstrumentConfig(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export instrument configuration", "",
            "JSON files (*.json)")
        if not filename:
            return
        if os.path.splitext(filename)[1].lower() != '.json':
            filename += '.json'
        try:
            with open(filename, 'w', encoding='utf-8') as config_file:
                json.dump(
                    self._configuration_values(), config_file, indent=2)
                config_file.write('\n')
        except OSError as exc:
            QMessageBox.warning(
                self, "Export failed",
                "Could not write instrument configuration:\n{0}".format(exc))

    def importInstrumentConfig(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import instrument configuration", "",
            "JSON files (*.json)")
        if not filename:
            return
        try:
            with open(filename, 'r', encoding='utf-8') as config_file:
                config = json.load(config_file)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(
                self, "Import failed",
                "Could not read instrument configuration:\n{0}".format(exc))
            return

        if (not isinstance(config, dict)
                or config.get('format') != _CONFIG_FORMAT
                or type(config.get('version')) is not int
                or config.get('version') != _CONFIG_VERSION
                or not isinstance(config.get('controls'), dict)):
            QMessageBox.warning(
                self, "Import failed",
                "This file is not a supported instrument configuration.")
            return

        controls = config['controls']
        expected_fields = (
            _CONFIG_LINE_FIELDS + _CONFIG_COMBO_FIELDS + _CONFIG_CHECK_FIELDS)
        missing = [name for name in expected_fields if name not in controls]
        if missing:
            QMessageBox.warning(
                self, "Import failed",
                "The configuration is missing required settings: {0}".format(
                    ", ".join(missing)))
            return
        invalid = []
        for name in _CONFIG_LINE_FIELDS:
            if not isinstance(controls[name], str):
                invalid.append(name)
        for name in _CONFIG_COMBO_FIELDS:
            combo = getattr(self, name)
            if (not isinstance(controls[name], str)
                    or combo.findText(controls[name]) < 0):
                invalid.append(name)
        for name in _CONFIG_CHECK_FIELDS:
            if not isinstance(controls[name], bool):
                invalid.append(name)
        if invalid:
            QMessageBox.warning(
                self, "Import failed",
                "The configuration contains invalid settings: {0}".format(
                    ", ".join(invalid)))
            return

        previous = self._configuration_values()['controls']
        self._set_configuration_controls(controls)

        try:
            loaded = self.load_instrument()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            loaded = False
            error = str(exc)
        else:
            error = "One or more settings could not be applied."
        if not loaded:
            self._set_configuration_controls(previous)
            self.load_instrument()
            QMessageBox.warning(
                self, "Import failed",
                "The configuration was not applied: {0}".format(error))
        else:
            self._last_fit_export = None

    def exportFitCsv(self):
        if self._last_fit_export is None:
            QMessageBox.warning(
                self, "No fitted results",
                "Fit the loaded data before exporting fitted results.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export fitted results", "", "CSV files (*.csv)")
        if not filename:
            return
        if os.path.splitext(filename)[1].lower() != '.csv':
            filename += '.csv'

        experimental = self._last_fit_export['experimental']
        fitted = self._last_fit_export['fitted']
        comments = self._last_fit_export['parameters'].splitlines()
        headers = [
            'H', 'K', 'L', 'W', 'Intensity',
            'newH', 'newK', 'newL', 'newW', 'fitted_Intensity',
        ]
        rows = []
        for index in range(max(experimental.shape[1], fitted.shape[1])):
            measured_row = (
                experimental[:, index].tolist()
                if index < experimental.shape[1] else [''] * 5)
            fitted_row = (
                fitted[:, index].tolist()
                if index < fitted.shape[1] else [''] * 5)
            rows.append([
                "{0:.10g}".format(value) if value != '' else ''
                for value in measured_row + fitted_row
            ])
        widths = [
            max(len(headers[column]), *(len(row[column]) for row in rows))
            for column in range(len(headers))
        ]
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
                for line in comments:
                    csv_file.write("# {0}\n".format(line))
                csv_file.write(", ".join(
                    header.ljust(width)
                    for header, width in zip(headers, widths)) + "\n")
                for row in rows:
                    csv_file.write(", ".join(
                        value.rjust(width)
                        for value, width in zip(row, widths)) + "\n")
        except OSError as exc:
            QMessageBox.warning(
                self, "Export failed",
                "Could not write fitted results:\n{0}".format(exc))

        
def main():        

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())



            
if __name__ == "__main__" :
    main()
    