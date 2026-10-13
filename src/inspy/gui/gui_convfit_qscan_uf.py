# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from matplotlib.backends.backend_qtagg  import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import uic
from qtpy.QtWidgets import (QApplication, QFileDialog, QMainWindow, QSizePolicy, QVBoxLayout,QTextEdit)
from lmfit import Parameters, Minimizer, fit_report, report_fit


from ..energy import Energy
from ..instrument.tools import get_tau, _cleanargs, _star, _modvec
from ..instrument.tas_spectr import TripleAxisSpectr
from ..insfit import FitConv, UltraFastFitConv
from .tools import SqwQScanTwoPeaks, PrefDemoFF, angle2, SelFormFactor



class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=261, height=201, dpi=100, qslice='QxQy'):
        self.fig = Figure(figsize=(width, height), dpi=dpi, edgecolor='k')
        self.fig.patch.set_facecolor('#F0F0F0')
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
        self.fig.patch.set_facecolor('#F0F0F0')
        self.fig.subplots_adjust(bottom=0.25, left=0.25)
        

        self.axes = self.fig.add_subplot(111)
        self.axes.set_position([0.17,0.17,0.81,0.81])
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
    r"""Main Window of the Q-scan fit dialog

    """
    def closeEvent(self, event):
        QApplication.quit()
        
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ui','ResConFitQScanFF.ui'), self)

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
        
        self.param       = np.array([1.0,1, 1, 1, 1, 1, 1, 1])
        self.param_fixed = np.array([1,  1, 1, 1, 1, 1, 1, 0])
        self.data        = []
        self.scanAxis      = ""  # set to "QH"/"QK"/"QL" at load time
        self.data_start  = 0
        self.data_end    = 0
        
        self.dplot=None

        self.load_instrument()
        self.load_signals()
        

    def load_instrument(self):
        self.param       = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.param_fixed = np.array([1,   1, 1, 1, 1, 1, 1, 0])

        self.instrument = TripleAxisSpectr(efixed=5)

        self.instrument.sample.a, self.instrument.sample.b, self.instrument.sample.c = [float(i) for i in
                                                                                        self.abc_input.text().split(',')]
        self.instrument.sample.alpha, self.instrument.sample.beta, self.instrument.sample.gamma = [float(i) for i in
                                                                                                   self.abg_input.text().split(',')]
        #Deng: the following statement has been moved from __init__ to here, in order to run the calculation after modifying energy
        self.edrop_dict = {'energy (meV)': float(self.energy_input.text()),
                           'wavelength (A)': Energy(wavelength=float(self.energy_input.text())).energy,
                           'wave vector (A-1)': Energy(wavevector=float(self.energy_input.text())).energy}
        self.instrument.sample.height     =    float(self.sample_height_input.text())
        self.instrument.sample.width      =    float(self.sample_width_input.text())
        self.instrument.sample.depth      =    float(self.sample_depth_input.text())
        self.instrument.sample.u          =   [float(i) for i in self.sample_u_input.text().split(',')]
        self.instrument.sample.v          =   [float(i) for i in self.sample_v_input.text().split(',')]
        self.instrument.sample.dir        =   self.dir_dict[self.sample_dir_select.currentText()]
        self.instrument.sample.mosaic     =    float(self.sample_mosaic_input.text())
        self.instrument.sample.vmosaic    =    float(self.sample_vmosaic_input.text())
        self.instrument.sample.shape_type =   self.sample_shape_dropdown.currentText().lower()
        
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

        self.instrument.moncor       = self.moncor_dict[self.moncor_dropdown.currentText()]
        self.instrument.method       = self.method_dict[self.method_dropdown.currentText()]

        if self.mono_hcurve_input.text() != 'None':
            self.instrument.mono.rh       = float(self.mono_hcurve_input.text())
        if self.mono_vcurve_input.text() != 'None':
            self.instrument.mono.rv       = float(self.mono_vcurve_input.text())

        if self.ana_hcurve_input.text()  != 'None':
            self.instrument.ana.rh        = float(self.ana_hcurve_input.text())
        if self.ana_vcurve_input.text()  != 'None':
            self.instrument.ana.rv        = float(self.ana_vcurve_input.text())
        self.instrument.description_string = ''

        self.q = [float(i) for i in self.q_input.text().split(',')]
        self.w = [float(i) for i in self.w_input.text().split(',')]
        
        
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
        self.hkle  =  [temph, tempk, templ, tempW]

        #self.instrument.calc_resolution(self.q)
        #self.instrument.calc_projections(self.q)
        
        self.param[0] = float(self.fit_param_p1.text())
        self.param[1] = float(self.fit_param_p2.text())
        self.param[2] = float(self.fit_param_ratio.text())
        self.param[3] = float(self.fit_param_w1.text())
        self.param[4] = float(self.fit_param_w2.text())
        self.param[5] = float(self.fit_param_int.text())
        self.param[6] = float(self.fit_param_bg.text())
        self.param[7] = float(self.fit_param_temp.text())
        
        
        self.data_start   = int(self.fit_data_start.text())
        self.data_end     = int(self.fit_data_end.text())
        self.str_param_fixed   = self.fit_param_fixed.text()
        self.param_fixed  = [int(ii) for ii in self.str_param_fixed.split() if ii.isdigit()]

        # render/fit path (live -- this dialog ships; formerly marked TEST CODE)
        try:
            self.clearLayout(self.qxqyplot)
            self.clearLayout(self.qxwplot)
            self.clearLayout(self.qywplot)
            #self.clearLayout(self.dataplot)
        except Exception:
            pass

        qxqy = MyStaticMplCanvas(self.qx_qy_plot_widget, width=261, height=201, dpi=100, qslice='QxQy')
        self.qxqyplot.addWidget(qxqy)
        self.instrument.ResolutionPlotProj(ax=qxqy.axes, qslice='QxQy',hkle=self.hkle)

        qxw = MyStaticMplCanvas(self.qx_w_plot_widget,   width=261, height=201, dpi=100, qslice='QxW')
        self.qxwplot.addWidget(qxw)
        self.instrument.ResolutionPlotProj(ax=qxw.axes,  qslice='QxE',hkle=self.hkle)

        qyw = MyStaticMplCanvas(self.qy_w_plot_widget,   width=261, height=201, dpi=100, qslice='QyW')
        self.qywplot.addWidget(qyw)
        self.instrument.ResolutionPlotProj(ax=qyw.axes,  qslice='QyE',hkle=self.hkle)
        
        if self.dplot is None:
            self.dplot = MyDataCanvas(self.fit_data_plot,width=261, height=201, dpi=100)
            self.dataplot.addWidget(self.dplot)
        else:
            self.dplot.fig.canvas.draw_idle()

        self.text_output.setText(str(self.instrument.description_string))
        

    def load_signals(self):
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
        self.fit_param_p1.editingFinished.connect(self.load_instrument)
        self.fit_param_p2.editingFinished.connect(self.load_instrument)
        self.fit_param_ratio.editingFinished.connect(self.load_instrument)
        self.fit_param_w1.editingFinished.connect(self.load_instrument)
        self.fit_param_w2.editingFinished.connect(self.load_instrument)
        self.fit_param_int.editingFinished.connect(self.load_instrument)
        self.fit_param_bg.editingFinished.connect(self.load_instrument)
        self.fit_param_temp.editingFinished.connect(self.load_instrument)
        
        self.fit_param_fixed.editingFinished.connect(self.load_instrument)
        self.fit_data_start.editingFinished.connect(self.load_instrument)
        self.fit_data_end.editingFinished.connect(self.load_instrument)
        self.fit_btn_init.pressed.connect(self.initData)
        self.fit_btn_fit.pressed.connect(self.fitData)
        self.fit_opendatafile.pressed.connect(self.loadData)
        self.chkMagFF.stateChanged.connect(self.load_instrument)
        self.mag_ion.editingFinished.connect(self.load_instrument)
        #self.mag_form_factor.editingFinished.connect(self.load_instrument)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            child.widget().deleteLater()
    
    def loadData(self):
        fname = QFileDialog.getOpenFileName(self, "open file", "", "data file(*.hklw)")
        #print(fname)
        if  not ( fname[0] == '') :
            self.fit_filepath.setText(fname[0])
            with open(fname[0], 'r') as f:
                #20261010: M3 Fix: accept both headerless and single-header .hklw files (retry with skiprows=1)
                try:
                    self.data      = np.loadtxt(f, unpack=True)
                except ValueError:
                    f.seek(0)
                    self.data      = np.loadtxt(f, skiprows=1, unpack=True)
            [H, K, L, W, Iobs] = self.data
            dIobs           =   np.sqrt(Iobs)
            self.data       =   np.array([H, K, L, W, Iobs, dIobs])
            self.data_start =   0
            self.data_end   =   self.data.shape[1]
            
            self.fit_data_start.setText(str(self.data_start))
            self.fit_data_end.setText(str(self.data_end))
            dH=np.abs(H[0] -H[-1])
            dK=np.abs(K[0] -K[-1])
            dL=np.abs(L[0] -L[-1])
            if dH > 0.02 :
                self.scanAxis = "QH"
            elif dK > 0.02:
                self.scanAxis = "QK"
            elif dL > 0.02:
                self.scanAxis = "QL"

            hkl_dict = {"QH": H, "QK": K, "QL": L }

            self.dplot.axes.clear()
            self.dplot.axes.plot(hkl_dict[self.scanAxis], Iobs, "bo")
            self.dplot.axes.set_xlabel(self.scanAxis+' [rlu]', fontsize=8)
            self.dplot.axes.set_ylabel('Intensity [a.u]', fontsize=8)
            self.dplot.fig.canvas.draw_idle()
        
    
    def initData(self):
        #20261010: M4 Fix: no data loaded yet -> return instead of TypeError on self.data[:, ...]
        if len(self.data) == 0:
            print('no data loaded')
            return
        self.load_instrument()
        [H, K, L, W, Iobs, dIobs] = self.data[:,self.data_start:self.data_end]
        
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
        newhkl_dict = {"QH": newH, "QK": newK, "QL": newL }
        sim_init  = self.instrument.ResConv(sqw=SqwQScanTwoPeaks, pref=PrefDemoFF, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=None, p=self.param)
        hkl_dict = {"QH": H, "QK": K, "QL": L }
        self.dplot.axes.clear()
        self.dplot.axes.plot(hkl_dict[self.scanAxis], Iobs, "bo", newhkl_dict[self.scanAxis], sim_init, "g-")
        self.dplot.fig.canvas.draw_idle()
        
        
    
    def fitData(self):
        #fit the data using the input parameters:
        #20261010: M4 Fix: no data loaded yet -> return instead of TypeError on self.data[:, ...]
        if len(self.data) == 0:
            print('no data loaded')
            return
        self.load_instrument()
        [H, K, L, W, Iobs, dIobs] = self.data[:,self.data_start:self.data_end]

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

        fitter =    UltraFastFitConv(self.instrument,  SqwQScanTwoPeaks, PrefDemoFF,[H,K,L,W], Iobs, dIobs)
        result = fitter.fit_ultrafast(param_initial=self.param, param_fixed_mask=self.param_fixed,maxfev=200,use_analytical_jacobian=True,early_stopping=True,verbose=True)
        final_params = result['params']
        param_errors = result['param_errors'] 
        chi2_reduced = result['chi2_reduced']
        model_fit = result['model']
        
        
        newH=np.linspace(H[0], H[-1], 101)
        newK=np.linspace(K[0], K[-1], 101)
        newL=np.linspace(L[0], L[-1], 101)
        newW=np.linspace(W[0], W[-1], 101)
        newhkl_dict = {"QH": newH, "QK": newK, "QL": newL }
        final = self.instrument.ResConv(sqw=SqwQScanTwoPeaks, pref=PrefDemoFF, nargout=2, hkle=[newH,newK,newL,newW], METHOD='fix', ACCURACY=None, p=final_params)

        self.dplot.axes.plot(newhkl_dict[self.scanAxis],final, "r-")
        self.dplot.fig.canvas.draw_idle()
        
        par_output="The fitted parameters:\n"
        
        par_output=par_output+self.scanAxis+"1  :\t{0:8f}  \t{1:8f}\n".format(final_params[0],param_errors[0])
        par_output=par_output+self.scanAxis+"2  :\t{0:8f}  \t{1:8f}\n".format(final_params[1],param_errors[1])
        #20261010: L11 Fix + §2.2 #13 resolution: the printed Int1/Int2 values were
        #   SWAPPED relative to the QH1/QH2 labels; now fixed. The swap is confirmed
        #   statically: SqwQScanTwoPeaks' Gaussians are unit-area in H (gui/tools.py:269-271)
        #   and PrefDemoFF applies p[5] as the overall scale (gui/tools.py:233-235), so
        #   peak-1 integral = p5 and peak-2 = ratio*p5. Int1 (under QH1) prints p5 with
        #   its plain error; Int2 prints ratio*p5 with first-order propagation over BOTH
        #   factors (the old Int1 error ignored the ratio-parameter uncertainty).
        _int2_err = np.sqrt((final_params[5]*param_errors[2])**2 + (final_params[2]*param_errors[5])**2)
        par_output=par_output+"Int1 :\t{0:8f}  \t{1:8f}\n".format(final_params[5],param_errors[5])
        par_output=par_output+"Int2 :\t{0:8f}  \t{1:8f}\n".format(final_params[2]*final_params[5], _int2_err)
        #20261010: L11 Fix (comment corrected) + §1.2 label fix applied: in THIS dialog
        #   the model is SqwQScanTwoPeaks (gui/tools.py:269-271), where p[3]/p[4] are
        #   Gaussian widths w in Q: exp(-4*ln2*(H-q)^2/w^2), so w IS the FWHM of the
        #   Q-direction Gaussian (SqwDemo's gamma=HWHM rationale applies only to
        #   the escan dialog, where the ratio multiplies peak 1 instead of peak 2).
        #   The HWHM1/HWHM2 labels had been copied from escan during the L11 pass --
        #   a mislabel we introduced; relabelled back to the pre-L11 FWHM1/FWHM2.
        par_output=par_output+"FWHM1:\t{0:8f}  \t{1:8f}\n".format(final_params[3],param_errors[3])
        par_output=par_output+"FWHM2:\t{0:8f}  \t{1:8f}\n".format(final_params[4],param_errors[4])
        par_output=par_output+"bg   :\t{0:8f}  \t{1:8f}\n".format(final_params[6],param_errors[6])
        par_output=par_output+"temp :\t{0:8f}  \t{1:8f}\n".format(final_params[7],param_errors[7])

        dat_output="The origin data:\n    H\t    K\t    L\t  W\t  Iobs\t dIobs\t \n"
        oldHKLW=np.column_stack([H,K,L,W,Iobs,dIobs])
        newHKLW=np.column_stack([newH,newK,newL,newW,final])
        for row in oldHKLW:
            dat_output=dat_output+"\n"
            for element in row:
                dat_output=dat_output+"{0:6f}\t".format(element)
                
        dat_output=dat_output+"\n\nThe fitted data:\n    H\t    K\t    L\t   W\t  Fitted\t \n"
        for row in newHKLW:
            dat_output=dat_output+"\n"
            for element in row:
                dat_output=dat_output+"{0:6f}\t".format(element)

        self.fit_output_text.clear()
        self.fit_output_text.insertPlainText(par_output+dat_output)


    
        
def main():        

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())



if __name__ == "__main__" :
    main()
    