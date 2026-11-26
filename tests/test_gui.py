# -*- coding: utf-8 -*-
"""GUI construction tests.

These run headless (QT_QPA_PLATFORM=offscreen, matplotlib Agg) and only check that
the windows build, that their instrument loads, and that the widget wiring is
sane. They do not drive the widgets: Qt event loops are not the subject here.
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

# Every test in this module constructs a Qt window; -m "not gui" skips the file.
pytestmark = pytest.mark.gui

from qtpy.QtWidgets import QApplication, QWidget   # noqa: E402

from inspy.gui.gui_convfit_uf import MainWindow as UnifiedFitWindow     # noqa: E402

WINDOWS = [('fit', UnifiedFitWindow)]


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, request):
    """A loaded MainWindow, closed at the end of the test."""
    name, cls = request.param
    win = cls()
    win.load_instrument()
    yield win
    win.close()


def _settle(win):
    """Wait for the background instrument rebuild to finish (review item 180).

    ``load_instrument`` now returns immediately and the resolution is computed on
    a worker thread, so tests must pump the event loop until the result lands.
    """
    runner = getattr(win, '_resolution_runner', None)
    if runner is None:
        return
    deadline = 30_000
    elapsed = 0
    while runner.busy and elapsed < deadline:
        QApplication.processEvents()
        QApplication.sendPostedEvents()
        elapsed += 10
    if runner.busy:
        raise AssertionError('background rebuild did not finish within 30 s')
    # One more pump so the result signal is delivered and the canvases are drawn.
    QApplication.processEvents()
    QApplication.sendPostedEvents()


class TestBackgroundWork:
    """Review items 180 and 181: the slow work must leave the GUI thread."""

    def test_a_worker_exception_reaches_the_gui_thread(self, qapp):
        """An exception on the worker must not vanish (it used to be swallowed)."""
        from inspy.gui.tasks import DebouncedRunner

        seen = []
        runner = DebouncedRunner(work=lambda: 1 / 0,
                                 on_result=seen.append,
                                 on_error=seen.append,
                                 parent=None)
        runner.request()
        deadline, elapsed = 20_000, 0
        while not seen and elapsed < deadline:
            QApplication.processEvents()
            QApplication.sendPostedEvents()
            # The signal is queued to the GUI thread; give the worker time to run.
            time.sleep(0.002)
            elapsed += 2
        assert seen, 'the worker exception was lost'
        assert isinstance(seen[0], ZeroDivisionError)
        assert getattr(seen[0], '_inspy_traceback', None), 'no traceback attached'

    def test_cancel_drops_the_pending_rerun(self, qapp):
        from inspy.gui.tasks import DebouncedRunner

        calls = []
        runner = DebouncedRunner(work=lambda: calls.append(1),
                                 on_result=lambda r: None,
                                 parent=None)
        runner.request()
        runner.cancel()
        runner.request()          # queued behind the first run
        runner.cancel()           # ...and dropped again
        runner.wait(10000)
        for _ in range(50):
            QApplication.processEvents()
            QApplication.sendPostedEvents()
        assert len(calls) == 1, 'expected exactly one run, got %d' % len(calls)


class TestConstruction:
    @pytest.mark.parametrize('window', WINDOWS, indirect=True)
    def test_window_builds_and_loads_an_instrument(self, window):
        _settle(window)
        assert window.instrument is not None
        assert window.instrument.efixed > 0
        assert window.instrument.mono is not None
        assert window.instrument.ana is not None
        # Where the rebuild is asynchronous, the resolution matrices must have been
        # computed by the worker and delivered back (review item 180).
        runner = getattr(window, '_resolution_runner', None)
        if runner is not None:
            assert np.shape(window.instrument.RMS)[1:] == (4, 4)
            assert np.shape(window.instrument.R0) == (np.shape(window.instrument.RMS)[0],)
            assert np.all(np.isfinite(window.instrument.RMS))

    @pytest.mark.parametrize('window', WINDOWS, indirect=True)
    def test_closing_a_window_does_not_quit_the_application(self, window, qapp):
        """closeEvent used to call QApplication.quit(), killing every other window."""
        from qtpy.QtGui import QCloseEvent
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted()
        assert QApplication.instance() is qapp

    def test_qapplication_singleton_is_reused(self, qapp):
        assert QApplication.instance() is qapp


class TestClearLayout:
    def test_widgets_and_non_widget_items_are_both_handled(self, qapp):
        """A spacer has widget() is None; the old code raised AttributeError on it."""
        from qtpy.QtWidgets import QVBoxLayout

        window = UnifiedFitWindow()
        try:
            layout = QVBoxLayout()
            widget = QWidget()
            layout.addWidget(widget)
            layout.addStretch(1)          # a QSpacerItem: no widget(), no deleteLater()
            assert layout.count() == 2

            window.clearLayout(layout)
            assert layout.count() == 0
        finally:
            window.close()

    def test_clearing_an_empty_layout_is_a_no_op(self, qapp):
        from qtpy.QtWidgets import QVBoxLayout

        window = UnifiedFitWindow()
        try:
            layout = QVBoxLayout()
            window.clearLayout(layout)
            assert layout.count() == 0
        finally:
            window.close()


class TestProjectionCanvases:
    @pytest.mark.parametrize('window', WINDOWS, indirect=True)
    def test_replacing_the_canvases_does_not_accumulate_widgets(self, window):
        """clearLayout + rebuild must leave the layout with exactly the new widgets."""
        for _ in range(3):
            window.clearLayout(window.qxqyplot)
            window.clearLayout(window.qxwplot)
            window.clearLayout(window.qywplot)
        assert window.qxqyplot.count() == 0
        assert window.qxwplot.count() == 0
        assert window.qywplot.count() == 0


class TestDataLoading:
    def test_unified_export_controls_are_visible_in_requested_positions(
            self, qapp):
        window = UnifiedFitWindow()
        try:
            assert window.import_config_button.parentWidget().objectName() == (
                'centralwidget')
            assert window.export_config_button.parentWidget().objectName() == (
                'centralwidget')
            assert (window.instr_dropdown.geometry().right()
                    < window.import_config_button.geometry().left())
            assert (window.import_config_button.geometry().right()
                    < window.export_config_button.geometry().left())
            assert window.export_fit_csv_button.parentWidget().objectName() == (
                'groupBox_8')
            assert (window.export_fit_csv_button.geometry().top()
                    >= window.fit_output_text.geometry().bottom())
            assert (window.export_fit_csv_button.geometry().bottom()
                    < window.frame.geometry().top())
        finally:
            window.close()

    def test_unified_scan_detection_uses_absolute_axis_tolerances(self):
        from inspy.gui.gui_convfit_uf import _detect_scan_axes

        assert _detect_scan_axes(
            [1.0, 1.004], [2, 2], [3, 3], [1, 1.049]) == []
        assert _detect_scan_axes(
            [1.0, 1.006], [2.0, 2.005], [3, 3], [1, 1.051]) == [
                'QH', 'W']
        assert _detect_scan_axes(
            [1, 1], [2, 2.006], [3, 3], [1, 1]) == ['QK']

    def test_unified_q_model_uses_first_varying_axis(self):
        from types import SimpleNamespace
        from inspy.gui.gui_convfit_uf import MainWindow as UnifiedFitWindow
        from inspy.gui.tools import SqwQScanTwoPeaks

        h = np.array([1.0, 1.1])
        k = np.array([2.0, 2.2])
        l = np.array([3.0, 3.0])
        w = np.array([1.0, 1.0])
        params = np.array([2.1, 3.2, 0.8, 0.1, 0.2, 10.0, 0.0, 2.0])
        window = SimpleNamespace(scan_axis='QH')

        actual = UnifiedFitWindow._sqw_for_scan_axis(
            window, h, k, l, w, params)
        expected = SqwQScanTwoPeaks(h, k, l, w, params)
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize(
        'scan_mode,scan_axis,expected_width_label,expected_summary',
        [('energy', None, 'HWHM1:', 'current energy scan is fitted'),
         ('q', 'QK', 'FWHM1:', 'current QK scan is fitted')])
    def test_unified_fit_uses_matching_model_and_width_convention(
            self, qapp, monkeypatch, scan_mode, scan_axis,
            expected_width_label, expected_summary):
        from types import SimpleNamespace
        import inspy.gui.gui_convfit_uf as fit_module
        from inspy.gui.gui_convfit_uf import MyDataCanvas
        from inspy.gui.tools import SqwDemo

        calls = {}

        class FakeFitter:
            def __init__(self, instrument, model, prefactor, hkle, observed, errors):
                calls['model'] = model

            def fit_ultrafast(self, **kwargs):
                return {
                    'params': np.array([1.0, 2.0, 0.8, 0.1, 0.2, 10.0, 0.0, 2.0]),
                    'param_errors': np.ones(8) * 0.01,
                }

        def fake_resconv(**kwargs):
            calls['plot_model'] = kwargs['sqw']
            return np.ones(101)

        monkeypatch.setattr(fit_module, 'UltraFastFitConv', FakeFitter)
        window = UnifiedFitWindow()
        try:
            assert window.height() == 721
            window.scan_mode = scan_mode
            window.scan_axis = scan_axis
            window.data = np.array([
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0],
                [10.0, 8.0, 6.0],
                np.sqrt([10.0, 8.0, 6.0]),
            ])
            window.data_end = 3
            window.load_instrument = lambda: True
            window.instrument = SimpleNamespace(ResConv=fake_resconv)
            window.chkMagFF.setChecked(False)
            window.dplot = MyDataCanvas(window.fit_data_plot)

            window.fitData()

            expected_model = SqwDemo if scan_mode == 'energy' else window._sqw_for_scan_axis
            assert calls['model'] == expected_model
            assert calls['plot_model'] == expected_model
            output = window.fit_output_text.toPlainText()
            assert expected_width_label in output
            assert expected_summary in output
            if scan_mode == 'energy':
                assert 'The fitted results are:' in output
            assert window._last_fit_export['experimental'].shape == (5, 3)
            assert window._last_fit_export['fitted'].shape == (5, 101)
        finally:
            window.close()

    def test_unified_instrument_configuration_json_round_trip(
            self, qapp, tmp_path, monkeypatch):
        import json
        import inspy.gui.gui_convfit_uf as fit_module

        config_path = tmp_path / 'instrument.json'
        monkeypatch.setattr(
            fit_module.QFileDialog, 'getSaveFileName',
            lambda *args: (str(config_path), ''))
        window = UnifiedFitWindow()
        try:
            window.instr_dropdown.setCurrentText('TAIPAN')
            window.exportInstrumentConfig()

            config = json.loads(config_path.read_text(encoding='utf-8'))
            assert config['format'] == 'inspy-conv-instrument-config'
            assert config['version'] == 1
            assert config['controls']['instr_dropdown'] == 'TAIPAN'
            assert config['controls']['energy_input'] == '14.87'
            assert config['controls']['arms_input'] == '202, 196, 179, 65, 106'

            window.instr_dropdown.setCurrentText('SIKA')
            monkeypatch.setattr(
                fit_module.QFileDialog, 'getOpenFileName',
                lambda *args: (str(config_path), ''))
            window.load_instrument = lambda: True
            window.importInstrumentConfig()
            assert window.instr_dropdown.currentText() == 'TAIPAN'
            assert window.energy_input.text() == '14.87'
            assert window.arms_input.text() == '202, 196, 179, 65, 106'
        finally:
            window.close()

    def test_unified_fit_csv_has_comments_and_aligned_columns(
            self, qapp, tmp_path, monkeypatch):
        import csv
        import inspy.gui.gui_convfit_uf as fit_module

        csv_path = tmp_path / 'fit.csv'
        monkeypatch.setattr(
            fit_module.QFileDialog, 'getSaveFileName',
            lambda *args: (str(csv_path), ''))
        window = UnifiedFitWindow()
        try:
            window._last_fit_export = {
                'parameters': (
                    'The current energy scan is fitted by using\n'
                    'the SqwDemo S(Q, w) function with UltraFastFitConv.\n'
                    'The fitted results are:\n'
                    'Fitted parameter: 1.25 +/- 0.05'),
                'experimental': np.array([
                    [1.0, 1.1], [2.0, 2.1], [3.0, 3.0],
                    [4.0, 4.1], [10.0, 8.0],
                ]),
                'fitted': np.array([
                    [1.0, 1.05, 1.1], [2.0, 2.05, 2.1],
                    [3.0, 3.0, 3.0], [4.0, 4.05, 4.1],
                    [9.5, 9.0, 8.5],
                ]),
            }
            window.exportFitCsv()

            with csv_path.open(encoding='utf-8', newline='') as csv_file:
                text = csv_file.read()
            lines = text.splitlines()
            assert lines[:4] == [
                '# The current energy scan is fitted by using',
                '# the SqwDemo S(Q, w) function with UltraFastFitConv.',
                '# The fitted results are:',
                '# Fitted parameter: 1.25 +/- 0.05',
            ]
            assert '"' not in text
            assert len(set(
                [index for index, character in enumerate(lines[4])
                 if character == ','])) == 9
            assert [index for index, character in enumerate(lines[4])
                    if character == ','] == [
                index for index, character in enumerate(lines[5])
                if character == ',']

            rows = list(csv.reader(lines[4:], skipinitialspace=True))
            assert [value.strip() for value in rows[0]] == [
                'H', 'K', 'L', 'W', 'Intensity',
                'newH', 'newK', 'newL', 'newW', 'fitted_Intensity',
            ]
            assert len(rows[3]) == 10
            assert [value.strip() for value in rows[1][:5]] == [
                '1', '2', '3', '4', '10']
            assert [value.strip() for value in rows[1][5:]] == [
                '1', '2', '3', '4', '9.5']
            assert [value.strip() for value in rows[3][:5]] == [
                '', '', '', '', '']
            assert [value.strip() for value in rows[3][5:]] == [
                '1.1', '2.1', '3', '4.1', '8.5']
        finally:
            window.close()

    def test_unified_load_prompts_when_multiple_axes_vary(
            self, qapp, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QDialog, QMessageBox
        import inspy.gui.gui_convfit_uf as fit_module

        h = np.array([1.0, 1.1, 1.2])
        k = np.array([2.0, 2.1, 2.2])
        l = np.array([3.0, 3.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        intensity = np.array([10.0, 8.0, 6.0])
        loaded = (h, k, l, w, intensity, np.sqrt(intensity))
        filename = tmp_path / 'q-scan.txt'
        filename.write_text('data')

        class FakeImportDialog:
            result_data = loaded

            def __init__(self, *args):
                pass

            def exec_(self):
                return QDialog.Accepted

        monkeypatch.setattr(
            fit_module.QFileDialog, 'getOpenFileName',
            lambda *args: (str(filename), ''))
        monkeypatch.setattr(fit_module, 'HKLWImportDialog', FakeImportDialog)
        warnings = []
        prompts = []
        monkeypatch.setattr(
            QMessageBox, 'warning',
            lambda parent, title, message: warnings.append((title, message)))
        monkeypatch.setattr(
            fit_module.QInputDialog, 'getItem',
            lambda parent, title, label, items, current, editable:
            (prompts.extend(items) or 'QK', True))

        window = UnifiedFitWindow()
        try:
            window.loadData()
            assert warnings and warnings[0][0] == 'Multiple scan axes detected'
            assert prompts == ['QH', 'QK']
            assert window.scan_mode == 'q'
            assert window.scan_axis == 'QK'
            assert window.label.text() == 'Q1:'
            assert np.allclose(window.dplot.axes.lines[0].get_xdata(), k)
            assert window.dplot.axes.get_xlabel() == 'QK [rlu]'
            assert window.data.shape == (6, 3)
        finally:
            window.close()

    def test_unified_load_selects_energy_model(self, qapp, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QDialog
        import inspy.gui.gui_convfit_uf as fit_module

        h = np.array([1.0, 1.0, 1.0])
        k = np.array([0.0, 0.0, 0.0])
        l = np.array([0.0, 0.0, 0.0])
        w = np.array([1.0, 2.0, 3.0])
        intensity = np.array([10.0, 8.0, 6.0])
        loaded = (h, k, l, w, intensity, np.sqrt(intensity))
        filename = tmp_path / 'energy-scan.txt'
        filename.write_text('data')

        class FakeImportDialog:
            result_data = loaded

            def __init__(self, *args):
                pass

            def exec_(self):
                return QDialog.Accepted

        monkeypatch.setattr(
            fit_module.QFileDialog, 'getOpenFileName',
            lambda *args: (str(filename), ''))
        monkeypatch.setattr(fit_module, 'HKLWImportDialog', FakeImportDialog)

        window = UnifiedFitWindow()
        try:
            window.loadData()
            assert window.scan_mode == 'energy'
            assert window.scan_axis is None
            assert window.label.text() == 'En1:'
            assert np.allclose(window.dplot.axes.lines[0].get_xdata(), w)
            assert window.dplot.axes.get_xlabel() == 'Energy [meV]'
        finally:
            window.close()

    def test_unified_load_prompts_when_energy_and_hkl_both_vary(
            self, qapp, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QDialog, QMessageBox
        import inspy.gui.gui_convfit_uf as fit_module

        filename = tmp_path / 'mixed-scan.txt'
        filename.write_text('data')
        intensity = np.array([10.0, 8.0])
        loaded = (
            np.array([1.0, 1.1]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            np.array([1.0, 2.0]), intensity, np.sqrt(intensity))

        class FakeImportDialog:
            result_data = loaded

            def __init__(self, *args):
                pass

            def exec_(self):
                return QDialog.Accepted

        monkeypatch.setattr(
            fit_module.QFileDialog, 'getOpenFileName',
            lambda *args: (str(filename), ''))
        monkeypatch.setattr(fit_module, 'HKLWImportDialog', FakeImportDialog)
        warnings = []
        prompts = []
        monkeypatch.setattr(
            QMessageBox, 'warning',
            lambda parent, title, message: warnings.append((title, message)))
        monkeypatch.setattr(
            fit_module.QInputDialog, 'getItem',
            lambda parent, title, label, items, current, editable:
            (prompts.extend(items) or 'W', True))

        window = UnifiedFitWindow()
        try:
            window.loadData()
            assert warnings and warnings[0][0] == 'Multiple scan axes detected'
            assert prompts == ['QH', 'W']
            assert window.scan_mode == 'energy'
            assert window.scan_axis is None
            assert np.allclose(window.dplot.axes.lines[0].get_xdata(), loaded[3])
        finally:
            window.close()

    def test_configurable_hklw_import(self, tmp_path):
        from inspy.gui.tools import _load_hklw

        custom = tmp_path / 'custom.dat'
        custom.write_text(
            'scan title\n'
            'H|K|L|W|Iobs|extra\n'
            '1|0|0|2|10|999\n'
            '1|0|0|3|8|999\n')

        loaded = _load_hklw(
            str(custom), skip_lines=2, delimiter='|',
            h_col=0, k_col=1, l_col=2, w_col=3, i_col=4)
        assert len(loaded) == 6
        assert np.allclose(loaded[3], [2, 3])
        assert np.allclose(loaded[4], [10, 8])
        assert np.allclose(loaded[5], np.sqrt([10, 8]))

    def test_unified_fitting_without_data_shows_a_clear_warning(
            self, qapp, monkeypatch):
        from qtpy.QtWidgets import QMessageBox

        warnings = []
        monkeypatch.setattr(
            QMessageBox, 'warning',
            lambda parent, title, message: warnings.append((title, message)))
        win = UnifiedFitWindow()
        try:
            win.fitData()
            assert warnings and warnings[0][0] == 'No data'
            assert 'load a scan data file' in warnings[0][1].lower()
        finally:
            win.close()

    def test_hklw_header_is_detected(self, tmp_path):
        """Both GUIs used to disagree on whether the file had a header row."""
        from inspy.gui.tools import _load_hklw

        # The file format is 5 columns (H K L W Iobs); dIobs is computed as sqrt(Iobs).
        with_header = tmp_path / 'with.hklw'
        with_header.write_text('H K L W Iobs\n1 0 0 0 10\n1 0 0 1 8\n')
        without = tmp_path / 'without.hklw'
        without.write_text('1 0 0 0 10\n1 0 0 1 8\n')
        commented = tmp_path / 'commented.hklw'
        commented.write_text('# a comment\n\n1 0 0 0 10\n1 0 0 1 8\n')

        a = _load_hklw(str(with_header))
        b = _load_hklw(str(without))
        c = _load_hklw(str(commented))
        assert len(a) == 6
        for other in (b, c):
            for col in range(6):
                assert np.allclose(np.asarray(a[col]), np.asarray(other[col]))
        # dIobs is derived as sqrt(Iobs), not read from the file.
        assert np.allclose(np.asarray(a[5]), np.sqrt(np.asarray(a[4])))

    def test_param_fixed_string_is_validated(self):
        """One 0/1 token per parameter; junk used to be silently filtered out."""
        from inspy.gui.tools import _parse_param_fixed

        assert list(_parse_param_fixed('1 0 0', 3)) == [1, 0, 0]
        # An empty field is an error, not a silent "all free": the mask length must
        # match the parameter vector exactly.
        for bad in ('', '1 0 x', '1 0 1.0', '1 0', '1 0 0 0 0'):
            with pytest.raises(ValueError):
                _parse_param_fixed(bad, 3)
        # A named mask gives a better message about which one is wrong.
        with pytest.raises(ValueError, match='fixed mask'):
            _parse_param_fixed('1 0', 3, name='fixed mask')

    def test_curvature_radius_uses_cm_and_accepts_flat_values(self):
        from inspy.gui.tools import _parse_curvature_radius

        assert _parse_curvature_radius('39.5', 'Analyzer horizontal') == 39.5
        assert np.isinf(_parse_curvature_radius('None', 'Analyzer horizontal'))
        assert np.isinf(_parse_curvature_radius('flat', 'Analyzer horizontal'))
        for invalid in ('0', '-1', 'radius'):
            with pytest.raises(ValueError, match='cm'):
                _parse_curvature_radius(invalid, 'Analyzer horizontal')

    def test_qscan_axis_detection_handles_multidimensional_and_small_scans(self):
        from inspy.gui.tools import _detect_qscan_axes

        axes = _detect_qscan_axes(
            [1.0, 1.001, 1.0, 1.001],
            [0.0, 0.0, 0.01, 0.01],
            [2.0, 2.0, 2.0, 2.0])
        assert axes == ['QH', 'QK']
        assert _detect_qscan_axes([1, 1], [2, 2], [3, 3]) == []

    @pytest.mark.parametrize(
        'field_name,value,message_part',
        [
            ('fit_param_fixed', '1 1 typo 0 0 0 0 0', 'parameter fixed mask'),
            ('mono_hcurve_input', '0', 'cm'),
        ])
    def test_invalid_mask_and_curvature_show_a_warning(
            self, qapp, monkeypatch, field_name, value, message_part):
        from qtpy.QtWidgets import QMessageBox

        win = UnifiedFitWindow()
        messages = []
        monkeypatch.setattr(
            QMessageBox, 'warning',
            lambda parent, title, message: messages.append(message))
        try:
            getattr(win, field_name).setText(value)
            assert win.load_instrument() is False
            assert messages and message_part in messages[0]
        finally:
            win.close()
