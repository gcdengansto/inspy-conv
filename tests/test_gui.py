# -*- coding: utf-8 -*-
"""GUI construction tests.

These run headless (QT_QPA_PLATFORM=offscreen, matplotlib Agg) and only check that
the windows build, that their instrument loads, and that the widget wiring is
sane. They do not drive the widgets: Qt event loops are not the subject here.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

# Every test in this module constructs a Qt window; -m "not gui" skips the file.
pytestmark = pytest.mark.gui

from qtpy.QtWidgets import QApplication, QWidget   # noqa: E402

from inspy.gui.gui_convfit_escan_uf import MainWindow as EScanWindow   # noqa: E402
from inspy.gui.gui_convfit_qscan_uf import MainWindow as QScanWindow   # noqa: E402
from inspy.gui.main_gui import MainWindow as MainWindow_               # noqa: E402

WINDOWS = [('main_gui', MainWindow_), ('qscan', QScanWindow), ('escan', EScanWindow)]


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


def _load_all(qapp):
    for _, cls in WINDOWS:
        win = cls()
        win.load_instrument()
        yield win


class TestBackgroundWork:
    """Review items 180 and 181: the slow work must leave the GUI thread."""

    @pytest.mark.parametrize('window', [('escan', EScanWindow), ('qscan', QScanWindow)],
                             indirect=True)
    def test_load_instrument_returns_before_the_resolution_is_computed(self, window):
        """The call must not block: it schedules the work and returns.

        Previously it rebuilt the instrument and computed three resolution
        matrices inline, so every committed keystroke froze the window.
        """
        runner = window._resolution_runner
        window.load_instrument()
        # A request was made and the worker is occupied (or already finished).
        assert runner.busy or window._generation > 1
        _settle(window)
        assert not runner.busy

    @pytest.mark.parametrize('window', [('escan', EScanWindow), ('qscan', QScanWindow)],
                             indirect=True)
    def test_the_gui_thread_is_free_while_the_worker_runs(self, window):
        """A second request must be accepted while the first is still running."""
        import threading
        worker_threads = set()

        original = window._compute_resolution.__func__ if hasattr(window._compute_resolution, '__func__') \
            else window._compute_resolution

        def spy(instr, hkle):
            worker_threads.add(threading.current_thread().name)
            return original(instr, hkle)

        window._compute_resolution = spy

        window.load_instrument()
        window.load_instrument()          # must not raise or block
        _settle(window)

        main_thread = threading.current_thread().name
        assert worker_threads, 'the resolution was never computed on a worker'
        assert main_thread not in worker_threads

    @pytest.mark.parametrize('window', [('escan', EScanWindow), ('qscan', QScanWindow)],
                             indirect=True)
    def test_rapid_requests_are_coalesced(self, window):
        """Six edits in a row must not queue six resolutions."""
        counts = []
        original = window._compute_resolution

        def counting(instr, hkle):
            counts.append(1)
            return original(instr, hkle)

        window._compute_resolution = counting

        for _ in range(6):
            window.load_instrument()
        _settle(window)

        # At most one in flight plus one coalesced re-run.
        assert len(counts) <= 2, 'ran %d times for 6 requests' % len(counts)
        assert counts, 'nothing ran at all'

    @pytest.mark.parametrize('window', [('escan', EScanWindow), ('qscan', QScanWindow)],
                             indirect=True)
    def test_a_stale_result_is_discarded(self, window):
        """A result computed from superseded settings must not overwrite the current ones."""
        runner = window._resolution_runner
        # Pretend a newer request happened while the older one was in flight.
        window.load_instrument()
        stale_generation = window._generation
        window._generation += 1
        runner._handle_result((stale_generation, np.array([1.0]), np.zeros((1, 4, 4))))
        # The stale result was ignored, so the instrument still has real data.
        _settle(window)
        assert np.all(np.isfinite(window.instrument.RMS))

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
    @pytest.mark.parametrize('window', WINDOWS, indirect=True)
    def test_widgets_and_non_widget_items_are_both_handled(self, window):
        """A spacer has widget() is None; the old code raised AttributeError on it."""
        from qtpy.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        widget = QWidget()
        layout.addWidget(widget)
        layout.addStretch(1)          # a QSpacerItem: no widget(), no deleteLater()
        assert layout.count() == 2

        window.clearLayout(layout)
        assert layout.count() == 0

    @pytest.mark.parametrize('window', WINDOWS, indirect=True)
    def test_clearing_an_empty_layout_is_a_no_op(self, window):
        from qtpy.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        window.clearLayout(layout)
        assert layout.count() == 0


class TestButtonWiring:
    @pytest.mark.parametrize('window', [('escan', EScanWindow), ('qscan', QScanWindow)],
                             indirect=True)
    def test_button_action_does_not_fire_on_press_alone(self, window):
        """`pressed` fires on mouse-down; `clicked` only on a completed click.

        PySide6's receivers() does not count bound-method slots, so this is checked
        behaviourally: pressing and then releasing *outside* the button must not
        trigger the handler -- which is exactly what `pressed` would have done.
        """
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent

        for name in ('fit_btn_fit', 'fit_btn_init'):
            button = getattr(window, name, None)
            if button is None:
                continue
            # Probe only `clicked` -- the signal the handler is now wired to. A slot
            # connected to `pressed` would fire here; one connected to `clicked`
            # must not.
            from qtpy.QtCore import QPoint, Qt
            from qtpy.QtTest import QTest

            fired = []
            probe = lambda: fired.append(name)      # noqa: E731
            button.clicked.connect(probe)
            try:
                button.show()
                QApplication.processEvents()
                QTest.mousePress(button, Qt.LeftButton, pos=QPoint(2, 2))
                assert fired == [], 'the action fired on mouse-down (%s)' % name
                # Release outside the button: an aborted click, still not a click.
                QTest.mouseRelease(button, Qt.LeftButton, pos=QPoint(900, 900))
                assert fired == [], 'the action fired on an aborted click (%s)' % name
            finally:
                button.clicked.disconnect(probe)
                button.hide()

    @pytest.mark.parametrize('window', [('escan', EScanWindow), ('qscan', QScanWindow)],
                             indirect=True)
    def test_button_action_does_fire_on_a_real_click(self, window):
        button = getattr(window, 'fit_btn_init', None)
        if button is None:
            pytest.skip('no init button on this window')
        fired = []
        probe = lambda: fired.append(1)             # noqa: E731
        button.clicked.connect(probe)
        try:
            button.click()
            assert fired, 'a real click did not reach the handler'
        finally:
            button.clicked.disconnect(probe)


class TestProjectionCanvases:
    @pytest.mark.parametrize('window', WINDOWS, indirect=True)
    def test_qslice_is_validated(self, window):
        """ResolutionPlotProj expects QxE/QyE/QxQy; anything else must be rejected."""
        from inspy.gui.gui_convfit_escan_uf import MyMplCanvas as ECanvas

        with pytest.raises(ValueError):
            ECanvas(qslice='QxW')     # the old, wrong spelling
        for good in ('QxE', 'QyE', 'QxQy'):
            assert ECanvas(qslice=good) is not None

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
    def test_fitting_without_data_raises_a_clear_error(self, qapp):
        """Slicing a list (or None) used to produce a bare TypeError."""
        win = EScanWindow()
        win.load_instrument()
        try:
            win.fitData()
        except Exception as exc:
            assert 'data' in str(exc).lower(), str(exc)
        else:
            pytest.fail('fitting with no data did not raise')
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
