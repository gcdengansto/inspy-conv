# -*- coding: utf-8 -*-
"""Background task helpers for the Qt GUIs.

Two things the conv-fit and resolution GUIs do are far too slow to run on the Qt
main thread:

* rebuilding the instrument and computing the resolution matrices (three
  ``CalcResMatHKL`` calls plus three projections) on every committed keystroke or
  dropdown change -- there are ~50 signals wired to ``load_instrument``; and
* a full ``least_squares`` fit, where each step costs ``1 + 2*n_free`` 4-D
  convolutions over every data point.

Both used to run synchronously, so every committed keystroke froze the window
(review items 180 and 181). This module provides the two pieces that fixes:

* :class:`DebouncedRunner` - runs a callable on a ``QThreadPool`` worker thread,
  **coalescing** requests: while one run is in flight, further requests collapse
  into a single pending re-run. Typing quickly therefore costs one extra
  computation, not one per keystroke.
* :class:`ProgressDialog` - a cancel-able progress dialog for a single
  long-running job (the fit), with a real cancel path.

The critical rule both rely on: **Qt widgets may only be touched on the GUI
thread.** Callers must therefore snapshot whatever they need from the widgets
*before* handing work to a runner, and apply results back *after* the result
signal fires (which Qt delivers on the GUI thread).
"""

import traceback

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QProgressDialog


class _WorkerSignals(QObject):
    """Signals a :class:`QRunnable` uses to report back. Emitted from the worker."""

    finished = Signal(object)   # (result,)
    failed = Signal(object)     # (exception,)
    done = Signal()             # always last, so the runner can clear its state


class _Worker(QRunnable):
    """Adapts a plain callable so it can run on a QThreadPool thread."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self.signals = _WorkerSignals()

    def run(self):
        try:
            result = self._fn()
        except BaseException as exc:            # noqa: BLE001 - reported, not swallowed
            # Keep the traceback attached: without it, a worker-thread exception
            # reaches the GUI as a bare ValueError with no indication of where.
            exc._inspy_traceback = traceback.format_exc()
            self.signals.failed.emit(exc)
        else:
            self.signals.finished.emit(result)
        finally:
            self.signals.done.emit()


class DebouncedRunner(QObject):
    """Run ``work()`` on a worker thread, coalescing overlapping requests.

    ``work`` is called with no arguments and its return value is passed to the
    ``on_result`` callback on the GUI thread. Exceptions go to ``on_error``; the
    default reports them in a message box rather than letting a worker-thread
    exception vanish.

    While a run is in flight, further :meth:`request` calls set a single pending
    flag; exactly one more run happens when the current one finishes. That is the
    property that matters for the ``editingFinished`` signals: a user typing into
    a field fires one of those per commit, and a rebuild per commit is what made
    the window unusable.
    """

    def __init__(self, work, on_result=None, on_error=None, parent=None):
        super().__init__(parent)
        self._work = work
        self._on_result = on_result
        self._on_error = on_error or self._default_error_handler
        self._pool = QThreadPool(self)
        # One thread: these jobs are CPU-bound pure-Python/numpy, so extra
        # threads would only contend for the GIL and make the coalescing
        # behaviour harder to reason about.
        self._pool.setMaxThreadCount(1)

        self._busy = False
        self._pending = False
        self._cancelled = False

    # -- public API ---------------------------------------------------------

    def request(self):
        """Ask for a run. Returns immediately; the GUI thread is never blocked."""
        if self._busy:
            # Collapse every further request into the single pending re-run.
            self._pending = True
        else:
            self._start()

    def cancel(self):
        """Ask the *next* run not to start.

        Work already in progress cannot be interrupted (the underlying numpy and
        scipy calls have no cancellation point); this drops the pending re-run so
        a cancelled result is not delivered late, after the user has moved on.
        """
        self._cancelled = True
        self._pending = False

    @property
    def busy(self):
        return self._busy

    def wait(self, msecs=-1):
        """Block until the pool is idle. Intended for tests, not for the GUI."""
        return self._pool.waitForDone(msecs)

    # -- internals ----------------------------------------------------------

    def _start(self):
        self._busy = True
        self._cancelled = False
        worker = _Worker(self._work)
        worker.signals.finished.connect(self._handle_result)
        worker.signals.failed.connect(self._handle_error)
        worker.signals.done.connect(self._handle_done)
        self._pool.start(worker)

    def _handle_result(self, result):
        if self._cancelled:
            return
        self._on_result(result)

    def _handle_error(self, exc):
        if self._cancelled:
            return
        self._on_error(exc)

    def _handle_done(self):
        self._busy = False
        if self._pending:
            self._pending = False
            self._start()

    @staticmethod
    def _default_error_handler(exc):
        tb = getattr(exc, '_inspy_traceback', None)
        if tb:
            print(tb)
        try:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'Calculation failed', str(exc))
        except Exception:                                  # pragma: no cover
            print('%s: %s' % (type(exc).__name__, exc))


class ProgressDialog(QProgressDialog):
    """Cancel-able progress dialog for one long job (the fit).

    ``run`` is executed on a worker thread; its return value is delivered to
    ``on_finished`` on the GUI thread. Cancelling sets a flag the worker can poll
    through :meth:`is_cancelled`, because a running ``least_squares`` cannot be
    preempted safely -- the honest options are "stop at the next step" or "let it
    finish and discard", and this offers the former.
    """

    def __init__(self, parent, label, on_finished=None, on_error=None):
        super().__init__(label, 'Cancel', 0, 0, parent)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(0)          # show immediately, not after 4 s
        self.setAutoClose(False)
        self.setAutoReset(False)

        self._cancelled = False
        self._on_finished = on_finished
        self._on_error = on_error or DebouncedRunner._default_error_handler
        self._worker = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self.canceled.connect(self._mark_cancelled)

    def is_cancelled(self):
        """Poll this from inside the worker to stop at a safe point."""
        return self._cancelled

    def start(self, work):
        self._cancelled = False
        self.setValue(0)
        self.show()
        worker = _Worker(work)
        worker.signals.finished.connect(self._deliver)
        worker.signals.failed.connect(self._fail)
        worker.signals.done.connect(self._close)
        self._worker = worker
        self._pool.start(worker)

    def _mark_cancelled(self):
        self._cancelled = True
        self.setLabelText('Stopping at the next iteration...')

    def _deliver(self, result):
        if self._cancelled:
            return
        if self._on_finished is not None:
            self._on_finished(result)

    def _fail(self, exc):
        if self._cancelled:
            return
        self._on_error(exc)

    def _close(self):
        self._worker = None
        self.close()
