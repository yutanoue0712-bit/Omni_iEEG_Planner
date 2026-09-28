"""Qt event-loop regression: modal source selection must not strand the next job."""
import time
import unittest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QInputDialog, QMainWindow
from brain_viewer.imaging import InputError
from brain_viewer.window import ViewerWindow


class JobHost(QMainWindow):
    run_job = ViewerWindow.run_job
    _job_finished = ViewerWindow._job_finished
    _update_job_progress = ViewerWindow._update_job_progress

    def __init__(self):
        super().__init__()
        self.worker = None
        self._after_job = None
        self._closing = False
        self._job_started = 0.
        self._job_stage = ''
        self._job_timer = QTimer(self)
        self._job_timer.setInterval(20)
        self._job_timer.timeout.connect(self._update_job_progress)
        self.busy = False
        self.messages = []
        self.failures = []

    def _set_busy(self, value):
        self.busy = value

    def message(self, value):
        self.messages.append(value)

    def _job_failed(self, message):
        self.failures.append((message, self.busy, self.worker is None, self._job_timer.isActive()))
        self._after_job = None


def pump_until(predicate, seconds=2):
    end = time.monotonic()+seconds
    while not predicate() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(.005)
    QApplication.processEvents()
    return predicate()


class JobLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = JobHost()

    def tearDown(self):
        if self.host.worker is not None:
            self.host.worker.wait(2000)
            QApplication.processEvents()
        self.host.deleteLater()
        QApplication.processEvents()

    def test_modal_source_selection_continues_to_import(self):
        calls = []
        def found(_):
            dialog = QInputDialog(self.host)
            dialog.setComboBoxItems(['Synthetic CT'])
            QTimer.singleShot(60, dialog.accept)
            dialog.exec()
            self.host._after_job = lambda: calls.append('import')
            dialog.deleteLater()
        self.host.run_job(lambda progress: 'sources', found, 'finding')
        self.assertTrue(pump_until(lambda: calls == ['import']))
        self.assertIsNone(self.host.worker)
        self.assertFalse(self.host.busy)
        self.assertFalse(self.host._job_timer.isActive())

    def test_failure_is_reported_after_worker_cleanup(self):
        def fail(_):
            raise InputError('invalid NIfTI geometry')
        self.host.run_job(fail, lambda _: None, 'checking')
        self.assertTrue(pump_until(lambda: bool(self.host.failures)))
        self.assertEqual(self.host.failures[0], ('invalid NIfTI geometry', False, True, False))

    def test_completion_can_start_another_job_without_losing_it(self):
        calls = []
        def first(_):
            self.host.run_job(lambda progress: 'second', calls.append, 'second stage')
        self.host.run_job(lambda progress: None, first, 'first stage')
        self.assertTrue(pump_until(lambda: calls == ['second']))
        self.assertTrue(pump_until(lambda: self.host.worker is None))


if __name__ == '__main__':
    unittest.main()
