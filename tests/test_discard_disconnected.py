import unittest
from unittest.mock import Mock
from scripts.collect_demonstrations import finish_recording

class DiscardTests(unittest.TestCase):
    def test_disconnected_discard_still_finishes(self):
        recorder=Mock();recorder.finish.return_value={'discarded':True}
        result=finish_recording(recorder,Mock(side_effect=RuntimeError('disconnected')),False)
        recorder.finish.assert_called_once_with(False)
        self.assertTrue(result['discarded']);self.assertIn('disconnected',result['stop_warning'])

    def test_disconnected_save_never_claims_success(self):
        recorder=Mock()
        with self.assertRaises(RuntimeError):finish_recording(recorder,Mock(side_effect=RuntimeError('disconnected')),True)
        recorder.finish.assert_not_called()
