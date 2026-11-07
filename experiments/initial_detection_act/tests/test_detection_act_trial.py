import sys
from pathlib import Path
import threading
import tempfile
import json
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_detection_act import bounded_targets, Trial


class TrialTest(unittest.TestCase):
    def test_next_trial_preserves_logs_and_invalidates_detections(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=Trial.__new__(Trial)
            trial.session_output=trial.output=Path(folder)
            (trial.output/'metadata.json').write_text('{}')
            previous=trial.output/'steps.jsonl';previous.write_text('{"index":1}\n')
            trial.trial_number=1;trial.completed_steps=10;trial.misses=3
            trial.last_state=[];trial.worker=None
            trial.returner=type('Returner',(),{'active':False})()
            trial.frozen=np.ones((2,6));trial.boxes=np.ones((2,5));trial.last_prediction=[1]*5
            trial.next_trial()
            self.assertEqual(previous.read_text(),'{"index":1}\n')
            self.assertEqual(trial.output.name,'trial_002')
            self.assertIsNone(trial.frozen);self.assertIsNone(trial.last_prediction)
            self.assertEqual(trial.completed_steps,0)
            self.assertEqual(json.loads((Path(folder)/'trial_summary.json').read_text())['steps'],10)
            trial.returner.active=True
            with self.assertRaises(ValueError):trial.next_trial()

    def test_limits_delta_and_wrist(self):
        current=np.array([1,179,40,80,149])
        result=bounded_targets([-300,900,50,12,300],current)
        np.testing.assert_equal(result,[0,180,42,80,150])
        self.assertLessEqual(np.abs(result-current).max(),2)
        for invalid in ([1,2],[1,2,3,float('nan'),5],[1,2,3,float('inf'),5]):
            with self.assertRaises(ValueError):bounded_targets(invalid,current)

    def test_all_batched_acknowledgements_required(self):
        class Port:
            def __init__(self,replies):self.replies=iter(replies);self.written=None
            def write(self,data):self.written=data
            def readline(self):return next(self.replies)
        trial=Trial.__new__(Trial)
        trial.arm=type('Arm',(),dict(lock=threading.RLock()))()
        trial.arm.port=Port([b'MOVE accepted: target set\n',b'DONE\n']*5)
        trial.move_batch([70,113,43,80,90])
        self.assertEqual(trial.arm.port.written,b'MOVE 1 70\nMOVE 2 113\nMOVE 3 43\nMOVE 4 80\nMOVE 5 90\n')
        trial.arm.port=Port([b'ERROR: inactive\n',b'DONE\n']*5)
        with self.assertRaises(RuntimeError):trial.move_batch([70,113,43,80,90])


if __name__=='__main__':unittest.main()
