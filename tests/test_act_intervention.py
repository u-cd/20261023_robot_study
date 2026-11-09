"""Intervention logging and resumed inference, without a camera or robot."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from act.imitate_episodes import RolloutState, predict_action
from ui.act_test import ACTTrial


class InterventionTests(unittest.TestCase):
    def make_trial(self, folder):
        trial=ACTTrial.__new__(ACTTrial)
        trial.output=Path(folder);trial.rollout=RolloutState(step=23)
        trial.lock=threading.RLock();trial.stop=threading.Event();trial.worker=None
        trial.returner=SimpleNamespace(active=False,thread=None,cancel=lambda:None)
        trial.last_state=[dict(joint=i+1,angle=a,target=a,active=True,speed=25)
                          for i,a in enumerate([70,113,43,80,90])]
        trial.snapshot=lambda:(np.zeros((480,640,3),np.uint8),123)
        def command(text):
            if text.startswith('MOVE'):
                _,joint,angle=text.split();s=trial.last_state[int(joint)-1]
                s.update(angle=int(angle),target=int(angle))
            return []
        trial.arm=SimpleNamespace(lock=threading.RLock(),port=object(),
                                  command=command,status=lambda:trial.last_state)
        return trial

    def test_loop_manual_correction_and_new_query_share_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.ensemble.action(20,np.ones((100,5),np.float32))
            trial.action({'action':'mark_loop'})
            trial.action({'action':'manual','command':{'action':'move','joint':1,'angle':72}})
            records=[json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in records],
                             ['stop_accepted','loop_marked','manual_requested','manual_accepted'])
            self.assertEqual(records[-2]['commanded_state'][0]['angle'],70)
            self.assertEqual(records[-1]['commanded_state'][0]['angle'],72)
            self.assertTrue(all((Path(folder)/r['image']).is_file() for r in records))
            self.assertEqual(trial.completed_steps,23)
            self.assertEqual(len(trial.ensemble.chunks),0)
            observation={'qpos':np.array([72,113,43,80,90]),'image':np.zeros((480,640,3),np.uint8)}
            with patch('act.imitate_episodes.predict',return_value=(np.ones((100,5),np.float32),{})) as model:
                _,details=predict_action(None,dict(action_std=np.ones(5),action_mean=np.zeros(5)),
                                        observation,trial.rollout,query_every=20)
            model.assert_called_once();self.assertTrue(details['queried'])

    def test_reject_manual_motion_during_model_run_and_log_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder);trial.worker=SimpleNamespace(is_alive=lambda:True)
            with self.assertRaises(ValueError):
                trial.action({'action':'manual','command':{'action':'move','joint':1,'angle':80}})
            self.assertEqual(trial.last_state[0]['angle'],70)
            records=[json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['event'],'manual_failed')

    def test_stop_still_executes_when_recording_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            with patch.object(trial,'record_event',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):trial.action({'action':'stop'})
            self.assertTrue(trial.stop.is_set())

    def test_every_step_records_aggregation_and_cached_query_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.timings=[];trial.misses=0;trial.config={'mode':'ACT'}
            observation={'qpos':np.array([70,113,43,80,90]),'frame_sequence':123}
            prediction=np.array([71,113,43,80,90])
            trial.record_prediction(observation,prediction,
                dict(logical_step=23,queried=False,aggregated_action_deg=prediction.tolist()))
            trial.record_step(24,observation,prediction,prediction,.02,.05)
            record=json.loads((Path(folder)/'steps.jsonl').read_text())
            self.assertFalse(record['prediction_details']['queried'])
            self.assertEqual(record['prediction_details']['logical_step'],23)
            self.assertEqual(record['sent_deg'],prediction.tolist())
