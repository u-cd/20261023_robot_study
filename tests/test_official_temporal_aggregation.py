"""Compare the live adapter to the original official dense-buffer calculation."""
import subprocess
import shutil
import sys
from pathlib import Path
import unittest
from unittest import mock
import tempfile
import threading
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
from act.imitate_episodes import temporal_aggregate, TemporalEnsemble, RolloutState
from ui.act_test import ACTTrial


class OfficialAggregationTest(unittest.TestCase):
    def make_trial(self,folder,stop_during_prediction=False):
        trial=ACTTrial.__new__(ACTTrial)
        trial.rollout=RolloutState();trial.temporal_agg=True
        trial.stop=threading.Event();trial.output=Path(folder);trial.timings=[];trial.misses=0
        trial.completed_steps=0;trial.query_count=0;trial.preview_only=False;trial.input_ready=True
        trial.query_every=1
        trial.ensemble=TemporalEnsemble(temporal_aggregate)
        joints=[dict(joint=i+1,angle=a,target=a,active=True,speed=25) for i,a in enumerate([70,113,43,80,90])]
        trial.arm=SimpleNamespace(lock=threading.RLock(),command=lambda text:None,status=lambda:joints)
        trial.refresh=lambda:setattr(trial,'last_state',joints)
        trial.snapshot=lambda:(np.zeros((480,640,3),dtype=np.uint8),trial.query_count)
        self.moves=[];trial.move_batch=lambda target:self.moves.append(target.copy())
        def predict(*args):
            if stop_during_prediction:trial.stop.set()
            return np.tile(np.deg2rad([70,113,43,80,90]),(100,1)),dict(roundtrip_ms=1500,gpu_inference_ms=54)
        trial.config={'mode':'ACT'}
        trial.stats={'action_mean':[0]*5,'action_std':[1]*5}
        trial.model=object()
        patcher=mock.patch('act.imitate_episodes.predict',side_effect=predict)
        patcher.start();self.addCleanup(patcher.stop)
        return trial

    def test_cpu_trial_next_run_resets_only_its_prediction_state(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.session_output=trial.output
            (trial.output/'metadata.json').write_text('{}')
            previous=trial.output/'steps.jsonl'
            previous.write_text('saved commands')
            trial.trial_number=1;trial.worker=None;trial.last_state=[]
            trial.returner=SimpleNamespace(active=False)
            trial.query_every=20
            trial.config={}
            trial.ensemble.action(0,np.ones((100,5),dtype=np.float32))
            trial.next_trial()
            self.assertFalse(trial.input_ready)
            self.assertEqual(trial.query_count,0)
            self.assertEqual(len(trial.ensemble.chunks),0)
            self.assertEqual(previous.read_text(),'saved commands')
            self.assertEqual(trial.output.name,'trial_002')

    def test_act_entrypoint_does_not_import_detection_experiment(self):
        result=subprocess.run([sys.executable,'-c',
            "import sys;import act.imitate_episodes;"
            "assert not {'ui.act_test','test_detection_act','initial_detection_policy','ultralytics','policy','detr','imitate_episodes'} & sys.modules.keys()"],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_every_update_queries_a_new_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.run(3)
            self.assertEqual(trial.completed_steps,3)
            self.assertEqual(trial.query_count,3)
            self.assertEqual(len(self.moves),3)
            self.assertEqual(list(trial.ensemble.chunks),[0,1,2])

    def test_query_every_twenty_steps_does_not_skip_control_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.query_every=20
            trial.run(41)
            self.assertEqual(trial.completed_steps,41)
            self.assertEqual(len(self.moves),41)
            self.assertEqual(trial.query_count,3)
            self.assertEqual(list(trial.ensemble.chunks),[0,20,40])

    def test_without_temporal_aggregation_consumes_newest_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.query_every=20;trial.temporal_agg=False
            def prediction(*args):
                chunk=np.tile(np.deg2rad([70,113,43,80,90]),(100,1))
                chunk[:,0]+=np.deg2rad(np.arange(100))
                return chunk,{}
            with mock.patch('act.imitate_episodes.predict',side_effect=prediction):
                trial.run(21)
            self.assertEqual(trial.query_count,2)
            self.assertEqual(len(self.moves),21)
            self.assertEqual([int(x[0]) for x in self.moves],[*range(70,90),70])

    def test_sparse_ensemble_matches_official_dense_buffer(self):
        adapter=TemporalEnsemble(temporal_aggregate)
        dense=torch.zeros((140,240,5))
        generator=np.random.default_rng(13)
        for t in range(140):
            chunk=generator.normal(size=(100,5)).astype(np.float32) if t%20==0 else None
            if chunk is not None:dense[t,t:t+100]=torch.from_numpy(chunk)
            expected=temporal_aggregate(dense[:,t],'cpu')[0].numpy()
            np.testing.assert_array_equal(adapter.action(t,chunk),expected)

    def test_stop_while_waiting_discards_result_without_moving(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder,True)
            trial.run(3)
            self.assertEqual(trial.query_count,1)
            self.assertEqual(len(self.moves),0)
            self.assertEqual(trial.completed_steps,0)

    def test_matches_original_code_at_all_steps_including_expiration(self):
        # Reconstruct upstream from the committed patch; no nested Git history required.
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'act/imitate_episodes.py'
            target.parent.mkdir()
            shutil.copyfile(ROOT/'act/imitate_episodes.py',target)
            subprocess.run(['git','apply','--reverse','--include=act/imitate_episodes.py',
                str(ROOT/'act/official.patch')],cwd=folder,check=True)
            original=target.read_text()
        start=original.index('                        actions_populated = torch.all(')
        end=original.index('\n                    else:',start)
        original_body='\n'.join(line[24:] for line in original[start:end].splitlines())
        # Upstream hard-codes CUDA here; only relocate that tensor for this CPU test.
        original_body=original_body.replace('.cuda()', ".to('cpu')")
        adapter=TemporalEnsemble(temporal_aggregate)
        dense=torch.zeros((125,225,5))
        generator=np.random.default_rng(7)
        for t in range(125):
            chunk=generator.normal(size=(100,5)).astype(np.float32)
            if t==0:chunk[1,0]=0 # Preserve the original all-components-nonzero mask.
            dense[t,t:t+100]=torch.from_numpy(chunk)
            scope=dict(torch=torch,np=np,device='cpu',actions_for_curr_step=dense[:,t])
            exec(original_body,scope)
            actual=adapter.action(t,chunk)
            np.testing.assert_array_equal(actual,scope['raw_action'][0].numpy())
            self.assertLessEqual(len(adapter.chunks),100)

    def test_preview_does_not_add_history_and_hardware_has_no_delta_clip(self):
        ensemble=TemporalEnsemble(temporal_aggregate)
        ensemble.action(0,np.ones((100,5),dtype=np.float32),remember=False)
        self.assertEqual(len(ensemble.chunks),0)
        trial=ACTTrial.__new__(ACTTrial)
        target=trial.targets_for_prediction([90,120,50,81,110],[70,113,43,80,90])
        np.testing.assert_equal(target,[90,120,50,81,110])
        with self.assertRaises(ValueError):trial.targets_for_prediction([90,120,50,81,160],[70,113,43,80,90])


if __name__=='__main__':unittest.main()
