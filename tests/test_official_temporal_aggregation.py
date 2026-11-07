"""Compare the live adapter to the original official dense-buffer calculation."""
import subprocess
import shutil
import sys
from pathlib import Path
import unittest
import tempfile
import threading
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from detection_act import configure_official
configure_official(ROOT/'runs/temporal_aggregation_test',1)
from imitate_episodes import temporal_aggregate
from test_image_act import OfficialTemporalEnsemble, ImageTrial


class OfficialAggregationTest(unittest.TestCase):
    def make_trial(self,folder,stop_during_prediction=False):
        trial=ImageTrial.__new__(ImageTrial)
        trial.stop=threading.Event();trial.output=Path(folder);trial.timings=[];trial.misses=0
        trial.completed_steps=0;trial.query_count=0;trial.preview_only=False;trial.frozen=np.zeros((2,6))
        trial.query_every=1
        trial.ensemble=OfficialTemporalEnsemble(temporal_aggregate)
        joints=[dict(joint=i+1,angle=a,target=a,active=True,speed=25) for i,a in enumerate([70,113,43,80,90])]
        trial.arm=SimpleNamespace(lock=threading.RLock(),command=lambda text:None,status=lambda:joints)
        trial.refresh=lambda:setattr(trial,'last_state',joints)
        trial.snapshot=lambda:(np.zeros((480,640,3),dtype=np.uint8),trial.query_count)
        self.moves=[];trial.move_batch=lambda target:self.moves.append(target.copy())
        def predict(*args):
            if stop_during_prediction:trial.stop.set()
            return np.tile(np.deg2rad([70,113,43,80,90]),(100,1)),dict(roundtrip_ms=1500,gpu_inference_ms=54)
        trial.backend=SimpleNamespace(config={'mode':'image'},health={'action_mean':[0]*5,'action_std':[1]*5},predict=predict)
        return trial

    def test_every_update_queries_a_new_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.run(0,3)
            self.assertEqual(trial.completed_steps,3)
            self.assertEqual(trial.query_count,3)
            self.assertEqual(len(self.moves),3)
            self.assertEqual(list(trial.ensemble.chunks),[0,1,2])

    def test_query_every_twenty_steps_does_not_skip_control_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            trial=self.make_trial(folder)
            trial.query_every=20
            trial.run(0,41)
            self.assertEqual(trial.completed_steps,41)
            self.assertEqual(len(self.moves),41)
            self.assertEqual(trial.query_count,3)
            self.assertEqual(list(trial.ensemble.chunks),[0,20,40])

    def test_sparse_ensemble_matches_official_dense_buffer(self):
        adapter=OfficialTemporalEnsemble(temporal_aggregate)
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
            trial.run(0,3)
            self.assertEqual(trial.query_count,1)
            self.assertEqual(len(self.moves),0)
            self.assertEqual(trial.completed_steps,0)

    def test_matches_original_code_at_all_steps_including_expiration(self):
        # Reconstruct upstream from the committed patch; no nested Git history required.
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'imitate_episodes.py'
            shutil.copyfile(ROOT/'reference/act/imitate_episodes.py',target)
            subprocess.run(['git','apply','--reverse','--include=imitate_episodes.py',
                str(ROOT/'patches/act_cpu.patch')],cwd=folder,check=True)
            original=target.read_text()
        start=original.index('                        actions_populated = torch.all(')
        end=original.index('\n                    else:',start)
        original_body='\n'.join(line[24:] for line in original[start:end].splitlines())
        # Upstream hard-codes CUDA here; only relocate that tensor for this CPU test.
        original_body=original_body.replace('.cuda()', ".to('cpu')")
        adapter=OfficialTemporalEnsemble(temporal_aggregate)
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
        ensemble=OfficialTemporalEnsemble(temporal_aggregate)
        ensemble.action(0,np.ones((100,5),dtype=np.float32),remember=False)
        self.assertEqual(len(ensemble.chunks),0)
        trial=ImageTrial.__new__(ImageTrial)
        target=trial.targets_for_prediction([90,120,50,81,110],[70,113,43,80,90])
        np.testing.assert_equal(target,[90,120,50,81,110])
        with self.assertRaises(ValueError):trial.targets_for_prediction([90,120,50,81,160],[70,113,43,80,90])


if __name__=='__main__':unittest.main()
