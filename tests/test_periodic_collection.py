import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import h5py
import numpy as np

from scripts.collect_demonstrations import Recorder, export_hdf5, validate_initial


CONFIG={'initial_angles_deg':[70,113,43,80,90]}


def state(angles=None):
    angles=angles or CONFIG['initial_angles_deg']
    return {'connected':True,'joints':[{'joint':i+1,'active':True,'angle':a,'target':a,'speed':25} for i,a in enumerate(angles)]}


class FakeCamera:
    def __init__(self):
        self.jpg=cv2.imencode('.jpg',np.full((12,16,3),(10,40,200),np.uint8))[1].tobytes()
    def snapshot(self):
        return (1,time.monotonic_ns(),self.jpg)


class CollectionTests(unittest.TestCase):
    def test_initial_pose_and_active_check(self):
        validate_initial(state(),CONFIG)
        with self.assertRaises(ValueError): validate_initial(state([70,112,43,80,90]),CONFIG)
        s=state();s['joints'][1]['active']=False
        with self.assertRaises(ValueError): validate_initial(s,CONFIG)

    def test_periodic_raw_and_export_without_motion_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            requests=[]
            def send(data):
                requests.append(data)
                return state()
            recorder=Recorder(FakeCamera(),send,CONFIG,Path(tmp),hz=50)
            recorder.start()
            deadline=time.monotonic()+2
            while recorder.samples<4 and time.monotonic()<deadline: time.sleep(.01)
            folder=recorder.folder
            recorder.event({'action':'jog','joint':1,'direction':'+'},{'accepted':True},10,20)
            result=recorder.finish(True)
            self.assertGreaterEqual(result['samples'],4)
            self.assertTrue(all(r=={'action':'status'} for r in requests))
            self.assertEqual(len(list((folder/'frames').glob('*.jpg'))),1)
            self.assertEqual(result['reused_frames'],result['samples']-1)
            self.assertTrue((folder/'viewer.html').exists())
            self.assertEqual(json.loads((folder/'episode.json').read_text())['status'],'success')
            with h5py.File(folder/'preview.hdf5') as f:
                self.assertFalse(f.attrs['training_ready'])
                self.assertFalse(f.attrs['sim'])
                np.testing.assert_allclose(f['observations/qpos'][0],np.deg2rad(CONFIG['initial_angles_deg']),rtol=1e-6)
                self.assertEqual(f['observations/images/top'].shape,(result['samples'],12,16,3))
                self.assertGreater(int(f['observations/images/top'][0,0,0,0]),150) # RGB, not BGR

    def test_export_preserves_raw_and_pairs_next_commanded_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'frames').mkdir();(p/'frames/1.jpg').write_bytes(FakeCamera().jpg)
            rows=[]
            for i,angle in enumerate([70,71,73]):
                rows.append({'state':state([angle,113,43,80,90]),'state_start_ns':i*20_000_000,
                             'state_end_ns':i*20_000_000+1,'tick':i,'image':'frames/1.jpg',
                             'frame_sequence':1,'frame_captured_ns':0})
            raw=''.join(json.dumps(r)+'\n' for r in rows);(p/'samples.jsonl').write_text(raw)
            export_hdf5(p)
            self.assertEqual((p/'samples.jsonl').read_text(),raw)
            with h5py.File(p/'preview.hdf5') as f:
                np.testing.assert_allclose(np.rad2deg(f['action'][:,0]),[71,73,73],atol=1e-4)
                self.assertTrue(np.isfinite(f['observations/qvel'][:]).all())

    def test_discard_deletes_only_current_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            sibling=Path(tmp)/'existing';sibling.mkdir()
            recorder=Recorder(FakeCamera(),lambda _:state(),CONFIG,Path(tmp))
            recorder.start();folder=recorder.folder
            recorder.finish(False)
            self.assertFalse(folder.exists());self.assertTrue(sibling.exists())


if __name__=='__main__': unittest.main()
