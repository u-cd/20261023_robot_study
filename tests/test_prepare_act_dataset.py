import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scripts.prepare_act_dataset import resample


def rows():
    result=[]
    for i,t in enumerate([0,70_000_000,150_000_000,200_000_000]):
        result.append(dict(state_start_ns=t,state_end_ns=t+10_000_000,
                           frame_sequence=i,frame_captured_ns=t,image=f'{i}.jpg',
                           state={'connected':True,'joints':[dict(joint=j+1,active=True,angle=70+t/1e9*20) for j in range(5)]}))
    return result


class ConversionTests(unittest.TestCase):
    def test_grid_and_causal_frames_next_position(self):
        data=resample(rows(),20)
        np.testing.assert_allclose(data['time'],[0,.05,.1,.15])
        np.testing.assert_allclose(np.rad2deg(data['qpos'][:,0]),[70,71,72,73],atol=1e-4)
        np.testing.assert_allclose(np.rad2deg(data['action'][:,0]),[71,72,73,74],atol=1e-4)
        self.assertEqual(data['images'],['0.jpg','0.jpg','1.jpg','2.jpg'])
        self.assertTrue((data['frame_age']>=0).all())

    def test_reject_disconnected_and_long_gap(self):
        r=rows();r[0]['state']['connected']=False
        with self.assertRaises(ValueError):resample(r,20)
        r=rows();r[-1]['state_start_ns']=900_000_000;r[-1]['state_end_ns']=910_000_000
        with self.assertRaises(ValueError):resample(r,20)

    def test_official_legacy_and_aligned_offsets(self):
        spec=importlib.util.spec_from_file_location('official_utils',Path('reference/act/utils.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'episode_0.hdf5'
            with h5py.File(path,'w') as f:
                f.attrs['sim']=False
                f.create_dataset('observations/qpos',data=np.zeros((10,5)))
                f.create_dataset('observations/qvel',data=np.zeros((10,5)))
                f.create_dataset('observations/images/top',data=np.zeros((10,2,2,3),dtype='uint8'))
                f.create_dataset('action',data=np.repeat(np.arange(10)[:,None],5,axis=1))
            stats={'action_mean':np.zeros(5),'action_std':np.ones(5),'qpos_mean':np.zeros(5),'qpos_std':np.ones(5)}
            with patch.object(module.np.random,'choice',return_value=4):
                ds=module.EpisodicDataset([0],tmp,['top'],stats)
                self.assertEqual(ds[0][2][0,0].item(),3) # upstream real-data default
                with h5py.File(path,'a') as f:f.attrs['action_offset']=0
                self.assertEqual(ds[0][2][0,0].item(),4)
                with h5py.File(path,'a') as f:
                    del f.attrs['action_offset'];f.attrs['sim']=True
                self.assertEqual(ds[0][2][0,0].item(),4) # upstream simulation default


if __name__=='__main__':unittest.main()
