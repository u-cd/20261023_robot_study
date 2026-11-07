"""Exercise a complete local ACT training/save/reload without the upstream repository."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from act.policy import CONFIG, load_policy_config
from act.imitate_episodes import train, load_act


class ACTConfigurationTest(unittest.TestCase):
    def test_checkpoint_snapshot_takes_priority_and_legacy_weights_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_policy_config(folder),json.loads(CONFIG.read_text())['policy'])
            saved=dict(load_policy_config(),hidden_dim=192)
            (Path(folder)/'policy_config.json').write_text(json.dumps(saved))
            self.assertEqual(load_policy_config(folder),saved)

    def test_train_save_and_infer_use_the_same_model_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);dataset=folder/'data';dataset.mkdir()
            for i in range(2):
                with h5py.File(dataset/f'episode_{i}.hdf5','w') as f:
                    f.attrs.update(sim=False,action_offset=0)
                    values=np.arange(15,dtype=np.float32).reshape(3,5)/100+i*.01
                    f['observations/qpos']=values
                    f['observations/qvel']=np.zeros_like(values)
                    f['action']=values+.01
                    f['observations/images/top']=np.zeros((3,64,64,3),dtype=np.uint8)
            config=json.loads(CONFIG.read_text())
            config['policy'].update(hidden_dim=16,nheads=2,dim_feedforward=32,enc_layers=1,dec_layers=1,num_queries=2)
            config['training'].update(dataset_dir=str(dataset),num_episodes=2,batch_size=1,num_workers=0,pretrained_backbone=False)
            source=folder/'config.json';source.write_text(json.dumps(config))
            output=folder/'trained'
            train(source,output,device='cpu',epochs=1)
            model,stats=load_act(output)
            self.assertEqual(model.config,dict(config['policy'],device='cpu'))
            with torch.inference_mode():
                actions=model(torch.zeros(1,5),torch.zeros(1,1,3,64,64))
            self.assertEqual(tuple(actions.shape),(1,2,5))
            self.assertTrue(torch.isfinite(actions).all())
            self.assertEqual(stats['qpos_mean'].shape,(5,))
            self.assertTrue((output/'policy_best.ckpt').exists())
            last=torch.load(output/'policy_last.ckpt',map_location='cpu',weights_only=True)
            best=torch.load(output/'policy_best.ckpt',map_location='cpu',weights_only=True)
            self.assertTrue(any(not torch.equal(last[key],best[key]) for key in last))


if __name__=='__main__':unittest.main()
