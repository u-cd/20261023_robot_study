"""Integration checks for recorded-time alignment and frozen object inputs."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/initial_detection_act'))
from initial_detection_policy import InitialDetectionDataset


@unittest.skipUnless((ROOT/'data/act_initial_detections_cpu_20261107/manifest.json').exists(),
                     'Prepared local demonstration data unavailable')
class RecordedDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT/'data/act_initial_detections_cpu_20261107'
        cls.manifest = json.loads((cls.directory/'manifest.json').read_text())
        with np.load(cls.directory/'normalization.npz') as f:
            cls.stats = {k:f[k].copy() for k in f.files}
        cls.dataset = InitialDetectionDataset(cls.directory, range(50), cls.stats)

    def test_alignment_frozen_detections_and_padding_all_episodes(self):
        for episode in range(50):
            arrays = self.dataset.arrays[episode]
            first = self.dataset.sample_at(episode, 0)[0]
            self.assertEqual(tuple(first.shape), (2,6))
            for t in [0, len(arrays['qpos'])//2, len(arrays['qpos'])-1]:
                detections, qpos, target, mask = self.dataset.sample_at(episode, t)
                self.assertTrue(torch.equal(first, detections))
                np.testing.assert_allclose(qpos.numpy()*self.stats['qpos_std']+self.stats['qpos_mean'],
                                           arrays['qpos'][t], atol=1e-6)
                valid = min(100,len(arrays['action'])-t)
                self.assertEqual(int((~mask).sum()), valid)
                restored = target.numpy()*self.stats['action_std']+self.stats['action_mean']
                np.testing.assert_allclose(restored[:valid],arrays['action'][t:t+valid],atol=1e-6)

    def test_split_and_real_detection_provenance(self):
        m = self.manifest
        self.assertEqual(m['status'], 'ready')
        train, val = set(m['train_ids_in_official_order']),set(m['validation_ids_in_official_order'])
        self.assertEqual((len(train),len(val)),(40,10))
        self.assertFalse(train & val)
        self.assertEqual(train | val,set(range(50)))
        self.assertFalse(m['manual_labels_used_as_inputs'])
        self.assertEqual(m['action_offset'],0)
        # Positions must vary across demonstrations; this rejects the old synthetic benchmark input.
        features = np.stack([self.dataset.arrays[i]['initial_detections'] for i in range(50)])
        self.assertGreater(float(features[:,:,0:2].std(axis=0).min()), .01)


if __name__ == '__main__': unittest.main()
