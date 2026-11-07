from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from act import utils


class VariableLengthTests(unittest.TestCase):
    def write_episode(self, folder, index, length):
        values = np.arange(length * 5, dtype=np.float32).reshape(length, 5) / 100
        with h5py.File(Path(folder) / f'episode_{index}.hdf5', 'w') as f:
            f.attrs.update(sim=False, action_offset=0)
            f['observations/qpos'] = values
            f['observations/qvel'] = np.zeros_like(values)
            f['action'] = values + .1
            f['observations/images/top'] = np.zeros((length, 2, 2, 3), dtype=np.uint8)
        return values

    def test_equal_length_matches_upstream_first_chunk_and_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = [self.write_episode(tmp, i, 120) for i in range(2)]
            stats = utils.get_norm_stats(tmp, 2)
            stacked = torch.from_numpy(np.stack(values))
            np.testing.assert_allclose(stats['qpos_mean'], stacked.mean((0, 1)).numpy(), rtol=1e-6)
            np.testing.assert_allclose(stats['qpos_std'], stacked.std((0, 1)).numpy(), rtol=1e-6)
            old = utils.EpisodicDataset([0], tmp, ['top'], stats)
            new = utils.EpisodicDataset([0], tmp, ['top'], stats, 100)
            for start in (0, 40, 119):
                with patch.object(utils.np.random, 'choice', return_value=start):
                    a, b = old[0], new[0]
                for i in (0, 1):
                    torch.testing.assert_close(a[i], b[i])
                for i in (2, 3):
                    torch.testing.assert_close(a[i][:100], b[i])

    def test_variable_length_batch_and_end_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = [self.write_episode(tmp, i, n) for i, n in enumerate((120, 175))]
            stats = utils.get_norm_stats(tmp, 2)
            np.testing.assert_allclose(stats['qpos_mean'], np.concatenate(values).mean(0), rtol=1e-6)
            ds = utils.EpisodicDataset([0, 1], tmp, ['top'], stats, 100)
            batch = next(iter(DataLoader(ds, batch_size=2)))
            self.assertEqual(tuple(batch[2].shape), (2, 100, 5))
            with patch.object(utils.np.random, 'choice', return_value=174):
                _, _, actions, mask = ds[1]
            self.assertFalse(mask[0])
            self.assertTrue(mask[1:].all())
            actual = actions[0].numpy() * stats['action_std'] + stats['action_mean']
            np.testing.assert_allclose(actual, values[1][-1] + .1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
