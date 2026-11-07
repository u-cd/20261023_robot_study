"""Initial-detection ACT: official CVAE/Transformer with two object tokens."""
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / '.build/initial_detection_act/act'


def configure_official(output, epochs):
    os.environ.update(TORCH_HOME=str(ROOT / 'models/torch'), MPLBACKEND='Agg',
                      MPLCONFIGDIR='/tmp/act-initial-detection-mpl')
    if not (OFFICIAL / 'policy.py').is_file():
        raise RuntimeError('小型ACT用の公式コードの別コピーを準備してください。experiments/initial_detection_act/README.md を参照。')
    sys.path[:0] = [str(OFFICIAL), str(OFFICIAL / 'detr')]
    # Upstream model factory parses the entrypoint's argv before applying overrides.
    sys.argv = ['detection-act', '--ckpt_dir', str(output), '--policy_class', 'ACT',
                '--task_name', 'adeept_sponge', '--seed', '0', '--num_epochs', str(epochs),
                '--batch_size', '8', '--lr', '0.00001', '--device', 'cpu', '--state_dim', '5']


class InitialDetectionDataset(Dataset):
    def __init__(self, directory, ids, stats, chunk_size=100):
        self.ids, self.stats, self.chunk_size = list(ids), stats, chunk_size
        self.arrays = {}
        for episode_id in self.ids:
            with np.load(Path(directory) / f'episode_{episode_id}.npz') as f:
                self.arrays[episode_id] = {k: f[k].copy() for k in f.files}

    def __len__(self):
        return len(self.ids)

    def sample_at(self, episode_id, t):
        arrays = self.arrays[episode_id]
        actions = arrays['action']
        if not 0 <= t < len(actions):
            raise IndexError(t)
        n = min(self.chunk_size, len(actions) - t)
        target = np.zeros((self.chunk_size, 5), dtype=np.float32)
        target[:n] = actions[t:t+n]
        return (torch.from_numpy(arrays['initial_detections'].copy()),
                torch.from_numpy((arrays['qpos'][t] - self.stats['qpos_mean']) / self.stats['qpos_std']),
                torch.from_numpy((target - self.stats['action_mean']) / self.stats['action_std']),
                torch.from_numpy(np.arange(self.chunk_size) >= n))

    def __getitem__(self, index):
        episode_id = self.ids[index]
        return self.sample_at(episode_id, np.random.choice(len(self.arrays[episode_id]['qpos'])))


class DetectionACTPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        from policy import ACTPolicy
        from initial_detection_adapter import DetectionACT
        original = ACTPolicy(config)
        self.adapter = DetectionACT(original.model)
        self.kl_weight = config['kl_weight']
        self.chunk_size = config['num_queries']
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['lr'], weight_decay=1e-4)

    def forward(self, qpos, detections, actions=None, is_pad=None):
        if actions is None:
            # Official model uses the CVAE prior mean z=0 at inference.
            return self.adapter(qpos, detections)[0]
        from policy import kl_divergence
        actions, is_pad = actions[:, :self.chunk_size], is_pad[:, :self.chunk_size]
        predicted, _, (mu, logvar) = self.adapter(qpos, detections, actions, is_pad)
        kl = kl_divergence(mu, logvar)[0][0]
        # Preserve upstream denominator, including masked padding slots.
        l1 = (F.l1_loss(actions, predicted, reduction='none') * ~is_pad.unsqueeze(-1)).mean()
        return dict(l1=l1, kl=kl, loss=l1 + self.kl_weight * kl)

    def configure_optimizers(self):
        return self.optimizer


def load_inference_bundle(path):
    """Load trusted local output for offline inference. No serial connection."""
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    configure_official(Path(path).parent, 1)
    policy = DetectionACTPolicy(bundle['config']['policy'])
    policy.load_state_dict(bundle['state_dict'], strict=True)
    policy.eval()
    return policy, bundle['stats'], bundle['config']
