"""The single ACT model used for training and inference.

Network computation is derived from ACT/DETR; no upstream runtime imports.
Existing ACTPolicy checkpoints keep their model.* state-dict keys.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize

from .detr.models import build_ACT_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/act.json'


def load_policy_config(checkpoint_dir=None):
    snapshot = Path(checkpoint_dir)/'policy_config.json' if checkpoint_dir is not None else None
    return json.loads(snapshot.read_text()) if snapshot is not None and snapshot.exists() else json.loads(CONFIG.read_text())['policy']


class ACTPolicy(nn.Module):
    def __init__(self, config, pretrained_backbone=False):
        super().__init__()
        os.environ.setdefault('TORCH_HOME', str(ROOT/'models/torch'))
        self.config = dict(config)
        defaults = dict(dilation=False, masks=False, position_embedding='sine', pre_norm=False, dropout=0.1)
        defaults.update(config)
        defaults['pretrained'] = pretrained_backbone
        self.model = build_ACT_model(SimpleNamespace(**defaults))
        self.kl_weight = config['kl_weight']

    def forward(self, qpos, image, actions=None, is_pad=None):
        image = Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(image)
        if actions is None:
            return self.model(qpos, image, None)[0]
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]
        predicted, _, (mu, logvar) = self.model(qpos, image, None, actions, is_pad)
        kl = kl_divergence(mu, logvar)[0][0]
        l1 = (F.l1_loss(actions, predicted, reduction='none') * ~is_pad.unsqueeze(-1)).mean()
        return dict(l1=l1, kl=kl, loss=l1 + self.kl_weight*kl)


def make_optimizer(model, config):
    groups = [dict(params=[p for n,p in model.model.named_parameters() if 'backbone' not in n and p.requires_grad]),
              dict(params=[p for n,p in model.model.named_parameters() if 'backbone' in n and p.requires_grad], lr=config['lr_backbone'])]
    return torch.optim.AdamW(groups, lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
