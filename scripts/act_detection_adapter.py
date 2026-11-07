"""Experimental detection-token input adapter for the official ACT model.

Vendor code is not modified. The original DETRVAE forward, CVAE, Transformer,
queries and heads are retained. This is a new input representation and requires
training; it cannot use an image-trained checkpoint as a working robot policy.
"""
import copy

import torch
from torch import nn


class DetectionTokens(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        # Shared projection; class-specific position embeddings identify sponge/coin.
        self.projection = nn.Linear(6, hidden_dim)
        self.class_position = nn.Embedding(2, hidden_dim)

    def forward(self, packed):
        # packed: [B, 6, 1, 2]; two object tokens, not a camera image.
        detections = packed[:, :, 0, :].transpose(1, 2)
        features = self.projection(detections).transpose(1, 2).unsqueeze(2)
        # Official Transformer broadcasts a batch-1 position tensor internally.
        positions = self.class_position.weight.T[None, :, None, :]
        return [features], [positions]


class DetectionACT(nn.Module):
    def __init__(self, official_image_model):
        super().__init__()
        self.model = copy.deepcopy(official_image_model)
        self.model.backbones = nn.ModuleList([DetectionTokens(self.model.transformer.d_model)])
        self.model.input_proj = nn.Identity()

    def forward(self, qpos, detections, actions=None, is_pad=None):
        # detections: [B, 2, 6], ordered sponge, coin.
        # Each row: cx/W, cy/H, width/W, height/H, confidence, valid.
        packed = detections.transpose(1, 2).unsqueeze(2).unsqueeze(1)
        return self.model(qpos, packed, None, actions, is_pad)
