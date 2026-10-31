"""Small feature-based BC policy. No robot I/O during model loading/prediction."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from control.imitation_record import ACTIONS, PHASES
from vision.prepare_imitation import FEATURE_NAMES, features


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(len(FEATURE_NAMES),64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.action = nn.Linear(64,len(ACTIONS))
        self.amount = nn.Linear(64,1)

    def forward(self, x):
        x = self.body(x)
        return self.action(x), self.amount(x).squeeze(-1)


class ImitationPolicy:
    def __init__(self, path):
        self.path = Path(path)
        checkpoint = torch.load(self.path,map_location='cpu',weights_only=True)
        if checkpoint['feature_names']!=FEATURE_NAMES or checkpoint['phases']!=PHASES:
            raise ValueError('モデルの入力・段階の定義が一致しません。')
        self.model = PolicyNet()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.mean = np.array(checkpoint['mean'],dtype=np.float32)
        self.scale = np.array(checkpoint['scale'],dtype=np.float32)
        self.bounds = checkpoint['joint_bounds']

    def predict(self, detections, state, phase):
        x = np.array(features(detections,state,phase),dtype=np.float32)
        with torch.inference_mode():
            logits, amount = self.model(torch.from_numpy((x-self.mean)/self.scale).unsqueeze(0))
            probabilities = logits.softmax(-1)[0]
            label = int(probabilities.argmax())
        action = ACTIONS[label]
        joint = None if label==8 else (1,2,3,5)[label//2]
        return {'action':action,'label':label,'joint':joint,
                'direction':None if label==8 else ('+' if label%2==0 else '-'),
                'confidence':float(probabilities[label]),
                'predicted_amount_deg':float(max(0,min(6,amount.item()*4))),
                'probabilities':probabilities.tolist(),'phase':phase,
                'within_joint_training_bounds':all(
                    self.bounds[str(j['joint'])][0]-3<=j['angle']<=self.bounds[str(j['joint'])][1]+3
                    for j in state['joints'] if j['joint'] in (1,2,3,5))}
