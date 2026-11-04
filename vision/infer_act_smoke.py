"""Load the compact ACT checkpoint and infer from one saved raw image/state. No robot I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vision.act_smoke import CompactACT, angles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True)
    parser.add_argument('--index',required=True,type=int)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/act_smoke/policy.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/act_smoke/saved_image_inference.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    folder=(ROOT/'data/imitation'/args.episode).resolve()
    if folder.parent != (ROOT/'data/imitation').resolve():raise ValueError('Invalid episode path')
    rows=[json.loads(l) for l in (folder/'samples.jsonl').read_text().splitlines()]
    sample=next(r for r in rows if r.get('event')=='result' and r['index']==args.index and r['accepted'])
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    weights=ROOT/'models/resnet18-f37072fd.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=checkpoint['backbone_sha256']:raise ValueError('Wrong image backbone')
    resnet=resnet18(weights=None)
    resnet.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    backbone=nn.Sequential(*list(resnet.children())[:-2]).eval()
    model=CompactACT(**checkpoint['config']).eval();model.load_state_dict(checkpoint['state_dict'])
    frame=cv2.imread(str(folder/sample['image']))
    if frame is None:raise ValueError('Image decode failed')
    x=torch.tensor(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(160,120))).permute(2,0,1)[None].float()/255
    x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    q=torch.tensor(angles(sample['state_before']))[None]
    tick=time.perf_counter()
    with torch.inference_mode():
        features=backbone(x).flatten(2).transpose(1,2)
        action=model(features,(q-checkpoint['q_mean'])/checkpoint['q_std'])[0]
        degrees=action*checkpoint['a_std']+checkpoint['a_mean']
    if not torch.isfinite(degrees).all():raise ValueError('Nonfinite prediction')
    result={'episode':args.episode,'sample_index':args.index,'image':sample['image'],
            'joints':checkpoint['joints'],'current_command_deg':q[0].tolist(),
            'future_times_s':[(i+1)/5 for i in range(model.chunk)],
            'predicted_command_deg':degrees[0].tolist(),'z':0,'robot_moved':False,
            'image_and_policy_inference_ms':(time.perf_counter()-tick)*1000,
            'note':'Raw predictions only. No command was sent; not a certified reachable/safe trajectory.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
