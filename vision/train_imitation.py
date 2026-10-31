"""Extract detections, train small BC model, evaluate by held-out episodes. CPU only; no robot I/O."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/robot-yolo-config')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from control.imitation_policy import PolicyNet
from control.imitation_record import ACTIONS, PHASES
from vision.prepare_imitation import FEATURE_NAMES, extract_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/imitation')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/imitation_bc')
    parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cpu_start=time.process_time()
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    weights=ROOT/'runs/sponge_comparison/yolo11n/weights/best.pt'
    sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    folders=[p.parent for p in sorted(args.data.glob('*/episode.json'))
             if json.loads(p.read_text()).get('outcome')=='success']
    model=None;episodes={};manifest=[]
    for folder in folders:
        path=folder/'features.json'
        data=json.loads(path.read_text()) if path.exists() else None
        if not data or data.get('weights_sha256')!=sha or any('delta_deg' not in r for r in data['samples']):
            if model is None:
                from ultralytics import YOLO
                model=YOLO(weights)
            print('Extract:',folder.name,flush=True)
            extract_episode(folder,weights,model)
            data=json.loads(path.read_text())
        if data['feature_names']!=FEATURE_NAMES or data['phases']!=PHASES:
            raise ValueError('入力定義が異なるエピソードです: '+folder.name)
        # Samples with no commanded change add no directional demonstration.
        rows=[r for r in data['samples'] if r['label']==8 or abs(r.get('delta_deg') or 0)>0]
        episodes[folder.name]=rows
        manifest.append({'id':folder.name,'features_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(rows)})
        print('Ready:',folder.name,len(rows),flush=True)
    if len(episodes)<5:raise ValueError('独立した分割には5試行以上必要です。')
    ids=sorted(episodes);random.Random(42).shuffle(ids)
    n=max(1,round(len(ids)*.15));splits={'test':ids[:n],'val':ids[n:2*n],'train':ids[2*n:]}
    def load(split):
        rows=[dict(r,episode_id=e) for e in splits[split] for r in episodes[e]]
        return rows,np.array([r['input'] for r in rows],dtype=np.float32),np.array([r['label'] for r in rows]),np.array([abs(r['delta_deg'] or 0)/4 for r in rows],dtype=np.float32)
    loaded={k:load(k) for k in splits}
    train_rows,x,y,a=loaded['train'];mean=x.mean(0);scale=x.std(0);scale[scale<.05]=.05
    tensors={k:(torch.tensor((v[1]-mean)/scale),torch.tensor(v[2]),torch.tensor(v[3])) for k,v in loaded.items()}
    net=PolicyNet();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    ce=nn.CrossEntropyLoss();mse=nn.SmoothL1Loss()
    best=None;best_loss=float('inf');best_epoch=0;history=[]
    for epoch in range(1,args.epochs+1):
        net.train();order=torch.randperm(len(x))
        for ix in order.split(128):
            logits,amount=net(tensors['train'][0][ix]);loss=ce(logits,tensors['train'][1][ix])+.5*mse(amount,tensors['train'][2][ix])
            opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.inference_mode():
            logits,amount=net(tensors['val'][0]);val_loss=float(ce(logits,tensors['val'][1])+.5*mse(amount,tensors['val'][2]))
            acc=float((logits.argmax(-1)==tensors['val'][1]).float().mean())
        history.append({'epoch':epoch,'val_loss':val_loss,'val_accuracy':acc})
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch;best=copy.deepcopy(net.state_dict())
        if epoch%25==0:print('Epoch',epoch,'val acc',round(acc,4),'best',best_epoch,flush=True)
        if epoch-best_epoch>=60:break
    net.load_state_dict(best);net.eval()
    bounds={str(j):[min(r['input'][8+i]*180 for r in train_rows),max(r['input'][8+i]*180 for r in train_rows)] for i,j in enumerate((1,2,3,5))}
    torch.save({'state_dict':best,'mean':mean.tolist(),'scale':scale.tolist(),'feature_names':FEATURE_NAMES,
                'phases':PHASES,'actions':ACTIONS,'joint_bounds':bounds,'best_epoch':best_epoch},args.output/'policy.pt')
    results={};predictions=[]
    for split,(xx,yy,aa) in tensors.items():
        with torch.inference_mode():logits,amount=net(xx)
        probs=logits.softmax(-1).numpy();pred=probs.argmax(-1);amounts=np.clip(amount.numpy()*4,0,6)
        confusion=np.zeros((9,9),dtype=int)
        for truth,predicted in zip(yy.numpy(),pred):confusion[truth,predicted]+=1
        # One nearest training feature vector is a simple non-neural baseline.
        near=[]
        for row in xx.numpy():near.append(y[np.square(tensors['train'][0].numpy()-row).sum(1).argmin()])
        results[split]={'episodes':len(splits[split]),'samples':len(yy),'action_accuracy':float(np.mean(pred==yy.numpy())),
            'nearest_neighbor_accuracy':float(np.mean(np.array(near)==yy.numpy())),
            'majority_accuracy':float(np.mean(yy.numpy()==Counter(y.tolist()).most_common(1)[0][0])),
            'amount_mae_deg':float(np.mean(np.abs(amounts-aa.numpy()*4))), 'confusion':confusion.tolist(),
            'per_phase':{PHASES[p]:float(np.mean(pred[[r['phase']==p for r in loaded[split][0]]]==yy.numpy()[[r['phase']==p for r in loaded[split][0]]])) for p in range(3)}}
        for row,pr,amount,prob in zip(loaded[split][0],pred,amounts,probs):
            predictions.append(dict(episode_id=row['episode_id'],image=row['image'],sample_index=row['sample_index'],phase=row['phase'],
                split=split,truth=ACTIONS[row['label']],predicted=ACTIONS[pr],predicted_amount_deg=float(amount),truth_delta_deg=row['delta_deg'],confidence=float(prob[pr])))
    probe=tensors['test'][0][:1]
    with torch.inference_mode():
        tick=time.perf_counter()
        for _ in range(1000):net(probe)
        latency=(time.perf_counter()-tick)*1000/1000
    report={'created_at':datetime.now(timezone.utc).isoformat(),'seed':42,'architecture':'15 -> 64 ReLU -> 64 ReLU -> action logits(9) + amount(1)',
        'settings':{'epochs_max':args.epochs,'epochs_ran':epoch,'best_epoch':best_epoch,'batch':128,'optimizer':'AdamW','lr':.001,'weight_decay':.001,
                    'loss':'CE + 0.5 SmoothL1(abs(delta_deg)/4)','threads':2,'device':'cpu','split':'episode-level 70/15/15 approx; placement independence unknown'},
        'splits':splits,'manifest':manifest,'metrics':results,'history':history,'inference_ms':latency,
        'elapsed_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu_start,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'robot_moved':False,'limitations':['offline action accuracy is not physical task success','phase chosen by human','no observed wait examples may prevent learned stopping','step and variable-duration jog mixed; delta regressed but runtime starts with bounded steps']}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.output/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2)+'\n')
    cards=[]
    for r in predictions:
        if r['split']!='test':continue
        rel=os.path.relpath(args.data/r['episode_id']/r['image'],args.output)
        cards.append('<figure style="display:inline-block;width:270px;vertical-align:top"><img loading="lazy" width="260" src="'+html.escape(rel)+'"><figcaption>'+html.escape(f"{r['episode_id']} #{r['sample_index']} {PHASES[r['phase']]} / 教師:{r['truth']} {r['truth_delta_deg']}° / 予測:{r['predicted']} {r['predicted_amount_deg']:.1f}°")+'</figcaption></figure>')
    (args.output/'viewer.html').write_text('<!doctype html><meta charset="utf-8"><title>模倣学習の評価</title><h1>未使用試行の教師と予測</h1><p>実機成功率ではありません。</p>'+''.join(cards))
    print(json.dumps({'metrics':results,'inference_ms':latency,'best_epoch':best_epoch,'resources':{'elapsed_s':report['elapsed_s'],'peak_rss_mb':report['peak_rss_mb']}},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
