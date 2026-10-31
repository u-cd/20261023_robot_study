"""Fit base terminal angle from demonstrated object image coordinates, by episode split."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    report=json.loads((ROOT/'runs/imitation_bc/report.json').read_text())
    models={}
    for phase,name in [(0,'sponge'),(1,'target')]:
        samples={}
        for episode in report['manifest']:
            folder=ROOT/'data/imitation'/episode['id']
            data=json.loads((folder/'features.json').read_text())
            rows=[r for r in data['samples'] if r['phase']==phase and name in r['detections'] and r['detections'][name]['confidence']>=.4]
            raw=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
            raw=[r for r in raw if r['event']=='result' and r['accepted'] and r['phase']==phase]
            if rows and raw:
                samples[episode['id']]=(np.median([r['detections'][name]['center_normalized'] for r in rows[:3]],axis=0),
                    next(j['angle'] for j in raw[-1]['state_after']['joints'] if j['joint']==1))
        train=[samples[e] for e in report['splits']['train'] if e in samples]
        x=np.array([s[0] for s in train]);y=np.array([s[1] for s in train]);mu=x.mean(0);sd=np.maximum(x.std(0),.001)
        candidates=[]
        for quadratic in (False,True):
            def design(x):
                z=(np.asarray(x)-mu)/sd
                return np.column_stack([np.ones(len(z)),z]+([z[:,0]*z[:,1],z[:,0]**2,z[:,1]**2] if quadratic else []))
            a=design(x);reg=np.eye(a.shape[1])*.1;reg[0,0]=0
            weights=np.linalg.solve(a.T@a+reg,a.T@y)
            metrics={}
            for split in ('train','val','test'):
                entries=[samples[e] for e in report['splits'][split] if e in samples]
                errors=design([v[0] for v in entries])@weights-np.array([v[1] for v in entries])
                metrics[split]={'count':len(entries),'mae_deg':float(np.abs(errors).mean()),'max_deg':float(np.abs(errors).max())}
            candidates.append(dict(object=name,quadratic=quadratic,mean=mu.tolist(),scale=sd.tolist(),weights=weights.tolist(),
                uv_min=x.min(0).tolist(),uv_max=x.max(0).tolist(),metrics=metrics))
        selected=min(candidates,key=lambda c:c['metrics']['val']['mae_deg'])
        selected['candidate_validation_mae']=[{'quadratic':c['quadratic'],'mae_deg':c['metrics']['val']['mae_deg']} for c in candidates]
        selected['training_episode_ids']=[e for e in report['splits']['train'] if e in samples]
        models[str(phase)]=selected
    result={'schema_version':1,'tolerance_deg':4,'ridge':.1,'models':models,
        'description':'Separate learned base alignment guide; filters BC actions, not a retrained BC model. Phase 2 unchanged.',
        'selection':'linear vs quadratic selected by validation MAE only; original test episodes held out'}
    path=ROOT/'runs/imitation_bc/base_alignment.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for phase,m in models.items():print('phase',phase,'quadratic',m['quadratic'],'metrics',m['metrics'])


if __name__=='__main__':main()
