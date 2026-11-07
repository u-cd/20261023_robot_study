"""Prepare numeric ACT data using only initial YOLO detections; preserve raw records."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/robot-yolo-act-prepare')
import cv2
import h5py
import numpy as np
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]


def box_iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))


def stationary_frames(raw, limit):
    first=None;chosen=[];seen=set()
    with (raw/'samples.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            state=[(j['joint'],j['angle'],j['target']) for j in sorted(row['state']['joints'],key=lambda j:j['joint'])]
            if first is None:first=state
            if state!=first:break
            if row['image'] not in seen:
                chosen.append(row);seen.add(row['image'])
            if len(chosen)==limit:break
    if not chosen:raise ValueError('No initial frames: '+str(raw))
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/act_initial_detections_cpu.json')
    args=p.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['dataset']
    if out.exists():raise SystemExit('Choose a new output dataset; refusing overwrite.')
    out.mkdir(parents=True);(out/'review').mkdir()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model=YOLO(str(ROOT/cfg['detector']['weights']))
    source=ROOT/cfg['source_dataset'];manifest=json.loads((source/'manifest.json').read_text())
    labels=ROOT/'data/initial_object_labels'
    label_manifest=json.loads((labels/'manifest.json').read_text())
    label_map={e['episode']:e for e in label_manifest['images']}
    results=[];problems=[];tiles=[]
    for episode in manifest['episodes']:
        eid=episode['episode_id'];raw=ROOT/'data/act_demonstrations_raw'/episode['source']
        entry=label_map[episode['source']];annotation=json.loads((labels/'annotations'/f"{entry['id']}.json").read_text())
        if not annotation.get('reviewed'):raise ValueError('Unreviewed reference: '+entry['id'])
        rows=stationary_frames(raw,cfg['detector']['initial_stationary_frames']);samples=[];frames=[];image=None
        if len(rows)!=cfg['detector']['initial_stationary_frames']:problems.append([eid,'fewer than five stationary frames'])
        for row in rows:
            path=raw/row['image'];im=cv2.imread(str(path))
            if im is None:raise ValueError('Cannot decode '+str(path))
            if image is None:image=im
            pred=model.predict(im,imgsz=cfg['detector']['imgsz'],device='cpu',conf=cfg['detector']['confidence_threshold'],verbose=False)[0]
            chosen=[]
            for label in cfg['detector']['class_order']:
                options=[b for b in pred.boxes if model.names[int(b.cls.item())]==label]
                if not options:raise ValueError(f'Episode {eid}, frame {row["image"]}: missing {label}')
                b=max(options,key=lambda b:float(b.conf.item()))
                xyxy=b.xyxy[0].cpu().numpy();chosen.append([*xyxy,float(b.conf.item())])
            samples.append(chosen)
            frames.append(dict(image=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               frame_sequence=row['frame_sequence'],frame_captured_ns=row['frame_captured_ns'],detections=chosen))
        samples=np.asarray(samples,dtype=np.float32);aggregate=np.median(samples,axis=0)
        centers=(samples[:,:,:2]+samples[:,:,2:4])/2
        spreads=np.linalg.norm(centers-np.median(centers,axis=0),axis=-1).max(axis=0)
        if float(spreads.max())>cfg['detector']['max_center_spread_px']:problems.append([eid,'unstable initial detections',spreads.tolist()])
        h,w=image.shape[:2];features=[];ious={}
        visual=image.copy()
        for ci,label in enumerate(cfg['detector']['class_order']):
            x,y,x2,y2,score=map(float,aggregate[ci])
            features.append([(x+x2)/2/w,(y+y2)/2/h,(x2-x)/w,(y2-y)/h,score,1.])
            ious[label]=box_iou([x,y,x2,y2],annotation['boxes'][label])
            if ious[label]<.5:problems.append([eid,'reference IoU below 0.5',label,ious[label]])
            color=(255,100,0) if ci==0 else (0,150,255)
            cv2.rectangle(visual,(round(x),round(y)),(round(x2),round(y2)),color,2)
            cv2.putText(visual,f'{label} {score:.2f}',(round(x),max(15,round(y)-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        with h5py.File(source/episode['file'],'r') as f:
            if int(f.attrs['action_offset'])!=0:raise ValueError('Expected aligned action_offset=0')
            qpos=f['observations/qpos'][:].astype(np.float32);actions=f['action'][:].astype(np.float32)
        if qpos.shape!=actions.shape or qpos.shape[1]!=5 or not np.isfinite(qpos).all() or not np.isfinite(actions).all():raise ValueError('Invalid numeric data')
        dest=out/f'episode_{eid}.npz'
        np.savez_compressed(dest,qpos=qpos,action=actions,initial_detections=np.asarray(features,dtype=np.float32))
        results.append(dict(episode_id=eid,source=episode['source'],file=dest.name,steps=len(qpos),
                            split=episode['split'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                            source_hdf5_sha256=episode['sha256'],frames=frames,features=features,
                            initial_boxes_xyxy=aggregate[:,:4].tolist(),reference_iou=ious,max_center_spread_px=spreads.tolist()))
        tile=cv2.resize(visual,(320,240));cv2.rectangle(tile,(0,0),(320,25),(255,255,255),-1)
        cv2.putText(tile,f'{eid:02d} {episode["split"]} initial x{len(rows)} median',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);tiles.append(tile)
        print('PREPARED',eid,'IoU',ious,'spread',spreads.tolist(),flush=True)
    for start in [0,25]:
        cv2.imwrite(str(out/'review'/f'initial_detections_{start//25+1}.jpg'),np.concatenate([np.concatenate(tiles[i:i+5],axis=1) for i in range(start,start+25,5)],axis=0))
    # Match the official all-frame, all-episode mean/sample-std convention.
    arrays=[np.load(out/r['file']) for r in results]
    stats={}
    for data_key,prefix in [('qpos','qpos'),('action','action')]:
        tensor=torch.from_numpy(np.concatenate([a[data_key] for a in arrays]))
        stats[prefix+'_mean']=tensor.mean(0).numpy();stats[prefix+'_std']=tensor.std(0).clamp(min=.01).numpy()
    for a in arrays:a.close()
    np.savez(out/'normalization.npz',**stats)
    output=dict(status='ready' if not problems else 'needs_review',config=cfg,episodes=results,problems=problems,
                train_ids_in_official_order=manifest['train_ids_in_official_order'],
                validation_ids_in_official_order=manifest['validation_ids_in_official_order'],
                detector_sha256=hashlib.sha256((ROOT/cfg['detector']['weights']).read_bytes()).hexdigest(),
                source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                initial_detections_held_fixed=True,manual_labels_used_as_inputs=False,
                manual_labels_usage='audit only; predictions are never replaced with ground truth',
                hz=20,action_offset=0,total_steps=sum(r['steps'] for r in results))
    # Convert numpy scalars in per-frame details to plain JSON numbers.
    (out/'manifest.json').write_text(json.dumps(output,indent=2,default=lambda x:float(x))+'\n')
    print('RESULT',output['status'],'problems',problems,flush=True)
    if problems:raise SystemExit('Do not train until audit problems are resolved.')


if __name__=='__main__':main()
