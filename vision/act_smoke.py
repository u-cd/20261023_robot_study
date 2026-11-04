"""Compact ACT mechanics check on saved demonstrations; no serial/camera access.

Adaptation, not an exact paper reproduction: frozen ImageNet ResNet18 at 160x120,
128-wide Transformers (2 layers each), 8-step/5Hz absolute commanded-angle chunks.
Original recordings are read-only. Derived features/checkpoints live under runs/.
"""
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
JOINTS = (1, 2, 3, 5)


def angles(state):
    by_id = {j['joint']: j['angle'] for j in state['joints']}
    return np.array([by_id[j] for j in JOINTS], dtype=np.float32)


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def make_samples(folder, chunk=8, hz=5, max_gap=.35):
    """Only jog intervals: interpolate commanded values, never physical positions.

    Split at absent/non-jog records or unobserved gaps > max_gap. Short inter-command
    gaps hold the last recorded command. Images are actual pre-command snapshots;
    missing images are not synthesized. Future targets start one grid step ahead.
    """
    raw = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    raw = [r for r in raw if r.get('event') == 'result']
    segments, current = [], []
    for r in raw:
        if not r.get('accepted') or r.get('control_mode') != 'jog':
            if current: segments.append(current); current = []
            continue
        start = stamp(r['command_sent_at'])
        end = stamp(r['interval_end_at'])
        if end <= start: continue
        if current and start - current[-1][1] > max_gap:
            segments.append(current); current = []
        current.append((start, end, r))
    if current: segments.append(current)
    output = []
    for segment in segments:
        starts = np.array([r[0] for r in segment])
        for start, end, r in segment:
            image_age = start - stamp(r['frame_captured_at'])
            if not 0 <= image_age <= .25: continue
            future = np.zeros((chunk, 4), dtype=np.float32)
            valid = np.zeros(chunk, dtype=bool)
            for j in range(chunk):
                target_time = start + (j+1)/hz
                if target_time > segment[-1][1]: break
                idx = int(np.searchsorted(starts, target_time, side='right')-1)
                a, b, source = segment[idx]
                f = np.clip((target_time-a)/(b-a), 0, 1)
                future[j] = angles(source['state_before'])*(1-f) + angles(source['state_after'])*f
                valid[j] = True
            if valid.sum() < 2: continue
            output.append({'episode': folder.name, 'index': r['index'],
                           'image': str((folder/r['image']).relative_to(ROOT)),
                           'frame_age_s': image_age, 'q': angles(r['state_before']),
                           'target': future, 'valid': valid})
    return output


def positional(length, width):
    p = torch.arange(length).float()[:, None]
    scale = torch.exp(torch.arange(0, width, 2).float()*(-np.log(10000)/width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(p*scale)
    result[:, 1::2] = torch.cos(p*scale)
    return result


class CompactACT(nn.Module):
    def __init__(self, chunk=8, width=128, latent=32):
        super().__init__()
        self.chunk, self.latent = chunk, latent
        self.image_proj = nn.Linear(512, width)
        self.q_proj = nn.Linear(4, width)
        self.action_proj = nn.Linear(4, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width)*.02)
        self.z_stats = nn.Linear(width, latent*2)
        self.z_proj = nn.Linear(latent, width)
        def encoder():
            layer = nn.TransformerEncoderLayer(width, 4, width*4, .1, batch_first=True)
            return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.posterior = encoder()
        self.observation = encoder()
        layer = nn.TransformerDecoderLayer(width, 4, width*4, .1, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.head = nn.Linear(width, 4)
        self.register_buffer('seq_pos', positional(chunk+2, width))
        # Distinct row/column sinusoidal positions for the 4x5 ResNet feature grid.
        image_pos = torch.cat((positional(4, width//2)[:, None].expand(4, 5, -1),
                               positional(5, width//2)[None].expand(4, 5, -1)), -1)
        self.register_buffer('image_pos', image_pos.reshape(20, width))
        self.type_pos = nn.Parameter(torch.randn(1, 2, width)*.02)

    def forward(self, image_features, q, actions=None, valid=None):
        batch = len(q)
        mu = logvar = None
        if actions is not None:
            # Padded targets must not affect posterior tokens or loss.
            actions = actions.masked_fill(~valid[..., None], 0)
            tokens = torch.cat((self.cls.expand(batch, -1, -1), self.q_proj(q)[:, None],
                                self.action_proj(actions)), 1) + self.seq_pos[None]
            mask = torch.cat((torch.zeros(batch, 2, dtype=torch.bool, device=q.device), ~valid), 1)
            encoded = self.posterior(tokens, src_key_padding_mask=mask)[:, 0]
            mu, logvar = self.z_stats(encoded).chunk(2, -1)
            z = mu + torch.exp(.5*logvar)*torch.randn_like(mu)
        else:
            z = q.new_zeros(batch, self.latent)
        state_tokens = torch.stack((self.z_proj(z), self.q_proj(q)), 1) + self.type_pos
        tokens = torch.cat((state_tokens, self.image_proj(image_features)+self.image_pos), 1)
        memory = self.observation(tokens)
        queries = self.seq_pos[2:][None].expand(batch, -1, -1)
        return self.head(self.decoder(queries, memory)), mu, logvar


def objective(pred, target, valid, mu, logvar, beta=10):
    l1 = ((pred-target).abs()*valid[..., None]).sum()/(valid.sum()*target.shape[-1])
    kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
    return l1+beta*kl, l1, kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/act_smoke')
    args = parser.parse_args()
    if args.steps < 1: parser.error('--steps must be positive')
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    started, cpu_started = time.monotonic(), time.process_time()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'runs/imitation_bc/report.json').read_text())['splits']
    sources = sorted((ROOT/'data/imitation').glob('*/samples.jsonl'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    samples = []
    for p in sources: samples.extend(make_samples(p.parent))
    split_idx = {key: np.array([i for i, s in enumerate(samples) if s['episode'] in ids]) for key, ids in split.items()}
    if any(len(ix)==0 for ix in split_idx.values()): raise ValueError('An episode split has no usable jog samples')
    print('Samples:', {k: len(v) for k,v in split_idx.items()}, flush=True)
    weights = ROOT/'models/resnet18-f37072fd.pth'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'): raise ValueError('ResNet weights checksum mismatch')
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    backbone = nn.Sequential(*list(backbone.children())[:-2]).eval().requires_grad_(False)
    metadata = {'weights_sha256': digest, 'images': [s['image'] for s in samples], 'size': [160,120]}
    cache = out/'image_features.pt'
    if cache.exists():
        cached = torch.load(cache, weights_only=True)
        if cached['metadata'] != metadata: raise ValueError('Feature cache mismatch; choose another output directory')
        images = cached['features']
    else:
        pieces = []
        for offset in range(0, len(samples), 16):
            frames = []
            for s in samples[offset:offset+16]:
                frame = cv2.imread(str(ROOT/s['image']))
                if frame is None: raise ValueError('Cannot decode '+s['image'])
                frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(160,120)))
            x = torch.tensor(np.stack(frames)).permute(0,3,1,2).float()/255
            x = (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            with torch.no_grad(): pieces.append(backbone(x).flatten(2).transpose(1,2).contiguous())
            if offset%512==0: print('Image features',offset,'/',len(samples),flush=True)
        images = torch.cat(pieces)
        torch.save({'metadata': metadata, 'features':images},cache)
    del backbone
    q = torch.tensor(np.stack([s['q'] for s in samples]))
    targets = torch.tensor(np.stack([s['target'] for s in samples]))
    valid = torch.tensor(np.stack([s['valid'] for s in samples]))
    train = split_idx['train']; val = split_idx['val']; test = split_idx['test']
    q_mean, q_std = q[train].mean(0), q[train].std(0).clamp_min(1)
    observed = targets[train][valid[train]]
    a_mean, a_std = observed.mean(0), observed.std(0).clamp_min(1)
    normalized_q, normalized_a = (q-q_mean)/q_std, (targets-a_mean)/a_std
    model = CompactACT(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    def predict(indices):
        batches=[]
        with torch.inference_mode():
            for ix in torch.tensor(indices).split(32):
                batches.append(model(images[ix], normalized_q[ix])[0]*a_std+a_mean)
        return torch.cat(batches)
    def mae(pred, truth, mask):
        return float(((pred-truth).abs()*mask[...,None]).sum()/(mask.sum()*4))
    model.eval(); initial_val = mae(predict(val),targets[val],valid[val])
    history=[]; best=initial_val; best_state={k:v.clone() for k,v in model.state_dict().items()}; best_step=0
    for step in range(1,args.steps+1):
        model.train(); ix=torch.tensor(np.random.choice(train,16))
        pred,mu,logvar=model(images[ix],normalized_q[ix],normalized_a[ix],valid[ix])
        loss,l1,kl=objective(pred,normalized_a[ix],valid[ix],mu,logvar)
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        opt.zero_grad();loss.backward()
        grad=nn.utils.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(grad): raise ValueError('Nonfinite gradients')
        opt.step()
        if step==1 or step%25==0 or step==args.steps:
            model.eval(); score=mae(predict(val),targets[val],valid[val])
            history.append({'step':step,'loss':float(loss.detach()),'l1':float(l1.detach()),'kl':float(kl.detach()),'val_mae_deg':score})
            if score<best:
                best=score;best_step=step;best_state={k:v.clone() for k,v in model.state_dict().items()}
            print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(best_state);model.eval()
    pred=predict(test); truth=targets[test]; mask=valid[test]
    abs_error=(pred-truth).abs()
    hold=q[test,None,:].expand_as(truth)
    # Include the same offline constant-command baseline for context.
    metrics={'initial_val_mae_deg':initial_val,'best_val_mae_deg':best,'best_step':best_step,
             'test_mae_deg':mae(pred,truth,mask),'hold_current_test_mae_deg':mae(hold,truth,mask),
             'test_per_joint_mae_deg':(abs_error*mask[...,None]).sum((0,1)).div(mask.sum()).tolist(),
             'test_first_future_mae_deg':mae(pred[:,:1],truth[:,:1],mask[:,:1])}
    probe=int(test[0]);tick=time.perf_counter()
    with torch.inference_mode():
        for _ in range(100): model(images[probe:probe+1],normalized_q[probe:probe+1])
    latency=(time.perf_counter()-tick)*10
    torch.save({'state_dict':best_state,'q_mean':q_mean,'q_std':q_std,'a_mean':a_mean,'a_std':a_std,
                'config':{'chunk':8,'width':128,'latent':32},'joints':list(JOINTS),'backbone_sha256':digest},out/'policy.pt')
    # Verify checkpoint load and z=0 deterministic offline inference independently.
    loaded=torch.load(out/'policy.pt',weights_only=True)
    restored=CompactACT(**loaded['config']).eval();restored.load_state_dict(loaded['state_dict'])
    with torch.inference_mode():
        again=restored(images[probe:probe+1],normalized_q[probe:probe+1])[0]*loaded['a_std']+loaded['a_mean']
    torch.testing.assert_close(again[0],pred[0])
    report={'kind':'compact ACT offline smoke test, not original-paper reproduction or MLP comparison',
            'settings':{'seed':42,'steps':args.steps,'batch':16,'lr':1e-4,'beta':10,'hz':5,'chunk':8,
                        'joints':list(JOINTS),'state':'commanded, not measured','phase_input':False,
                        'backbone':'ImageNet ResNet18 frozen; spatial features 4x5x512; 160x120 input',
                        'transformer':'width128, heads4, posterior/observation/decoder each2layers, FF512, latent32',
                        'gap_rule':'jog only; split if end-to-next-command gap >0.35s; actual image age <=0.25s; >=2 future targets',
                        'interpolation':'linear commanded angle inside recorded interval; hold in short gaps; truncate at segment end',
                        'loss':'valid-element mean L1 + 10 * summed-latent KL; no temporal ensembling in offline test',
                        'normalization':'train split only; std floor 1 degree'},
            'split_episode_ids':split,'usable_episodes':len(set(s['episode'] for s in samples)),
            'samples':{k:len(v) for k,v in split_idx.items()},'source_sha256':hashes,
            'metrics':metrics,'history':history,'checkpoint_reload_verified':True,'policy_only_inference_ms':latency,
            'elapsed_s':time.monotonic()-started,'cpu_s':time.process_time()-cpu_started,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'robot_moved':False,'limitations':['No physical joint sensing or task success measured',
                 'Unknown long gaps and non-jog demonstrations excluded, no images synthesized',
                 'Reduced/frozen backbone and smaller transformer differ from original ACT',
                 'Same episode IDs as MLP split, but inputs/targets/usable samples differ; metrics not comparable']}
    # Plot six actual held-out observations and their future target/prediction chunks.
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/robot-act-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=np.linspace(0,len(test)-1,6,dtype=int)
    fig,axes=plt.subplots(6,2,figsize=(12,18))
    examples=[]
    colors=['tab:blue','tab:orange','tab:green','tab:red']
    for row,position in enumerate(chosen):
        s=samples[int(test[position])]; frame=cv2.imread(str(ROOT/s['image']))
        axes[row,0].imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));axes[row,0].axis('off')
        axes[row,0].set_title(f"{s['episode']} #{s['index']}",fontsize=9)
        n=int(mask[position].sum());t=np.arange(1,n+1)/5
        for j,(name,color) in enumerate(zip(('base','shoulder','elbow','gripper'),colors)):
            axes[row,1].plot(t,truth[position,:n,j],color=color,label=name+' demo')
            axes[row,1].plot(t,pred[position,:n,j],color=color,ls='--',label=name+' ACT')
        axes[row,1].set(xlabel='Future time (s)',ylabel='Commanded angle (deg)');axes[row,1].grid(alpha=.2)
        if row==0:axes[row,1].legend(fontsize=7,ncol=2)
        examples.append({'episode':s['episode'],'sample_index':s['index'],'image':s['image'],
                         'q':s['q'].tolist(),'target_deg':truth[position,:n].tolist(),'predicted_deg':pred[position,:n].tolist()})
    fig.tight_layout();fig.savefig(out/'predictions.png',dpi=110);plt.close(fig)
    (out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    for path,d in hashes.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=d: raise ValueError('Source log changed during test')
    report['source_logs_unchanged']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'viewer.html').write_text('<!doctype html><meta charset="utf-8"><title>ACT offline test</title>'
        '<h1>ACT：保存画像での学習・推論テスト</h1><p>実機は動かしていません。角度は指令値です。実線＝実演、破線＝予測。</p>'
        '<p>小型構成・短時間学習です。搬送成功やMLPより優れることを示す結果ではありません。</p>'
        '<pre>'+html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))+'</pre>'
        '<img style="max-width:100%" src="predictions.png"><p><a href="report.json">設定と結果</a></p>')
    print(json.dumps({'metrics':metrics,'elapsed_s':report['elapsed_s'],'peak_rss_mb':report['peak_rss_mb'],
                      'output':str(out)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
