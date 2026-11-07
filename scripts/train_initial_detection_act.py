"""Train CPU ACT on frozen initial YOLO detections; never controls hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from detection_act import ROOT, configure_official, DetectionACTPolicy, InitialDetectionDataset


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def evaluate(policy, dataset, stats, output):
    """Deterministic teacher-forced, prior-z=0 evaluation; not robot success."""
    import matplotlib.pyplot as plt
    errors, baseline, first_errors, per_episode = [], [], [], []
    timings = []
    policy.eval()
    std = torch.from_numpy(stats['action_std'])
    mean = torch.from_numpy(stats['action_mean'])
    with torch.inference_mode():
        for episode_id in dataset.ids:
            arrays = dataset.arrays[episode_id]
            starts = np.unique(np.linspace(0, len(arrays['qpos'])-1, 64).astype(int))
            episode_errors = []
            for t in starts:
                det, qpos, action, mask = dataset.sample_at(episode_id, t)
                started = time.perf_counter()
                prediction = policy(qpos[None], det[None])[0]
                timings.append(time.perf_counter() - started)
                pred_radians = prediction * std + mean
                true_radians = action * std + mean
                valid = ~mask
                err = (pred_radians[valid] - true_radians[valid]).abs().numpy() * 180 / np.pi
                hold = np.abs(arrays['qpos'][t] - true_radians[valid].numpy()) * 180 / np.pi
                errors.append(err); baseline.append(hold); first_errors.append(err[0]); episode_errors.append(err)
            per_episode.append(dict(episode_id=episode_id, starts=len(starts),
                                    chunk_mae_deg=np.concatenate(episode_errors).mean(axis=0).tolist()))
            # One representative future chunk, all joints, from a recorded middle state.
            t = int(starts[len(starts)//2])
            det, qpos, action, mask = dataset.sample_at(episode_id, t)
            pred = (policy(qpos[None], det[None])[0] * std + mean).numpy() * 180 / np.pi
            truth = (action * std + mean).numpy() * 180 / np.pi
            fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
            for j, name in enumerate(['base', 'shoulder', 'elbow', 'wrist', 'gripper']):
                n = int((~mask).sum())
                axes[j].plot(np.arange(n)/20, truth[:n,j], label='recorded action')
                axes[j].plot(np.arange(n)/20, pred[:n,j], label='ACT prior z=0')
                axes[j].set_ylabel(name + ' (deg)')
            axes[0].legend(); axes[-1].set_xlabel('future seconds')
            fig.suptitle(f'Validation episode {episode_id}, recorded start step {t}')
            fig.tight_layout(); fig.savefig(output / f'validation_episode_{episode_id}.png'); plt.close(fig)
    result = dict(mode='offline teacher-forced qpos, frozen initial detections, CVAE prior z=0',
                  joint_order=['base','shoulder','elbow','wrist','gripper'],
                  chunk_mae_deg=np.concatenate(errors).mean(axis=0).tolist(),
                  hold_current_qpos_chunk_mae_deg=np.concatenate(baseline).mean(axis=0).tolist(),
                  first_step_mae_deg=np.stack(first_errors).mean(axis=0).tolist(),
                  inference_median_ms=float(np.median(timings[5:])*1000),
                  inference_p95_ms=float(np.percentile(timings[5:],95)*1000),
                  per_episode=per_episode, robot_moved=False,
                  limitation='Validation episodes select checkpoints. Not an independent test or closed-loop success rate.')
    write_json(output / 'evaluation.json', result)
    (output / 'evaluation.md').write_text('# 初期検出値ACTのオフライン評価\n\n'
        '記録済みの関節角度を入力する評価です。実機での搬送成功率ではありません。\n\n'
        '|関節|ACT MAE (度)|現在角度維持 MAE (度)|\n|---|---:|---:|\n' +
        ''.join(f'|{name}|{a:.3f}|{b:.3f}|\n' for name,a,b in zip(result['joint_order'],result['chunk_mae_deg'],result['hold_current_qpos_chunk_mae_deg'])) +
        '\n' + '\n'.join(f'![episode {i}](validation_episode_{i}.png)\n' for i in dataset.ids))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/act_initial_detections_cpu.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, help='Smoke override; production uses configuration')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    epochs = config['training']['epochs']
    if epochs < 1: parser.error('epochs must be positive')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    configure_official(output, epochs)
    torch.set_num_threads(config['training']['cpu_threads'])
    torch.set_num_interop_threads(config['training']['interop_threads'])
    import imitate_episodes as official
    import matplotlib.pyplot as plt
    data = ROOT/config['dataset']
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['status'] != 'ready': raise ValueError('Dataset audit did not pass')
    for entry in manifest['episodes']:
        if hashlib.sha256((data/entry['file']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Dataset changed: {entry['file']}")
    with np.load(data/'normalization.npz') as f:
        stats = {k:f[k].copy() for k in f.files}
    with (output/'dataset_stats.pkl').open('wb') as f: pickle.dump(stats, f)
    write_json(output/'config.json', config)
    write_json(output/'dataset_manifest.json', manifest)
    provenance = {'vendor_commit':subprocess.check_output(['git','-C',str(ROOT/'reference/act'),'rev-parse','HEAD'],text=True).strip(),
                  'vendor_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT/'reference/act'),'diff','HEAD'])).hexdigest(),
                  'torch':torch.__version__, 'robot_moved':False,
                  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),ROOT/'scripts/detection_act.py',ROOT/'scripts/act_detection_adapter.py',
                     ROOT/'scripts/prepare_initial_detection_act.py', ROOT/'reference/act/imitate_episodes.py',
                     ROOT/'reference/act/policy.py', ROOT/'reference/act/detr/models/detr_vae.py',
                     ROOT/'reference/act/detr/models/transformer.py']}}
    write_json(output/'provenance.json', provenance)
    train = InitialDetectionDataset(data, manifest['train_ids_in_official_order'], stats)
    val = InitialDetectionDataset(data, manifest['validation_ids_in_official_order'], stats)
    assert not set(train.ids) & set(val.ids)
    progress = {'state':'starting','epochs_total':epochs,'epochs_completed':0,'updates_completed':0}
    started = time.perf_counter(); policy_holder = []
    def factory(policy_class, policy_config):
        if policy_class != 'ACT': raise ValueError(policy_class)
        p = DetectionACTPolicy(policy_config); policy_holder.append(p)
        progress['parameters'] = sum(x.numel() for x in p.parameters())
        return p
    official.make_policy = factory
    original_plot = official.plot_history
    def plot(*a, **kw):
        try: return original_plot(*a, **kw)
        finally: plt.close('all')
    official.plot_history = plot
    # Observe official loop without changing its sampling, losses or update ordering.
    original_forward = official.forward_pass
    last_metrics = {}
    def forward(batch, policy):
        result = original_forward(batch, policy)
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite training/validation metric')
        last_metrics.clear(); last_metrics.update({k:float(v.detach()) for k,v in result.items()})
        return result
    official.forward_pass = forward
    history = {'train':[], 'validation':[]}; best_loss = float('inf')
    class ObservedLoader:
        def __init__(self, dataset, kind):
            self.loader = DataLoader(dataset, batch_size=8, shuffle=kind=='train', num_workers=0)
            self.kind = kind; self.epoch = 0
        def __len__(self): return len(self.loader)
        def __iter__(self):
            nonlocal best_loss
            metrics = []
            for batch in self.loader:
                yield batch
                metrics.append(dict(last_metrics))
                if self.kind == 'train': progress['updates_completed'] += 1
            averaged = {k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
            history[self.kind].append(averaged)
            if self.kind == 'validation' and averaged['loss'] < best_loss:
                best_loss = averaged['loss']
                torch.save(policy_holder[0].state_dict(), output/'policy_best.tmp')
                (output/'policy_best.tmp').replace(output/'policy_best.ckpt')
                progress.update(best_epoch=self.epoch,best_validation_loss=best_loss)
            if self.kind == 'train':
                elapsed = time.perf_counter()-started
                progress.update(state='training',epochs_completed=self.epoch+1,elapsed_s=elapsed,
                    estimated_remaining_s=elapsed/(self.epoch+1)*(epochs-self.epoch-1),
                    train=averaged, validation=history['validation'][-1])
                write_json(output/'status.json', progress)
                with (output/'history.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=self.epoch,train=averaged,validation=history['validation'][-1],elapsed_s=elapsed))+'\n')
            self.epoch += 1
    write_json(output/'status.json',progress)
    try:
        best = official.train_bc(ObservedLoader(train,'train'),ObservedLoader(val,'validation'),
            dict(num_epochs=epochs, ckpt_dir=str(output), seed=0, policy_class='ACT',device='cpu',policy_config=config['policy']))
        torch.save(best[2], output/'policy_best.ckpt')
        policy = policy_holder[0]; policy.load_state_dict(best[2])
        progress.update(state='evaluating',best_epoch=int(best[0]),best_validation_loss=float(best[1]))
        write_json(output/'status.json',progress)
        result = evaluate(policy,val,stats,output)
        # Portable inference bundle; no optimizer, no hardware actions.
        torch.save(dict(state_dict=best[2],config=config,
                        stats={k:torch.from_numpy(v) for k,v in stats.items()},
                        detector_sha256=manifest['detector_sha256']), output/'inference_bundle.pt')
        progress.update(state='complete',elapsed_s=time.perf_counter()-started,evaluation=result)
        write_json(output/'status.json',progress)
        print('COMPLETE',json.dumps(progress),flush=True)
    except BaseException:
        progress.update(state='failed',error=traceback.format_exc(),elapsed_s=time.perf_counter()-started)
        write_json(output/'status.json',progress)
        raise


if __name__ == '__main__': main()
