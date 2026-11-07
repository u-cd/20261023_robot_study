"""ACT training, validation and rollout in one place, based on official imitate_episodes.py.

UI/camera/USB are an environment adapter in ui/act_test.py. This module owns model
loading, preprocessing, inference cadence, temporal aggregation and rollout.
"""
import argparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robot-act-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .policy import ACTPolicy, CONFIG, ROOT, load_policy_config, make_optimizer
from .utils import load_data, compute_dict_mean, detach_dict, set_seed


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    device = next(policy.parameters()).device
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = ACTPolicy(policy_config, pretrained_backbone=config.get("pretrained_backbone", True))
    policy.to(config['device'])
    optimizer = make_optimizer(policy, policy_config)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def train(config_path, output, device=None, epochs=None):
    config=json.loads(Path(config_path).read_text())
    settings=dict(config['training']);policy=dict(config['policy'])
    device=device or settings['device']
    epochs=epochs if epochs is not None else settings['num_epochs']
    if epochs < 1:raise ValueError('epochs must be positive')
    output=Path(output)
    if output.exists():raise ValueError('Choose a new output directory; refusing overwrite')
    torch.set_num_threads(settings['cpu_threads'])
    set_seed(1)
    source=ROOT/settings['dataset_dir']
    train_loader,val_loader,stats,_=load_data(str(source),settings['num_episodes'],policy['camera_names'],
        settings['batch_size'],settings['batch_size'],action_chunk_size=policy['num_queries'],num_workers=settings.get('num_workers',1))
    if len(stats['qpos_mean']) != policy['state_dim']:raise ValueError('Dataset state dimension does not match model')
    output.mkdir(parents=True)
    policy['device']=device
    (output/'policy_config.json').write_text(json.dumps(policy,indent=2)+'\n')
    with (output/'dataset_stats.pkl').open('wb') as f:pickle.dump(stats,f)
    run=dict(num_epochs=epochs,ckpt_dir=str(output),seed=settings['seed'],policy_config=policy,device=device,pretrained_backbone=settings.get('pretrained_backbone',True))
    (output/'training_config.json').write_text(json.dumps(dict(config=config,resolved=run),indent=2)+'\n')
    best=train_bc(train_loader,val_loader,run)
    torch.save(best[2],output/'policy_best.ckpt')
    return best


def load_act(checkpoint_dir):
    checkpoint_dir=Path(checkpoint_dir)
    config=load_policy_config(checkpoint_dir)
    config['device']='cpu'
    model=ACTPolicy(config)
    model.load_state_dict(torch.load(checkpoint_dir/'policy_best.ckpt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    with (checkpoint_dir/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    return model, stats


def predict(model, stats, frame, qpos_deg):
    started=time.perf_counter()
    if frame.shape != (480,640,3):raise ValueError('画像サイズは640x480です')
    qpos=np.deg2rad(np.asarray(qpos_deg,dtype=np.float64)).astype(np.float32)
    if qpos.shape != (5,) or not np.isfinite(qpos).all():raise ValueError('Invalid qpos')
    image=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None,None]/255
    normalized=(torch.from_numpy(qpos)-torch.as_tensor(stats['qpos_mean']))/torch.as_tensor(stats['qpos_std'])
    before=time.perf_counter()
    with torch.inference_mode():chunk=model(normalized[None],image)[0].numpy()
    inference_ms=(time.perf_counter()-before)*1000
    if chunk.shape != (100,5) or not np.isfinite(chunk).all():raise ValueError('Invalid prediction')
    return chunk, dict(local_inference_ms=inference_ms,total_local_ms=(time.perf_counter()-started)*1000)


def temporal_aggregate(actions_for_curr_step, device):
    """Original eval_bc temporal ensemble, shared with the Adeept hardware adapter."""
    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / exp_weights.sum()
    exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    return raw_action


class TemporalEnsemble:
    """Keep exactly the populated rows that original eval_bc would read at t."""
    def __init__(self,aggregate):
        self.aggregate=aggregate;self.chunks=OrderedDict()

    def clear(self):self.chunks.clear()

    def action(self,t,chunk=None,remember=True,temporal_agg=True):
        if chunk is not None:
            chunk=np.asarray(chunk,dtype=np.float32)
            if chunk.shape!=(100,5) or not np.isfinite(chunk).all():raise ValueError('Invalid normalized chunk')
        populated=[(start,values) for start,values in self.chunks.items()
                   if 0<=t-start<100 and (chunk is None or start!=t)]
        rows=[values[t-start] for start,values in populated]
        if chunk is not None:
            rows.append(chunk[0]);populated.append((t,chunk.copy()))
        if not rows:raise ValueError('No valid predictions for this step')
        value=(self.aggregate(torch.from_numpy(np.stack(rows)),'cpu')[0].numpy()
               if temporal_agg else rows[-1].copy())
        if not np.isfinite(value).all():raise ValueError('Invalid ensemble action')
        if remember:self.chunks=OrderedDict(populated)
        return value


@dataclass
class RolloutState:
    step: int = 0
    query_count: int = 0
    ensemble: TemporalEnsemble = field(default_factory=lambda: TemporalEnsemble(temporal_aggregate))


def predict_action(model, stats, observation, state, query_every=20, temporal_agg=True, preview=False):
    """One rollout step: query if due, select/aggregate predictions, undo normalization."""
    last_query=next(reversed(state.ensemble.chunks)) if state.ensemble.chunks else None
    new_query=preview or last_query is None or state.step-last_query >= query_every
    chunk=None
    details=dict(logical_step=state.step, preview=preview, query_every_steps=query_every,
                 qpos_deg=np.asarray(observation['qpos']).tolist(), queried=new_query)
    if new_query:
        request_id=time.time_ns()
        chunk,metrics=predict(model,stats,observation['image'],observation['qpos'])
        state.query_count+=1
        details.update(request_id=request_id, normalized_actions=chunk.tolist(), **metrics)
    normalized=state.ensemble.action(state.step,chunk,remember=not preview,temporal_agg=temporal_agg)
    radians=normalized*np.asarray(stats['action_std'])+np.asarray(stats['action_mean'])
    prediction=np.rad2deg(radians)
    details['aggregated_action_deg']=prediction.tolist()
    return prediction,details


def eval_bc(model, stats, env, max_timesteps, state=None, query_every=20, temporal_agg=True, stop_event=None):
    """Rollout without UI dependencies, following the official observation→policy→env.step loop.

    env.observe(): {image: BGR frame, qpos: commanded degrees, ...}
    env.step(prediction, current): validate/send targets, return sent degrees.
    env.record_prediction()/record_step(): persist observations and execution logs.
    env.halt(): stop motion while retaining joint holding.
    A shared state allows the UI to pause/resume a trial without losing history.
    """
    if type(max_timesteps) is not int or max_timesteps < 1:raise ValueError('Invalid rollout length')
    if type(query_every) is not int or not 1<=query_every<=100:raise ValueError('Invalid query interval')
    state=state if state is not None else RolloutState()
    deadline=time.monotonic()+1800
    try:
        with torch.inference_mode():
            for index in range(max_timesteps):
                if stop_event is not None and stop_event.is_set():break
                started=time.monotonic()
                if started >= deadline:break
                observation=env.observe()
                prediction,details=predict_action(model,stats,observation,state,query_every,temporal_agg)
                env.record_prediction(observation,prediction,details)
                if stop_event is not None and stop_event.is_set():break
                target=env.step(prediction,np.asarray(observation['qpos']))
                duration=time.monotonic()-started
                state.step+=1
                if index+1 < max_timesteps:
                    delay=max(0,.05-(time.monotonic()-started))
                    if stop_event is None:time.sleep(delay)
                    else:stop_event.wait(delay)
                env.record_step(state.step,observation,prediction,target,duration,time.monotonic()-started)
    finally:
        env.halt()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval',action='store_true',help='Open CPU rollout UI; otherwise train')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--checkpoint-dir',type=Path,default=ROOT/'models/act_image_runpod_20261107')
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--http-port',type=int,default=8767)
    parser.add_argument('--query-every',type=int,default=20,choices=range(1,101),metavar='1..100')
    parser.add_argument('--no-temporal-agg',action='store_true',help='Use the newest chunk without weighted aggregation')
    args=parser.parse_args()
    if args.eval:
        if args.device not in (None,'cpu'):parser.error('Rollout inference is CPU-only')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        model,stats=load_act(args.checkpoint_dir)
        from ui.act_test import ACTTrial, serve_trial
        trial=ACTTrial(model,stats,args.checkpoint_dir,args.port,args.query_every,not args.no_temporal_agg)
        serve_trial(trial,args.http_port)
    else:
        if args.output is None:parser.error('--output is required for training')
        train(args.config,args.output,args.device,args.epochs)


if __name__=='__main__':main()
