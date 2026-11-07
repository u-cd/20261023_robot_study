"""Trial UI for the RunPod-trained official image ACT via a localhost SSH tunnel."""
import argparse
import base64
import json
import pickle
from pathlib import Path
import time
import urllib.request
from collections import OrderedDict

import cv2
import numpy as np
import torch

from test_detection_act import Trial, serve_trial
from detection_act import ROOT, configure_official


class OfficialTemporalEnsemble:
    """Keep exactly the populated rows that original eval_bc would read at t."""
    def __init__(self,aggregate):
        self.aggregate=aggregate;self.chunks=OrderedDict()

    def clear(self):self.chunks.clear()

    def action(self,t,chunk=None,remember=True):
        if chunk is not None:
            chunk=np.asarray(chunk,dtype=np.float32)
            if chunk.shape!=(100,5) or not np.isfinite(chunk).all():raise ValueError('Invalid normalized chunk')
        populated=[(start,values) for start,values in self.chunks.items()
                   if 0<=t-start<100 and (chunk is None or start!=t)]
        rows=[values[t-start] for start,values in populated]
        if chunk is not None:
            rows.append(chunk[0]);populated.append((t,chunk.copy()))
        if not rows:raise ValueError('No valid predictions for this step')
        value=self.aggregate(torch.from_numpy(np.stack(rows)),'cpu')[0].numpy()
        if not np.isfinite(value).all():raise ValueError('Invalid ensemble action')
        if remember:self.chunks=OrderedDict(populated)
        return value


class RemoteImagePolicy:
    def __init__(self,url,token_file):
        self.url=url.rstrip('/');self.token=Path(token_file).read_text().strip()
        with urllib.request.urlopen(self.url+'/health',timeout=5) as r:self.health=json.load(r)
        if not self.health['ready'] or self.health['device']!='cuda':raise ValueError('GPU推論サーバーが未準備です')
        self.config=dict(mode='official image ACT on RunPod GPU',checkpoint=self.health['checkpoint'],
            control='official eval_bc temporal aggregation: query every step, k=0.01, original oldest-to-newest row order',
            requested_hz=20,temporal_aggregation=True,image_size=[640,480],state_dim=5,
            action_mapping='absolute model targets rounded to integer degrees; reject out-of-range; no 2-degree delta clip or forced wrist')

    def predict(self,frame,qpos_deg,request_id):
        ok,compressed=cv2.imencode('.png',frame)
        if not ok:raise ValueError('画像を符号化できません')
        body=json.dumps(dict(request_id=request_id,qpos_rad=np.deg2rad(qpos_deg).tolist(),
            image_base64=base64.b64encode(compressed).decode())).encode()
        request=urllib.request.Request(self.url+'/predict',data=body,
            headers={'Content-Type':'application/json','X-ACT-Token':self.token})
        started=time.monotonic()
        with urllib.request.urlopen(request,timeout=15) as response:result=json.load(response)
        chunk=np.asarray(result['normalized_actions'],dtype=np.float32)
        if result['request_id']!=request_id or chunk.shape!=(100,5) or not np.isfinite(chunk).all():
            raise ValueError('GPUからの予測応答が不正です')
        return chunk,dict(roundtrip_ms=(time.monotonic()-started)*1000,
            gpu_inference_ms=result['inference_ms'])


class LocalImagePolicy:
    """The same full-sized trained ACTPolicy, with CPU execution and no network."""
    def __init__(self,checkpoint_dir):
        checkpoint_dir=Path(checkpoint_dir)
        configure_official(ROOT/'runs/act_robot_trials',1)
        from policy import ACTPolicy
        config=dict(lr=1e-5,lr_backbone=1e-5,num_queries=100,kl_weight=10,hidden_dim=512,
            dim_feedforward=3200,enc_layers=4,dec_layers=7,nheads=8,camera_names=['top'],
            backbone='resnet18',device='cpu',state_dim=5)
        self.model=ACTPolicy(config)
        self.model.load_state_dict(torch.load(checkpoint_dir/'policy_best.ckpt',map_location='cpu',weights_only=True),strict=True)
        self.model.eval()
        with (checkpoint_dir/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
        self.mean=torch.as_tensor(stats['qpos_mean'],dtype=torch.float32)
        self.std=torch.as_tensor(stats['qpos_std'],dtype=torch.float32)
        self.health=dict(ready=True,device='cpu',checkpoint=str(checkpoint_dir/'policy_best.ckpt'),
            action_mean=stats['action_mean'].tolist(),action_std=stats['action_std'].tolist(),
            output_shape=[100,5],robot_io=False)
        self.config=dict(mode='RunPod-trained official image ACT on local CPU',checkpoint=self.health['checkpoint'],
            requested_hz=20,temporal_aggregation=True,image_size=[640,480],state_dim=5,
            device='cpu',precision='float32',cpu_threads=torch.get_num_threads(),
            policy_config=config,control='official temporal aggregation; same weights and normalization; no network',
            action_mapping='absolute model targets rounded to integer degrees; reject out-of-range; no 2-degree delta clip or forced wrist')

    def predict(self,frame,qpos_deg,request_id):
        started=time.perf_counter()
        if frame.shape!=(480,640,3):raise ValueError('画像サイズは640x480です')
        qpos=np.deg2rad(np.asarray(qpos_deg,dtype=np.float64)).astype(np.float32)
        if qpos.shape!=(5,) or not np.isfinite(qpos).all():raise ValueError('Invalid qpos')
        image=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None,None]/255
        normalized=(torch.from_numpy(qpos)-self.mean)/self.std
        before=time.perf_counter()
        with torch.inference_mode():chunk=self.model(normalized[None],image)[0].numpy()
        inference_ms=(time.perf_counter()-before)*1000
        if chunk.shape!=(100,5) or not np.isfinite(chunk).all():raise ValueError('Invalid local prediction')
        return chunk,dict(local_inference_ms=inference_ms,total_local_ms=(time.perf_counter()-started)*1000)


class ImageTrial(Trial):
    def __init__(self,backend,port,query_every=1):
        configure_official(ROOT/'runs/act_robot_trials',1)
        from imitate_episodes import temporal_aggregate
        self.ensemble=OfficialTemporalEnsemble(temporal_aggregate)
        self.chunk=None;self.chunk_index=0;self.query_count=0;self.preview_only=False
        self.query_every=query_every
        super().__init__(backend.health['checkpoint'],port,backend=backend)
        metadata=json.loads((self.output/'metadata.json').read_text())
        metadata.update(temporal_aggregation=True,max_delta_per_update_deg=None,
            aggregation='official imitate_episodes.temporal_aggregate; extracted unchanged from eval_bc',
            normalization='aggregate normalized actions, then action*std+mean; radians to integer degrees at hardware boundary')
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        self.set_query_every(query_every)

    def set_query_every(self,steps):
        if type(steps) is not int or not 1<=steps<=100:raise ValueError('推論間隔は1〜100制御ステップです')
        if self.running() or self.completed_steps:raise ValueError('停止して「次の試行」を作成してから変更してください')
        self.query_every=steps;self.ensemble.clear()
        self.backend.config.update(query_every_steps=steps,query_interval_training_seconds=steps/20,
            control=f'query every {steps} logical control steps; consume matching chunk elements and use official k=0.01 aggregation of available predictions')
        metadata=json.loads((self.output/'metadata.json').read_text())
        metadata.update(query_every_steps=steps,query_interval_training_seconds=steps/20,config=self.backend.config,
            control=self.backend.config['control'])
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        self.message=f'推論間隔：{steps}制御ステップ（学習時の{steps/20:g}秒分）に1回。実時間とは異なります。'

    def detect(self):
        if self.completed_steps:raise ValueError('配置を変える場合は「次の試行」を押してください')
        frame,sequence=self.snapshot();self.refresh()
        if any(s['angle']!=s['target'] for s in self.last_state):raise ValueError('先に移動を停止してください')
        self.frozen=np.zeros((2,6),dtype=np.float32) # Readiness flag only; never passed to the image model.
        self.ensemble.clear()
        cv2.imwrite(str(self.output/'initial_image.png'),frame)
        (self.output/'initial_image.json').write_text(json.dumps(dict(frame_sequence=sequence,state=self.last_state),indent=2))
        self.message='画像入力を確認しました。検出器は使わず、現在のカメラ画像をモデルへ入力します。'

    def state(self):
        state=super().state();state['detections']=None
        state.update(model='RunPod image ACT',temporal_aggregation=True,queries=self.query_count,
            inference_device=self.backend.health['device'],
            query_every_steps=self.query_every,query_interval_training_seconds=self.query_every/20)
        elapsed=getattr(self,'control_elapsed_s',0)
        if elapsed:state['actual_update_hz']=self.completed_steps/elapsed
        return state

    def next_trial(self):
        super().next_trial();self.ensemble.clear();self.query_count=0;self.set_query_every(self.query_every)
        self.message='次の試行です。開始姿勢へ戻し、配置を変えて画像入力を確認してください。'

    def predict(self):
        if self.frozen is None:raise ValueError('先に画像入力を確認してください')
        frame,sequence=self.snapshot();self.refresh()
        if not all(s['active'] for s in self.last_state):raise ValueError('全関節を有効にしてください')
        current=np.array([s['angle'] for s in self.last_state])
        t=self.completed_steps
        last_query=next(reversed(self.ensemble.chunks)) if self.ensemble.chunks else None
        new_query=self.preview_only or last_query is None or t-last_query>=self.query_every
        chunk=None
        if new_query:
            request_id=time.time_ns()
            chunk,metrics=self.backend.predict(frame,current,request_id)
            self.query_count+=1
        normalized=self.ensemble.action(self.completed_steps,chunk,remember=not self.preview_only)
        radians=normalized*np.asarray(self.backend.health['action_std'])+np.asarray(self.backend.health['action_mean'])
        prediction=np.rad2deg(radians)
        if new_query:
            cv2.imwrite(str(self.output/f'query_{request_id}.png'),frame)
            (self.output/f'query_{request_id}.json').write_text(json.dumps(dict(frame_sequence=sequence,
                logical_step=t,preview=self.preview_only,qpos_deg=current.tolist(),query_every_steps=self.query_every,
                normalized_actions=chunk.tolist(),aggregated_action_deg=prediction.tolist(),**metrics),indent=2))
        self.last_prediction=prediction.tolist()
        return prediction,current

    def after_step(self):pass

    def targets_for_prediction(self,prediction,current):
        values=np.asarray(prediction,dtype=float)
        if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('Invalid target')
        target=np.rint(values).astype(int)
        if np.any(target<np.array([0,0,0,0,50])) or np.any(target>np.array([180,180,180,180,150])):
            raise ValueError('モデルの目標角度が実機の指令範囲外のため停止しました')
        return target

    def action(self,data):
        if data.get('action')=='query_interval':
            with self.lock:self.set_query_every(data.get('steps'))
            return
        if self.running():return super().action(data)
        if data.get('action') in ['manual','return_initial','connect']:self.ensemble.clear()
        self.preview_only=data.get('action')=='preview'
        try:return super().action(data)
        finally:self.preview_only=False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:9120')
    parser.add_argument('--token-file',type=Path)
    parser.add_argument('--local',action='store_true',help='Run the same trained full-sized image ACT on this CPU')
    parser.add_argument('--checkpoint-dir',type=Path,default=ROOT/'models/act_image_runpod_20261107')
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--http-port',type=int,default=8767)
    parser.add_argument('--enable-angles',type=int,nargs=5)
    parser.add_argument('--query-every',type=int,default=1,choices=range(1,101),metavar='1..100')
    args=parser.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not args.local and args.token_file is None:parser.error('--token-file is required for remote inference')
    backend=LocalImagePolicy(args.checkpoint_dir) if args.local else RemoteImagePolicy(args.url,args.token_file)
    trial=ImageTrial(backend,args.port,args.query_every)
    page=Path(__file__).with_name('test_detection_act.html').read_text()
    page=page.replace('CPU ACT 実機テスト','RunPod 画像入力ACT 実機テスト')
    page=page.replace('YOLOv8nで最初だけ検出し、その位置と現在の関節指令角度からACTが操作を予測します。',
        '指定した制御ステップ間隔で現在の画像と角度をGPUへ送り、間は予測済みの操作列を使います。有効な予測を公式のtemporal aggregation（k=0.01）で集約します。検出器は使いません。')
    page=page.replace('手首80°固定、速度25°/秒、1更新の変更は最大2°です。',
        'モデルの絶対目標角度を送ります。基板速度は実演時と同じ25°/秒。独自の2°制限や手首の出力固定は行いません。')
    page=page.replace('初期検出を確定（5枚）','画像入力を確認').replace('初期検出の枠は固定値です。動作中の検出・追跡ではありません。スポンジとcoinを動かしたら再取得してください。',
        '現在時刻に対応する過去の予測を、公式と同じ順序・重みで集約します。学習データの20Hzに対し、通信・基板を含む実際の周期を表示します。')
    page=page.replace('初期検出','画像入力')
    page=page.replace("act({action:'run',seconds:0})", "act({action:'run',steps:1})")
    page=page.replace("act({action:'run',seconds:1})", "act({action:'run',steps:10})").replace('1秒試す','10回更新する')
    page=page.replace("act({action:'run',seconds:10})", "act({action:'run',steps:1000})").replace('10秒試す','連続実行（最大1000回）')
    page=page.replace('各試験は自動停止します。',
        '推論する更新では応答を待ちます。指定回数または30分で自動停止します。')
    page=page.replace('<div id="joints"></div>', '<div id="joints"></div>'
        '<p>推論間隔 <input id="interval" type="number" min="1" max="100" value="'+str(args.query_every)+'">'
        '制御ステップに1回 <button onclick="act({action:\'query_interval\',steps:Number(document.querySelector(\'#interval\').value)})">適用</button>'
        ' 20ステップ＝学習時の1秒分。実時間の1秒ではありません。変更は次の試行の開始前に行います。</p>')
    page=page.replace('処理時間中央値ms:s.actual_update_median_ms,','処理時間中央値ms:s.actual_update_median_ms,実測更新Hz:s.actual_update_hz,GPUへの推論要求数:s.queries,')
    page=page.replace('試行番号:s.trial_number,','試行番号:s.trial_number,推論間隔ステップ:s.query_every_steps,学習時の秒数:s.query_interval_training_seconds,')
    if args.local:
        page=page.replace('RunPod 画像入力ACT 実機テスト','ローカルCPU 画像入力ACT 実機テスト')
        page=page.replace('GPUへ送り','このPCに入力し').replace('GPUへの推論要求数','モデルへの推論要求数')
    page=page.replace('試行番号:s.trial_number,','試行番号:s.trial_number,推論実行場所:s.inference_device,')
    serve_trial(trial,args.http_port,args.enable_angles,page)


if __name__=='__main__':main()
