"""Browser UI and Adeept environment adapter; ACT algorithms live in imitate_episodes.py."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time

import cv2
import numpy as np

from .arm import ArmConnection, ReturnToInitial
from act.policy import ROOT
from act.imitate_episodes import RolloutState, predict_action, eval_bc

DEMONSTRATION_CONFIG = ROOT / 'configs/demonstration.json'
INITIAL = np.asarray(json.loads(DEMONSTRATION_CONFIG.read_text())['initial_angles_deg'])
if (INITIAL.shape != (5,) or not np.issubdtype(INITIAL.dtype, np.integer)
        or np.any(INITIAL < [0, 0, 0, 0, 50])
        or np.any(INITIAL > [180, 180, 180, 180, 150])):
    raise ValueError('demonstration.json の initial_angles_deg は指令範囲内の整数5個が必要です')


class ACTTrial:
    def __init__(self, model, stats, checkpoint_dir, port, query_every=20, temporal_agg=True):
        self.rollout = RolloutState()
        self.temporal_agg = temporal_agg
        self.model = model
        self.stats = stats
        self.config = dict(mode="ACT on local CPU", policy_config=model.config)
        self.arm = ArmConnection()
        self.port = port
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self.closed = False
        self.returning = False
        self.returner = ReturnToInitial(self.return_send, INITIAL.tolist(), self.arm.lock)
        self.frame_lock = threading.Lock()
        self.frame = None
        self.frame_seq = 0
        self.frame_at = 0
        self.camera_error = ''
        self.message = '未接続。実物の姿勢に合う角度で有効にしてください。'
        self.input_ready = False
        self.last_prediction = None
        self.last_state = None
        self.completed_steps = 0
        self.timings = []
        self.misses = 0
        self.control_elapsed_s = 0
        self.query_count = 0
        self.preview_only = False
        camera_config = json.loads((ROOT / 'configs/camera.json').read_text())
        self.camera = cv2.VideoCapture(camera_config['camera'], cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.output = ROOT/'runs/act_robot_trials'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.output.mkdir(parents=True)
        self.session_output = self.output
        self.trial_number = 1
        metadata = dict(checkpoint=str(Path(checkpoint_dir)/'policy_best.ckpt'), config=self.config,
            initial_angles_deg=INITIAL.tolist(), requested_hz=20,
            max_delta_per_update_deg=None, temporal_aggregation=self.temporal_agg,
            aggregation='temporal aggregation k=0.01' if temporal_agg else 'newest chunk',
            normalization='aggregate normalized actions, then action*std+mean; radians to integer degrees',
            note='Commanded angles, not physical measurements. No guarantee of 20Hz.')
        (self.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
        self.set_query_every(query_every)
        threading.Thread(target=self.capture, daemon=True).start()

    def capture(self):
        while not self.closed:
            ok, frame = self.camera.read()
            if ok and frame.shape == (480,640,3):
                with self.frame_lock:
                    self.frame = frame; self.frame_seq += 1; self.frame_at = time.monotonic()
                self.camera_error = ''
            else:
                self.camera_error = 'カメラ画像を取得できません'; time.sleep(.1)

    def snapshot(self):
        with self.frame_lock:
            if self.frame is None or time.monotonic()-self.frame_at > 1:
                raise RuntimeError('カメラ画像が未取得、または1秒以上更新されていません')
            return self.frame.copy(),self.frame_seq

    def return_send(self, command):
        result = self.arm.action(command)
        self.last_state = result['joints']
        return result

    def running(self):
        return self.returner.active or (self.worker is not None and self.worker.is_alive())

    def state(self):
        message = self.returner.status()['message'] if self.returning else self.message
        return dict(connected=self.arm.port is not None, running=self.running(), message=message,
            joints=self.last_state or [], input_ready=self.input_ready, prediction_deg=self.last_prediction,
            steps=self.completed_steps, camera_error=self.camera_error,
            actual_update_median_ms=float(np.median(self.timings)*1000) if self.timings else None,
            deadline_misses=self.misses, output=str(self.output), trial_number=self.trial_number,
            model='ACT', inference_device='cpu', temporal_aggregation=self.temporal_agg, queries=self.query_count,
            query_every_steps=self.query_every, query_interval_training_seconds=self.query_every/20,
            actual_update_hz=self.completed_steps/self.control_elapsed_s if self.control_elapsed_s else None)

    def next_trial(self):
        if self.running(): raise ValueError('先に停止してください')
        (self.output/'trial_summary.json').write_text(json.dumps(dict(trial_number=self.trial_number,
            steps=self.completed_steps,deadline_misses=self.misses,
            outcome='not automatically judged',last_state=self.last_state),indent=2))
        self.trial_number += 1
        self.output = self.session_output/f'trial_{self.trial_number:03d}'
        self.output.mkdir()
        metadata = json.loads((self.session_output/'metadata.json').read_text())
        metadata['trial_number'] = self.trial_number
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        self.input_ready = False; self.last_prediction = None
        self.completed_steps = 0; self.timings = []; self.misses = 0; self.returning = False
        self.control_elapsed_s = 0
        self.ensemble.clear(); self.query_count = 0
        self.set_query_every(self.query_every)
        self.message = '次の試行です。開始姿勢へ戻し、配置を変えて画像入力を確認してください。'

    def refresh(self):
        with self.arm.lock:
            self.last_state = self.arm.status()
        return self.last_state

    def set_query_every(self,steps):
        if type(steps) is not int or not 1<=steps<=100:raise ValueError('推論間隔は1〜100制御ステップです')
        if self.running() or self.completed_steps:raise ValueError('停止して「次の試行」を作成してから変更してください')
        self.query_every=steps;self.ensemble.clear()
        self.config.update(query_every_steps=steps,query_interval_training_seconds=steps/20,
            control=f'query every {steps} logical control steps; '+('temporal aggregation k=0.01' if self.temporal_agg else 'newest predicted chunk'))
        metadata=json.loads((self.output/'metadata.json').read_text())
        metadata.update(query_every_steps=steps,query_interval_training_seconds=steps/20,config=self.config,
            control=self.config['control'])
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        self.message=f'推論間隔：{steps}制御ステップ（学習時の{steps/20:g}秒分）に1回。実時間とは異なります。'

    def confirm_input(self):
        if self.completed_steps:raise ValueError('配置を変える場合は「次の試行」を押してください')
        frame,sequence=self.snapshot();self.refresh()
        if any(s['angle']!=s['target'] for s in self.last_state):raise ValueError('先に移動を停止してください')
        self.input_ready=True
        self.ensemble.clear()
        cv2.imwrite(str(self.output/'initial_image.png'),frame)
        (self.output/'initial_image.json').write_text(json.dumps(dict(frame_sequence=sequence,state=self.last_state),indent=2))
        self.message='画像入力を確認しました。検出器は使わず、現在のカメラ画像をモデルへ入力します。'

    @property
    def completed_steps(self):return self.rollout.step

    @completed_steps.setter
    def completed_steps(self,value):self.rollout.step=value

    @property
    def query_count(self):return self.rollout.query_count

    @query_count.setter
    def query_count(self,value):self.rollout.query_count=value

    @property
    def ensemble(self):return self.rollout.ensemble

    @ensemble.setter
    def ensemble(self,value):self.rollout.ensemble=value

    def observe(self):
        if not self.input_ready:raise ValueError('先に画像入力を確認してください')
        frame,sequence=self.snapshot();self.refresh()
        if not all(s['active'] for s in self.last_state):raise ValueError('全関節を有効にしてください')
        return dict(image=frame,qpos=np.array([s['angle'] for s in self.last_state]),frame_sequence=sequence)

    def record_prediction(self,observation,prediction,details):
        if details['queried']:
            request_id=details['request_id']
            cv2.imwrite(str(self.output/f'query_{request_id}.png'),observation['image'])
            (self.output/f'query_{request_id}.json').write_text(json.dumps(dict(
                frame_sequence=observation['frame_sequence'],**details),indent=2))
        self.last_prediction=prediction.tolist()

    def predict(self):
        observation=self.observe()
        prediction,details=predict_action(self.model,self.stats,observation,self.rollout,
            self.query_every,self.temporal_agg,self.preview_only)
        self.record_prediction(observation,prediction,details)
        return prediction,np.asarray(observation['qpos'])

    def step(self,prediction,current):
        target=self.targets_for_prediction(prediction,current)
        self.move_batch(target)
        return target

    def record_step(self,index,observation,prediction,target,duration,elapsed):
        self.timings.append(duration);self.misses+=int(duration>.05)
        with (self.output/'steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(index=index,time_ns=time.time_ns(),qpos_deg=observation['qpos'].tolist(),
                prediction_deg=prediction.tolist(),sent_deg=target.tolist(),compute_and_send_s=duration,
                input_mode=self.config['mode']))+'\n')
        self.control_elapsed_s=getattr(self,'control_elapsed_s',0)+elapsed

    def halt(self):
        with self.arm.lock:
            self.arm.command('STOP');self.last_state=self.arm.status()

    def targets_for_prediction(self,prediction,current):
        values=np.asarray(prediction,dtype=float)
        if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('Invalid target')
        target=np.rint(values).astype(int)
        if np.any(target<np.array([0,0,0,0,50])) or np.any(target>np.array([180,180,180,180,150])):
            raise ValueError('モデルの目標角度が実機の指令範囲外のため停止しました')
        return target

    def move_batch(self,targets):
        # One write avoids a separate serial round trip for every joint.
        with self.arm.lock:
            commands = [f'MOVE {i+1} {int(a)}' for i,a in enumerate(targets)]
            self.arm.port.write(('\n'.join(commands)+'\n').encode('ascii'))
            done=0;lines=[];deadline=time.monotonic()+1
            while done < len(commands) and time.monotonic() < deadline:
                line=self.arm.port.readline().decode('ascii',errors='replace').strip()
                if line=='DONE': done+=1
                elif line: lines.append(line)
            if done != len(commands) or any(s.startswith('ERROR:') for s in lines):
                raise RuntimeError('MOVE指令の受理を確認できません: '+' / '.join(lines))

    def run(self,step_limit):
        self.message='モデルで動作中'
        try:
            eval_bc(self.model,self.stats,self,step_limit,self.rollout,self.query_every,self.temporal_agg,self.stop)
            self.message='停止・保持しました。'
        except Exception as e:
            self.message='停止: '+str(e)

    def action(self,data):
        action=data.get('action')
        if action=='stop':
            self.stop.set()
            self.returner.cancel()
            if self.worker: self.worker.join(timeout=2)
            if self.returner.thread: self.returner.thread.join(timeout=2)
            with self.arm.lock:
                if self.arm.port: self.arm.command('STOP'); self.last_state=self.arm.status()
            self.returning=False;self.message='停止・保持しました。'; return
        with self.lock:
            if self.running(): raise ValueError('先に停止してください')
            self.returning=False
            if action in ['manual','return_initial','connect']:self.ensemble.clear()
            if action=='connect':
                result=self.arm.action(dict(action='connect',port=self.port));self.last_state=result['joints']
                self.input_ready=False; self.message='接続しました。実際の姿勢に合う角度で有効にしてください。'
            elif action=='manual':
                command=data['command']
                if command.get('action') not in ['start','status','speed']: raise ValueError('不正な手動操作')
                result=self.arm.action(command)
                self.last_state=result['joints'];self.message=result.get('message','')
            elif action=='confirm_input': self.confirm_input()
            elif action=='query_interval': self.set_query_every(data.get('steps'))
            elif action=='next_trial': self.next_trial()
            elif action=='return_initial':
                self.returner.start();self.returning=True
            elif action=='preview':
                self.preview_only=True
                try:self.predict()
                finally:self.preview_only=False
                self.message='予測だけ表示しました。動作指令は送っていません。'
            elif action=='run':
                step_limit=data.get('steps')
                if type(step_limit) is not int or step_limit not in [1,10,1000]:
                    raise ValueError('試験更新回数が不正です')
                if not self.input_ready:raise ValueError('先に画像入力を確認してください')
                self.refresh()
                if not all(s['active'] for s in self.last_state): raise ValueError('全関節を有効にしてください')
                if self.completed_steps==0 and np.max(np.abs(np.array([s['angle'] for s in self.last_state])-INITIAL))>2:
                    raise ValueError(f'最初の試験は設定された開始姿勢 {INITIAL.tolist()} 度から行ってください')
                if self.last_state[3]['angle']!=INITIAL[3]: raise ValueError(f'手首を{INITIAL[3]}度にしてください')
                with self.arm.lock:
                    for i in range(5): self.arm.command(f'SPEED {i+1} 25')
                self.stop.clear()
                self.worker=threading.Thread(target=self.run,args=(step_limit,),daemon=True);self.worker.start()
            else: raise ValueError('不正な操作')


def serve_trial(trial, http_port):
    token=secrets.token_urlsafe(32)
    page=Path(__file__).with_suffix('.html').read_text()
    page=page.replace('__QUERY_EVERY__', str(trial.query_every))
    page=page.replace('__AGGREGATION__', '有効（k=0.01）' if trial.temporal_agg else '無効（最新の予測列を使用）')
    page=page.replace('__TOKEN__',token)
    page=page.replace('__ENABLE_ANGLES__',json.dumps(INITIAL.tolist()))
    page=page.replace('__INITIAL_POSE__', '・'.join(f'{name}{angle}°' for name,angle in zip(
        ['台座','肩','肘','手首','グリッパー'], INITIAL.tolist())))
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,body,mime='application/json'):
            if isinstance(body,str):body=body.encode()
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                if self.path=='/': return self.respond(200,page,'text/html; charset=utf-8')
                if self.path=='/state': return self.respond(200,json.dumps(trial.state(),ensure_ascii=False))
                if self.path.startswith('/frame.jpg'):
                    frame,_=trial.snapshot()
                    ok,jpg=cv2.imencode('.jpg',frame)
                    if not ok: raise RuntimeError('JPEG生成失敗')
                    return self.respond(200,jpg.tobytes(),'image/jpeg')
                self.respond(404,'{}')
            except Exception as e:self.respond(400,json.dumps(dict(error=str(e)),ensure_ascii=False))
        def do_POST(self):
            if self.path!='/api' or self.headers.get('X-Arm-Token')!=token:return self.respond(403,'{}')
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<2048:raise ValueError('不正な要求サイズ')
                trial.action(json.loads(self.rfile.read(size)))
                self.respond(200,json.dumps(trial.state(),ensure_ascii=False))
            except Exception as e:self.respond(400,json.dumps(dict(error=str(e)),ensure_ascii=False))
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',http_port),Handler)
    print(f'READY http://127.0.0.1:{http_port} output={trial.output}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        trial.action({'action':'stop'});trial.closed=True;trial.camera.release();server.server_close()


