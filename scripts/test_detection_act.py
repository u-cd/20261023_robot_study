"""Local, bounded robot trials of the trained initial-detection ACT policy."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/robot-act-live-yolo')
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from arm_panel import ArmConnection
from detection_act import ROOT, load_inference_bundle
from fine_step import fine_step
from return_to_initial import ReturnToInitial

INITIAL = np.array([70,113,43,80,90])


def bounded_targets(prediction, current, max_delta=2):
    prediction = np.asarray(prediction, dtype=float)
    current = np.asarray(current, dtype=int)
    if prediction.shape != (5,) or not np.isfinite(prediction).all():
        raise ValueError('モデル出力が不正です')
    limits_low = np.array([0,0,0,80,50])
    limits_high = np.array([180,180,180,80,150])
    rounded = np.rint(prediction).astype(int)
    limited = np.clip(rounded,limits_low,limits_high)
    return np.clip(limited,current-max_delta,current+max_delta).astype(int)


def parse_status(lines):
    result = []
    for line in lines:
        m = re.fullmatch(r'([1-5]) (ON|OFF) (?:last-)?commanded=(\d+) target=(\d+) speed=(\d+)',line)
        if m:
            result.append(dict(joint=int(m[1]),active=m[2]=='ON',angle=int(m[3]),target=int(m[4]),speed=int(m[5])))
    if len(result) != 5: raise RuntimeError('基板の状態応答が不正です')
    return result


class Trial:
    def __init__(self, bundle, port, backend=None):
        self.arm = ArmConnection(); self.port = port
        self.lock = threading.RLock(); self.stop = threading.Event()
        self.worker = None; self.closed = False
        self.returning = False
        self.returner = ReturnToInitial(self.return_send, INITIAL.tolist(), self.arm.lock)
        self.frame_lock = threading.Lock(); self.frame = None; self.frame_seq = 0; self.frame_at = 0
        self.camera_error = ''; self.message = '未接続。まず開始姿勢と一致する角度で各関節を有効にしてください。'
        self.frozen = None; self.boxes = None; self.last_prediction = None; self.last_state = None
        self.completed_steps = 0; self.timings = []; self.misses = 0
        self.control_elapsed_s = 0
        self.backend = backend
        if backend is None:
            self.policy,self.stats,self.config = load_inference_bundle(bundle)
            weights = ROOT/self.config['detector']['weights']
            bundle_data = torch.load(bundle,map_location='cpu',weights_only=True)
            if hashlib.sha256(weights.read_bytes()).hexdigest() != bundle_data['detector_sha256']:
                raise ValueError('学習時と異なる検出器です')
            detector_sha256 = bundle_data['detector_sha256']
            self.detector = YOLO(str(weights))
        else:
            self.config = backend.config
            detector_sha256 = None
        self.detector_lock = threading.Lock()
        camera_config = json.loads((ROOT/'configs/camera.json').read_text())
        self.camera = cv2.VideoCapture(camera_config['camera'], cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,640); self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.output = ROOT/'runs/act_robot_trials'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.output.mkdir(parents=True)
        self.session_output = self.output
        self.trial_number = 1
        (self.output/'metadata.json').write_text(json.dumps(dict(bundle=str(bundle),config=self.config,
            detector_sha256=detector_sha256,initial_angles_deg=INITIAL.tolist(),
            requested_hz=20,max_delta_per_update_deg=2,temporal_aggregation=False,
            control='first action of each freshly predicted chunk' if backend is None else backend.config['control'],
            note='Actual timing recorded. No guarantee of 20Hz. Commanded angles, not physical measurements.'),indent=2))
        threading.Thread(target=self.capture,daemon=True).start()

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
        return dict(connected=self.arm.port is not None,running=self.running(),message=message,
            joints=self.last_state or [], detections=self.frozen.tolist() if self.frozen is not None else None,
            prediction_deg=self.last_prediction,steps=self.completed_steps,camera_error=self.camera_error,
            actual_update_median_ms=float(np.median(self.timings)*1000) if self.timings else None,
            deadline_misses=self.misses,output=str(self.output),trial_number=self.trial_number)

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
        self.frozen = None; self.boxes = None; self.last_prediction = None
        self.completed_steps = 0; self.timings = []; self.misses = 0; self.returning = False
        self.control_elapsed_s = 0
        self.message = '次の試行です。開始姿勢へ戻し、スポンジとcoinを置き直して初期検出を確定してください。'

    def refresh(self):
        with self.arm.lock:
            self.last_state = self.arm.status()
        return self.last_state

    def detect(self):
        if self.completed_steps:
            raise ValueError('配置を変える場合は「次の試行」を押してから初期検出してください')
        self.refresh()
        before = [(s['angle'],s['target'],s['active']) for s in self.last_state]
        if any(a != b for a,b,_ in before): raise ValueError('移動を停止してから初期検出してください')
        boxes = []; frames = []; sequences = []; deadline = time.monotonic()+10
        while len(boxes) < 5:
            if time.monotonic() > deadline: raise RuntimeError('異なる開始フレームを5枚取得できませんでした')
            frame,seq = self.snapshot()
            if seq in sequences: time.sleep(.02); continue
            with self.detector_lock:
                result = self.detector.predict(frame,imgsz=640,conf=.25,device='cpu',verbose=False)[0]
            selected = []
            for name in ['sponge','coin']:
                candidates = [b for b in result.boxes if result.names[int(b.cls[0])]==name]
                if not candidates: raise ValueError(f'{name}を検出できません。配置を確認してください')
                best = max(candidates,key=lambda b:float(b.conf[0]))
                selected.append([*best.xyxy[0].cpu().tolist(),float(best.conf[0])])
            boxes.append(selected);frames.append(frame);sequences.append(seq)
        self.refresh()
        if before != [(s['angle'],s['target'],s['active']) for s in self.last_state]:
            raise ValueError('検出中に関節指令が変わりました。やり直してください')
        raw = np.asarray(boxes,dtype=np.float32); median = np.median(raw,axis=0)
        centers = (raw[:,:,:2]+raw[:,:,2:4])/2
        if np.linalg.norm(centers-np.median(centers,axis=0),axis=2).max() > 8:
            raise ValueError('検出位置が安定しません。再取得してください')
        features = []
        for x1,y1,x2,y2,score in median:
            features.append([(x1+x2)/1280,(y1+y2)/960,(x2-x1)/640,(y2-y1)/480,score,1])
        self.frozen = np.asarray(features,dtype=np.float32);self.boxes=median
        self.completed_steps=0;self.timings=[];self.last_prediction=None
        stamp = time.time_ns()
        cv2.imwrite(str(self.output/f'initial_{stamp}.jpg'),frames[0])
        (self.output/f'initial_{stamp}.json').write_text(json.dumps(dict(features=self.frozen.tolist(),
            raw_boxes=raw.tolist(),frame_sequences=sequences,state=self.last_state),indent=2))
        self.message = '初期検出を固定しました。動作中は物体を置き直さないでください。'

    def predict(self):
        if self.frozen is None: raise ValueError('先に初期検出を確定してください')
        self.snapshot()
        self.refresh()
        if not all(s['active'] for s in self.last_state): raise ValueError('全関節を有効にしてください')
        qpos = torch.tensor([s['angle'] for s in self.last_state],dtype=torch.float32)*torch.pi/180
        normalized = (qpos-self.stats['qpos_mean'])/self.stats['qpos_std']
        with torch.inference_mode():
            action = self.policy(normalized[None],torch.from_numpy(self.frozen)[None])[0,0]
            degrees = (action*self.stats['action_std']+self.stats['action_mean'])*180/torch.pi
        if not torch.isfinite(degrees).all(): raise RuntimeError('非有限な予測値です')
        self.last_prediction = degrees.tolist()
        return degrees.numpy(),np.array([s['angle'] for s in self.last_state])

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

    def after_step(self):
        pass

    def targets_for_prediction(self,prediction,current):
        return bounded_targets(prediction,current)

    def run(self,seconds,step_limit=None):
        self.message='モデルで動作中'
        if step_limit is None:step_limit=1 if seconds==0 else None
        deadline=time.monotonic()+(1800 if step_limit is not None else seconds)
        run_steps=0
        try:
            while not self.stop.is_set():
                started=time.monotonic()
                if started >= deadline: break
                prediction,current=self.predict()
                if self.stop.is_set(): break
                target=self.targets_for_prediction(prediction,current)
                self.move_batch(target)
                self.after_step()
                duration=time.monotonic()-started
                self.timings.append(duration);self.misses+=int(duration>.05)
                self.completed_steps+=1
                run_steps+=1
                with (self.output/'steps.jsonl').open('a') as f:
                    record=dict(index=self.completed_steps,time_ns=time.time_ns(),
                        qpos_deg=current.tolist(),prediction_deg=prediction.tolist(),sent_deg=target.tolist(),
                        compute_and_send_s=duration)
                    if self.backend is None:record['frozen_detections']=self.frozen.tolist()
                    else:record['input_mode']=self.backend.config['mode']
                    f.write(json.dumps(record)+'\n')
                if step_limit is not None and run_steps>=step_limit:
                    self.control_elapsed_s=getattr(self,'control_elapsed_s',0)+time.monotonic()-started
                    break
                self.stop.wait(max(0,.05-(time.monotonic()-started)))
                self.control_elapsed_s=getattr(self,'control_elapsed_s',0)+time.monotonic()-started
            self.message='停止・保持しました。'
        except Exception as e:
            self.message='停止: '+str(e)
        finally:
            with self.arm.lock:
                try: self.arm.command('STOP');self.last_state=self.arm.status()
                except Exception as e: self.message+=' / 停止応答: '+str(e)

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
            if action=='connect':
                result=self.arm.action(dict(action='connect',port=self.port));self.last_state=result['joints']
                self.frozen=None;self.boxes=None; self.message='接続しました。実際の姿勢に合う角度で有効にしてください。'
            elif action=='manual':
                command=data['command']
                if command.get('action') not in ['start','fine_step','status','speed']: raise ValueError('不正な手動操作')
                result=fine_step(self.arm.action,command) if command['action']=='fine_step' else self.arm.action(command)
                self.last_state=result['joints'];self.message=result.get('message','')
            elif action=='detect': self.detect()
            elif action=='next_trial': self.next_trial()
            elif action=='return_initial':
                self.returner.start();self.returning=True
            elif action=='preview': self.predict();self.message='予測だけ表示しました。動作指令は送っていません。'
            elif action=='run':
                seconds=data.get('seconds')
                step_limit=data.get('steps')
                if step_limit is not None:
                    if self.backend is None or type(step_limit) is not int or step_limit not in [1,10,1000]:
                        raise ValueError('試験更新回数が不正です')
                    seconds=0
                elif seconds not in [0,1,10]: raise ValueError('試験時間が不正です')
                if self.frozen is None: raise ValueError('先に初期検出を確定してください')
                self.refresh()
                if not all(s['active'] for s in self.last_state): raise ValueError('全関節を有効にしてください')
                if self.completed_steps==0 and np.max(np.abs(np.array([s['angle'] for s in self.last_state])-INITIAL))>2:
                    raise ValueError('最初の試験は学習時の開始姿勢70,113,43,80,90度から行ってください')
                if self.last_state[3]['angle']!=80: raise ValueError('手首を80度にしてください')
                with self.arm.lock:
                    for i in range(5): self.arm.command(f'SPEED {i+1} 25')
                self.stop.clear()
                self.worker=threading.Thread(target=self.run,args=(seconds,step_limit),daemon=True);self.worker.start()
            else: raise ValueError('不正な操作')


def serve_trial(trial, http_port, enable_angles=None, page_override=None):
    token=secrets.token_urlsafe(32)
    page=page_override if page_override is not None else Path(__file__).with_suffix('.html').read_text()
    page=page.replace('__TOKEN__',token)
    page=page.replace('__ENABLE_ANGLES__',json.dumps(enable_angles or INITIAL.tolist()))
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
                    if trial.boxes is not None:
                        for name,box in zip(['sponge initial','coin initial'],trial.boxes):
                            x1,y1,x2,y2=map(int,box[:4]);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,180,255),2)
                            cv2.putText(frame,name,(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,255),1)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'runs/act_initial_detections_cpu_20261107/inference_bundle.pt')
    parser.add_argument('--port',default='/dev/ttyUSB0');parser.add_argument('--http-port',type=int,default=8767)
    parser.add_argument('--enable-angles',type=int,nargs=5,help='Known current commanded pose for manual re-enabling after restart')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    trial=Trial(args.bundle,args.port)
    serve_trial(trial,args.http_port,args.enable_angles)


if __name__=='__main__':main()
