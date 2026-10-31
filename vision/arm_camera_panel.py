"""既存の操作サーバーを維持したまま、ライブ映像付き画面を提供する。"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit, parse_qs
from urllib.request import urlopen, Request

import cv2
import numpy as np
if __package__:
    from .table_geometry import load_calibration, project
else:
    from table_geometry import load_calibration, project

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from control.fine_step import fine_step
from control.arm_calibration import apply_hover_condition
from vision.arm_position_test import position_preview, save_position_test
from vision.arm_position_test import stationary_commands
from control.pick_place_plan import build_transfer
from control.imitation_record import DemonstrationRecorder
from vision.live_detector import LiveDetector
from control.imitation_policy import ImitationPolicy
from control.imitation_trial import PolicyTrial
from control.base_alignment import BaseAlignment
from control.motion_order import MotionOrder


class LiveCamera:
    def __init__(self,source,expected_size=None):
        self.source=source
        self.expected_size=expected_size
        self.stop_event=threading.Event()
        self.condition=threading.Condition()
        self.frame=None
        self.sequence=0
        self.error=None
        self.closed=False
        self.captured_at=None
        self.captured_monotonic=0
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        while not self.closed:
            camera=None
            try:
                if not self.source.isdigit() and not Path(self.source).exists():
                    raise RuntimeError('Webカメラが未接続です。USBを差し直すと自動復旧します。')
                camera=cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source,cv2.CAP_V4L2)
                if not camera.isOpened():
                    raise RuntimeError('カメラを開けません。他のカメラアプリを終了してください。')
                for prop,value in [(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),
                                   (cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),(cv2.CAP_PROP_BUFFERSIZE,1)]:
                    camera.set(prop,value)
                while not self.closed:
                    ok,frame=camera.read()
                    if not ok:
                        raise RuntimeError('カメラの接続が切れました。再接続を待っています。')
                    if self.expected_size and [frame.shape[1],frame.shape[0]]!=list(self.expected_size):
                        raise RuntimeError('カメラ解像度が校正時と異なります。')
                    with self.condition:
                        self.frame=frame
                        self.error=None
                        self.sequence+=1
                        self.captured_at=datetime.now(timezone.utc).isoformat()
                        self.captured_monotonic=time.monotonic()
                        self.condition.notify_all()
            except Exception as error:
                with self.condition:
                    self.error=str(error)
                    self.frame=None
                    self.captured_monotonic=0
                    self.condition.notify_all()
            finally:
                if camera is not None:camera.release()
            if not self.closed:self.stop_event.wait(1)

    def snapshot(self,fresh=False):
        with self.condition:
            initial=self.sequence
            ready=self.condition.wait_for(lambda:self.error is not None or
                (self.frame is not None and (not fresh or self.sequence>initial)),timeout=3)
            if not ready or self.error or time.monotonic()-self.captured_monotonic>1:
                raise RuntimeError(self.error or '新しいカメラ画像を取得できません。')
            return self.frame.copy(),self.captured_at,self.sequence

    def close(self):
        self.closed=True
        self.stop_event.set()
        self.thread.join(timeout=2)


def camera_markup():
    return '''<style>
body{display:grid;grid-template-columns:minmax(540px,1fr) minmax(320px,.6fr);gap:18px}
main{min-width:0;width:100%;max-width:none;grid-column:1;grid-row:1}
.live-camera{grid-column:2;grid-row:1;position:sticky;top:20px;align-self:start;background:white;padding:16px;border:1px solid #d8e0e9;border-radius:14px}
.live-camera img{display:block;width:100%;height:auto;background:#172b40;min-height:160px}
.live-camera p{font-size:14px}.live-camera h2{margin-bottom:12px}
@media(max-width:900px){body{display:block}.live-camera{position:sticky;top:0;z-index:3}.live-camera img{max-height:36vh;object-fit:contain}}
</style><aside class="live-camera"><h2>ライブカメラ</h2><img id="live-camera" alt="現在のカメラ映像">
<p id="live-camera-state" role="status">映像を接続中…</p>
<p>スポンジは緑、コイン（ターゲット）は紫の検出枠で表示します。数値は検出の信頼度です。記録用画像は枠のない元画像を保存します。</p></aside>'''


def camera_script():
    return '''<script>
const liveImage=document.getElementById('live-camera');
const liveState=document.getElementById('live-camera-state');
function updateCamera(){liveImage.src='/live.jpg?t='+Date.now();}
liveImage.onload=()=>{liveState.textContent='ライブ映像 更新 '+new Date().toLocaleTimeString();setTimeout(updateCamera,120);};
liveImage.onerror=()=>{liveState.textContent='映像が更新できません。表示中の画像は現在の状態ではありません。';setTimeout(updateCamera,1000);};
updateCamera();
// 状態だけを取得し、既存の接続・有効角度を引き継ぐ。connectは送らない。
enqueue({action:'status'}).catch(()=>{});
</script>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-url',default='http://127.0.0.1:8765')
    parser.add_argument('--http-port',type=int,default=8766)
    parser.add_argument('--calibration',type=Path,default=ROOT/'configs/table_calibration.json')
    parser.add_argument('--position-model',type=Path,default=ROOT/'runs/arm_kinematics/model.json')
    args=parser.parse_args()
    if not re.fullmatch(r'http://127\.0\.0\.1:\d+',args.arm_url) or args.arm_url.endswith(':'+str(args.http_port)):
        parser.error('別ポートのローカル操作サーバーを指定してください。')
    # Prove that the existing panel is available before taking the camera.
    with urlopen(args.arm_url,timeout=3) as response:
        upstream_page=response.read().decode()
        if 'calibration-section' not in upstream_page:
            parser.error('校正記録用の操作サーバーを先に起動してください。')
        panel_token=re.search(r"'X-Arm-Token':'([^']+)'",upstream_page).group(1)
    saved=json.loads(args.calibration.read_text())
    source=saved['camera']
    camera=LiveCamera(source,saved['image_size'])
    detector=LiveDetector(camera,ROOT/'runs/sponge_comparison/yolo11n/weights/best.pt')
    server=None
    try:
        calibration=load_calibration(args.calibration,source,tuple(saved['image_size']))
        targets={p['number']:p['expected_mm'] for p in calibration['points']}
        inverse=np.linalg.inv(calibration['matrix'])
        fine_lock=threading.RLock()
        def arm_send(action):
            request=Request(args.arm_url+'/api',data=json.dumps(action).encode(),
                headers={'Content-Type':'application/json','X-Arm-Token':panel_token})
            try:
                with urlopen(request,timeout=5) as response:return json.loads(response.read())
            except HTTPError as error:
                detail=json.loads(error.read())
                raise ValueError(detail.get('error','操作を実行できません。')) from error
        recorder=DemonstrationRecorder(ROOT/'data/imitation',camera,arm_send)
        trial=None
        class Handler(BaseHTTPRequestHandler):
            def send(self,code,content,mime,headers=None):
                self.send_response(code)
                self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(content)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Frame-Options','DENY')
                for key,value in (headers or {}).items():self.send_header(key,str(value))
                self.end_headers()
                try:self.wfile.write(content)
                except (BrokenPipeError,ConnectionResetError):pass

            def relay(self,method):
                with fine_lock:
                    self.relay_locked(method)

            def relay_locked(self,method):
                headers={}
                body=None
                if method=='POST':
                    try:length=int(self.headers.get('Content-Length','0'))
                    except ValueError:length=0
                    if not 0<length<=1024:
                        self.send(400,b'{}','application/json');return
                    body=self.rfile.read(length)
                    headers={'Content-Type':'application/json','X-Arm-Token':self.headers.get('X-Arm-Token','')}
                if method=='POST':
                    try:
                        data=json.loads(body)
                    except (ValueError,TypeError):
                        data=None
                    if recorder.active and (not isinstance(data,dict) or data.get('action') not in ('status','stop')):
                        self.send(409,json.dumps({'error':'デモ記録中は模倣学習の記録用矢印を使ってください。'}).encode(),'application/json');return
                    if isinstance(data,dict) and data.get('action')=='fine_step':
                        self.handle_fine(data,headers);return
                request=Request(args.arm_url+self.path,data=body,headers=headers,method=method)
                try:
                    with urlopen(request,timeout=20) as response:
                        content=response.read();mime=response.headers.get('Content-Type','application/octet-stream')
                        if method=='POST' and isinstance(data,dict) and data.get('action')=='record_calibration':
                            result=json.loads(content)
                            folder=Path(result['calibration_sample']['path']).resolve()
                            if not folder.is_relative_to((ROOT/'runs/arm_calibration').resolve()):
                                raise ValueError('記録先が不正です。')
                            apply_hover_condition(folder/'sample.json')
                        if self.path=='/':
                            upstream_page=content.decode()
                            token=re.search(r"'X-Arm-Token':'([^']+)'",upstream_page).group(1)
                            page=(ROOT/'control/arm_panel.html').read_text().replace('__TOKEN__',token)
                            page=page.replace('__CALIBRATION_POINTS__',json.dumps([{'number':n,'xy':xy} for n,xy in sorted(targets.items())]))
                            page=page.replace('<main>',camera_markup()+'<main>',1)
                            if args.position_model.exists():
                                for filename in ['arm_position_controls.html','transfer_controls.html']:
                                    controls=(ROOT/'vision'/filename).read_text().replace('__TOKEN__',token)
                                    markup,script=controls.split('<script>',1)
                                    page=page.replace('</main>',markup+'</main>',1)
                                    page=page.replace('</html>','<script>'+script+'</html>')
                            controls=(ROOT/'vision/imitation_controls.html').read_text().replace('__TOKEN__',token)
                            markup,script=controls.split('<script>',1)
                            page=page.replace('<main>','<main>'+markup,1)
                            page=page.replace('</html>','<script>'+script+'</html>')
                            controls=(ROOT/'vision/imitation_trial_controls.html').read_text().replace('__TOKEN__',token)
                            markup,script=controls.split('<script>',1)
                            page=page.replace('<main>','<main>'+markup,1)
                            page=page.replace('</html>','<script>'+script+'</html>')
                            content=page.replace('</html>',camera_script()+'</html>').encode()
                        self.send(response.status,content,mime)
                except HTTPError as error:
                    self.send(error.code,error.read(),error.headers.get('Content-Type','application/json'))
                except OSError:
                    self.send(502,json.dumps({'error':'操作サーバーに接続できません。'}).encode(),'application/json')

            def handle_fine(self,data,headers):
                def send(action):
                    request=Request(args.arm_url+'/api',data=json.dumps(action).encode(),headers=headers,method='POST')
                    try:
                        with urlopen(request,timeout=5) as response:return json.loads(response.read())
                    except HTTPError as error:
                        detail=json.loads(error.read())
                        raise ValueError(detail.get('error','操作を実行できません。')) from error
                try:
                    with fine_lock:state=fine_step(send,data)
                    self.send(200,json.dumps(state,ensure_ascii=False).encode(),'application/json')
                except (ValueError,RuntimeError,OSError) as error:
                    self.send(400,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

            def do_GET(self):
                path=urlsplit(self.path)
                try:
                    if path.path=='/camera-info':
                        self.send(200,json.dumps({'service':'arm-calibration-live-camera-v1','camera':source}).encode(),'application/json');return
                    if path.path in ('/live.jpg','/snapshot.png'):
                        if path.path=='/live.jpg':
                            frame,stamp,sequence=detector.snapshot()
                            ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85]);mime='image/jpeg'
                        else:
                            frame,stamp,sequence=camera.snapshot(fresh=True)
                            ok,encoded=cv2.imencode('.png',frame);mime='image/png'
                        if not ok:raise RuntimeError('画像を変換できません。')
                        self.send(200,encoded.tobytes(),mime,{'X-Camera-Source':source,'X-Frame-Captured-At':stamp,'X-Frame-Sequence':sequence});return
                    if self.path=='/' or re.fullmatch(r'/calibration-image/[A-Za-z0-9_]+',self.path):
                        self.relay('GET');return
                    self.send(404,b'{}','application/json')
                except (ValueError,RuntimeError) as error:
                    self.send(503,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

            def do_POST(self):
                nonlocal trial
                if self.path=='/policy':
                    if self.headers.get('X-Arm-Token')!=panel_token:
                        self.send(403,b'{}','application/json');return
                    try:
                        length=int(self.headers.get('Content-Length','0'))
                        if not 0<length<=1024:raise ValueError('入力の長さが不正です。')
                        data=json.loads(self.rfile.read(length))
                        with fine_lock:
                            if recorder.active:raise ValueError('デモ記録を終了してからAI試行を使ってください。')
                            if trial is None:
                                guide_path=ROOT/'runs/imitation_bc/base_alignment.json'
                                trial=PolicyTrial(ImitationPolicy(ROOT/'runs/imitation_bc/policy.pt'),detector,arm_send,
                                    BaseAlignment(guide_path) if guide_path.exists() else None,MotionOrder())
                            if data.get('action')=='preview':result=trial.preview(data.get('phase'),use_base_alignment=data.get('base_alignment',True) is True,
                                use_motion_order=data.get('motion_order',True) is True)
                            elif data.get('action')=='advance_order':result=trial.advance_order()
                            elif data.get('action')=='confirm_lift':result=trial.confirm_lift()
                            elif data.get('action')=='reset_order':result=trial.reset_order()
                            elif data.get('action')=='execute':result=trial.execute(data.get('ticket'))
                            else:raise ValueError('AI試行の操作が不正です。')
                        self.send(200,json.dumps(result,ensure_ascii=False).encode(),'application/json')
                    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
                        self.send(400,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')
                elif self.path=='/demo':
                    if self.headers.get('X-Arm-Token')!=panel_token:
                        self.send(403,b'{}','application/json');return
                    try:
                        length=int(self.headers.get('Content-Length','0'))
                        if not 0<length<=4096:raise ValueError('入力の長さが不正です。')
                        data=json.loads(self.rfile.read(length))
                        with fine_lock:
                            action=data.get('action')
                            if action=='status':result=recorder.info()
                            elif action=='start':result=recorder.start(data.get('note',''))
                            elif action=='phase':result=recorder.set_phase(data.get('phase'))
                            elif action=='step':result=recorder.action(data.get('joint'),data.get('direction'),degrees=data.get('degrees',1))
                            elif action=='jog':result=recorder.jog(data.get('joint'),data.get('direction'))
                            elif action=='stop_jog':result=recorder.stop_jog()
                            elif action=='wait':result=recorder.action(wait=True)
                            elif action=='finish':result=recorder.finish(data.get('outcome'),data.get('note',''))
                            else:raise ValueError('記録操作が不正です。')
                        self.send(200,json.dumps(result,ensure_ascii=False).encode(),'application/json')
                    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
                        self.send(400,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')
                elif recorder.active and self.path!='/api':
                    self.send(409,json.dumps({'error':'デモ記録を終了してから操作してください。'}).encode(),'application/json')
                elif self.path=='/api':self.relay('POST')
                elif self.path in ('/transfer/point','/transfer/preview','/transfer/record'):
                    if self.headers.get('X-Arm-Token')!=panel_token:
                        self.send(403,b'{}','application/json');return
                    try:
                        length=int(self.headers.get('Content-Length','0'))
                        if not 0<length<=1024:raise ValueError('入力の長さが不正です。')
                        data=json.loads(self.rfile.read(length))
                        def status():
                            request=Request(args.arm_url+'/api',data=b'{"action":"status"}',
                                headers={'Content-Type':'application/json','X-Arm-Token':panel_token})
                            with urlopen(request,timeout=5) as response:return json.loads(response.read())
                        if self.path.endswith('/point'):
                            uv=[float(data['u']),float(data['v'])]
                            if not np.isfinite(uv).all() or cv2.pointPolygonTest(np.array(calibration['corners_px'],dtype=np.float32),tuple(uv),False)<0:
                                raise ValueError('青枠内の紙面位置を指定してください。')
                            result={'xy_mm':project(calibration['matrix'],[uv])[0].tolist()}
                        elif self.path.endswith('/preview'):
                            with fine_lock:
                                before=status();commands=stationary_commands(before)
                                plan=build_transfer(args.position_model,data['pick_xy'],data['place_xy'],commands,
                                    float(data.get('pick_z',15)),float(data.get('travel_z',40)),float(data.get('release_z',15)))
                                frame,stamp,sequence=camera.snapshot(fresh=True)
                                if commands!=stationary_commands(status()):raise ValueError('計算中に指令値が変わりました。')
                                plan_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                                folder=ROOT/'runs/pick_place'/plan_id
                                folder.mkdir(parents=True,exist_ok=False)
                                ok,encoded=cv2.imencode('.png',frame)
                                if not ok:raise RuntimeError('写真を変換できません。')
                                (folder/'setup.png').write_bytes(encoded.tobytes())
                                plan.update(plan_id=plan_id,setup_frame_captured_at=stamp,setup_frame_sequence=sequence)
                                (folder/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
                                result=plan
                        else:
                            plan_id=data.get('plan_id','');index=data.get('index')
                            if not isinstance(plan_id,str) or not re.fullmatch(r'\d{8}T\d{6}_\d{6}Z',plan_id):raise ValueError('計画IDが不正です。')
                            plan=json.loads((ROOT/'runs/pick_place'/plan_id/'plan.json').read_text())
                            if type(index) is not int or not 0<=index<len(plan['stages']) or data.get('confirmed') is not True:
                                raise ValueError('現在の段階と静止を確認してください。')
                            stage=dict(plan['stages'][index],model_sha256=plan['model_sha256'],plan_id=plan_id,stage_index=index)
                            with fine_lock:
                                before=status();time.sleep(1)
                                frame,stamp,sequence=camera.snapshot(fresh=True)
                                after=status()
                                folder=save_position_test(ROOT/'runs/pick_place'/plan_id/f'stage_{index+1:02d}',stage,before,after,
                                    frame,stamp,sequence,calibration)
                            result={'path':str(folder)}
                        self.send(200,json.dumps(result,ensure_ascii=False).encode(),'application/json')
                    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
                        self.send(400,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')
                elif self.path in ('/position-test/preview','/position-test/record'):
                    if self.headers.get('X-Arm-Token')!=panel_token:
                        self.send(403,b'{}','application/json');return
                    try:
                        length=int(self.headers.get('Content-Length','0'))
                        if not 0<length<=1024:raise ValueError('入力の長さが不正です。')
                        data=json.loads(self.rfile.read(length))
                        preview=position_preview(args.position_model,data['x'],data['y'])
                        if self.path.endswith('/preview'):
                            result=preview
                        else:
                            if data.get('confirmed') is not True:raise ValueError('静止を確認してください。')
                            def status():
                                request=Request(args.arm_url+'/api',data=b'{"action":"status"}',
                                    headers={'Content-Type':'application/json','X-Arm-Token':panel_token})
                                with urlopen(request,timeout=5) as response:return json.loads(response.read())
                            with fine_lock:
                                before=status()
                                time.sleep(1)
                                frame,stamp,sequence=camera.snapshot(fresh=True)
                                after=status()
                                folder=save_position_test(ROOT/'runs/arm_position_tests',preview,before,after,
                                    frame,stamp,sequence,calibration,data.get('note',''))
                            result={'path':str(folder)}
                        self.send(200,json.dumps(result,ensure_ascii=False).encode(),'application/json')
                    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
                        self.send(400,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')
                else:self.send(404,b'{}','application/json')

            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
        print(f'映像付き操作画面: http://127.0.0.1:{args.http_port}',flush=True)
        print('既存のアーム接続を維持。接続・有効化・移動指令は起動時に送信しません。',flush=True)
        server.serve_forever()
    finally:
        if server:server.server_close()
        if 'recorder' in locals() and recorder.active:
            with fine_lock:recorder.finish('aborted','カメラサーバー終了')
        detector.close()
        camera.close()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nライブカメラを終了しました。アーム操作サーバーは維持します。')
    except (OSError,RuntimeError,ValueError,cv2.error) as error:raise SystemExit(str(error))
