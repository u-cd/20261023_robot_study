"""Live camera and manual demonstration collection; no geometric calibration required."""
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
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.fine_step import fine_step
from control.imitation_record import DemonstrationRecorder


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


def page_html(token, detections=False):
    page=(ROOT/'control/arm_panel.html').read_text().replace('__TOKEN__',token)
    camera='''<section><h2>ライブカメラ</h2><img id="live-camera" alt="現在のカメラ映像" style="max-width:100%;width:640px"><p id="live-state" role="status">接続中…</p></section>'''
    if detections:camera+='<p>スポンジとcoinの検出枠を表示。保存する画像は枠のない元画像です。</p>'
    controls=(ROOT/'vision/imitation_controls.html').read_text().replace('__TOKEN__',token)
    markup,script=controls.split('<script>',1)
    page=page.replace('<main>','<main>'+camera+markup,1)
    camera_script='''<script>
const liveImage=document.getElementById('live-camera');
function updateCamera(){liveImage.src='/live.jpg?t='+Date.now();}
liveImage.onload=()=>{document.getElementById('live-state').textContent='映像更新 '+new Date().toLocaleTimeString();setTimeout(updateCamera,120);};
liveImage.onerror=()=>{document.getElementById('live-state').textContent='カメラ未接続・再接続待ち。表示中の画像は最新とは限りません。';setTimeout(updateCamera,1000);};
updateCamera();enqueue({action:'status'}).catch(()=>{});
</script>'''
    return page.replace('</html>','<script>'+script+camera_script+'</html>')


def make_handler(camera, arm_send, token, root, detector=None):
    """Factory also used with fake devices in tests. Creation sends no commands."""
    recorder=DemonstrationRecorder(root,camera,arm_send)
    lock=threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def send(self,code,content,mime='application/json'):
            if isinstance(content,dict):content=json.dumps(content,ensure_ascii=False).encode()
            elif isinstance(content,str):content=content.encode()
            self.send_response(code)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Frame-Options','DENY')
            self.end_headers()
            try:self.wfile.write(content)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path=='/':self.send(200,page_html(token,detector is not None),'text/html; charset=utf-8')
                elif path=='/camera-info':self.send(200,{'service':'arm-demonstration-camera','camera':camera.source})
                elif path in ('/live.jpg','/snapshot.png'):
                    if path=='/live.jpg' and detector is not None:frame,_,_=detector.snapshot()
                    else:frame,_,_=camera.snapshot(fresh=path=='/snapshot.png')
                    ext='.jpg' if path=='/live.jpg' else '.png'
                    ok,encoded=cv2.imencode(ext,frame)
                    if not ok:raise RuntimeError('画像の変換に失敗しました。')
                    self.send(200,encoded.tobytes(),'image/jpeg' if ext=='.jpg' else 'image/png')
                else:self.send(404,{})
            except (ValueError,RuntimeError,OSError,cv2.error) as error:self.send(503,{'error':str(error)})

        def do_POST(self):
            if self.headers.get('X-Arm-Token')!=token:self.send(403,{});return
            if self.path not in ('/api','/demo'):self.send(404,{});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('入力の長さが不正です。')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('操作が不正です。')
                with lock:
                    action=data.get('action')
                    if self.path=='/api':
                        if recorder.active and action not in ('status','stop'):
                            raise ValueError('記録中はデモ記録用の矢印を使ってください。')
                        if action=='stop' and recorder.active:result=recorder.stop_jog()['state']
                        elif action=='fine_step':result=fine_step(arm_send,data)
                        else:result=arm_send(data)
                    elif action=='status':result=recorder.info()
                    elif action=='start':result=recorder.start(data.get('note',''))
                    elif action=='phase':result=recorder.set_phase(data.get('phase'))
                    elif action=='step':result=recorder.action(data.get('joint'),data.get('direction'),degrees=data.get('degrees',1))
                    elif action=='jog':result=recorder.jog(data.get('joint'),data.get('direction'))
                    elif action=='stop_jog':result=recorder.stop_jog()
                    elif action=='wait':result=recorder.action(wait=True)
                    elif action=='finish':result=recorder.finish(data.get('outcome'),data.get('note',''))
                    else:raise ValueError('記録操作が不正です。')
                self.send(200,result)
            except (ValueError,RuntimeError,OSError,KeyError,TypeError,cv2.error) as error:
                self.send(400,{'error':str(error)})

        def log_message(self,*args):pass

    return Handler,recorder,lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-url',default='http://127.0.0.1:8765')
    parser.add_argument('--http-port',type=int,default=8766)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/camera.json')
    parser.add_argument('--detections',action='store_true',help='任意のYOLO検出枠を表示する')
    parser.add_argument('--weights',type=Path,default=ROOT/'models/yolo11n_sponge.pt')
    args=parser.parse_args()
    if not re.fullmatch(r'http://127\.0\.0\.1:\d+',args.arm_url) or args.arm_url.endswith(':'+str(args.http_port)):
        parser.error('別ポートのローカル操作サーバーを指定してください。')
    with urlopen(args.arm_url,timeout=3) as response:
        upstream=response.read().decode()
    match=re.search(r"'X-Arm-Token':'([^']+)'",upstream)
    if not match:parser.error('先にcontrol/arm_panel.pyを起動してください。')
    token=match.group(1)
    def arm_send(action):
        request=Request(args.arm_url+'/api',data=json.dumps(action).encode(),
                        headers={'Content-Type':'application/json','X-Arm-Token':token})
        try:
            with urlopen(request,timeout=10) as response:return json.loads(response.read())
        except HTTPError as error:raise ValueError(json.loads(error.read()).get('error','基板の操作に失敗しました。')) from error
    saved=json.loads(args.config.read_text())
    camera=LiveCamera(str(saved['camera']),saved.get('image_size',[640,480]))
    detector=None;server=None;recorder=None
    try:
        if args.detections:
            from vision.live_detector import LiveDetector
            detector=LiveDetector(camera,args.weights)
        handler,recorder,lock=make_handler(camera,arm_send,token,ROOT/'data/imitation',detector)
        server=ThreadingHTTPServer(('127.0.0.1',args.http_port),handler)
        print(f'実演収集画面: http://127.0.0.1:{args.http_port}',flush=True)
        print('校正は不要です。起動時に接続・有効化・移動指令は送りません。終了はCtrl+C。',flush=True)
        server.serve_forever()
    finally:
        if server:server.server_close()
        if recorder is not None and recorder.active:
            with lock:recorder.finish('aborted','収集サーバー終了')
        if detector is not None:detector.close()
        camera.close()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nカメラを終了しました。操作サーバーは別途終了してください。')
    except (OSError,RuntimeError,ValueError,cv2.error) as error:raise SystemExit(str(error))
