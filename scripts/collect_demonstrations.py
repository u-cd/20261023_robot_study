"""Camera + periodic raw demonstrations; proxy the existing manual arm server.

Does not open a serial port, enable joints, or send movement on startup.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import h5py
import numpy as np

try:
    from .return_to_initial import ReturnToInitial
except ImportError:
    from return_to_initial import ReturnToInitial

ROOT = Path(__file__).resolve().parents[1]


class ArmProxy:
    def __init__(self, url):
        self.url = url
        page = urlopen(url, timeout=3).read().decode()
        self.token = re.search(r"'X-Arm-Token':'([^']+)'", page)[1]
        self.lock = threading.RLock()

    def send(self, data):
        with self.lock:
            request = Request(self.url + '/api', json.dumps(data).encode(),
                              {'Content-Type': 'application/json', 'X-Arm-Token': self.token})
            try:
                with urlopen(request, timeout=4) as response:
                    return json.load(response)
            except HTTPError as error:
                raise ValueError(json.load(error).get('error', str(error))) from error


class Camera:
    def __init__(self, source, size):
        self.lock = threading.Lock()
        self.latest = None
        self.error = ''
        self.done = threading.Event()
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError('Webカメラを開けません。ほかのカメラアプリを閉じてください。')
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        sequence = 0
        while not self.done.is_set():
            ok, frame = self.cap.read()
            captured = time.monotonic_ns()
            if not ok:
                self.error = 'カメラの画像取得に失敗しました。'
                self.done.wait(.05)
                continue
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                with self.lock:
                    sequence += 1
                    self.latest = (sequence, captured, jpg.tobytes())
                    self.error = ''

    def snapshot(self):
        with self.lock:
            if self.latest is None or time.monotonic_ns() - self.latest[1] > 500_000_000:
                raise RuntimeError(self.error or 'カメラ画像が古いか、まだ取得できていません。')
            return self.latest

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.cap.release()


def validate_initial(state, config):
    joints = sorted(state.get('joints', []), key=lambda x: x['joint'])
    if not state.get('connected') or [j['joint'] for j in joints] != [1, 2, 3, 4, 5]:
        raise ValueError('操作画面で基板へ接続してください。')
    if not all(j['active'] and j['angle'] == j['target'] for j in joints):
        raise ValueError('全5関節を有効にし、停止・保持してから記録してください。')
    if [j['angle'] for j in joints] != config['initial_angles_deg']:
        raise ValueError('開始姿勢を70, 113, 43, 80, 90度に合わせてください。自動移動は行いません。')


def export_hdf5(folder):
    """Raw timing preserved. This is a variable-length preview, not padded training data."""
    rows = [json.loads(line) for line in (folder / 'samples.jsonl').read_text().splitlines()]
    if len(rows) < 2:
        raise ValueError('記録が短すぎます。2サンプル以上必要です。')
    qpos_deg = np.array([[j['angle'] for j in r['state']['joints']] for r in rows], dtype=np.float32)
    target_deg = np.array([[j['target'] for j in r['state']['joints']] for r in rows], dtype=np.float32)
    # Use the following observed commanded position as the imitation target.
    # During firmware JOG, target==commanded, so endpoint targets alone lose the movement.
    actions_deg = np.concatenate([qpos_deg[1:], qpos_deg[-1:]])
    qpos = np.deg2rad(qpos_deg)
    dt = np.diff([r['state_end_ns'] for r in rows]) / 1e9
    qvel = np.zeros_like(qpos)
    qvel[1:] = np.diff(qpos, axis=0) / np.maximum(dt[:, None], 1e-9)
    with h5py.File(folder / 'preview.hdf5', 'w') as f:
        f.attrs.update(sim=False, angle_source='commanded_not_measured',
                       joint_units='radian_all_5_including_gripper',
                       action_semantics='next_sample_commanded_position; final repeats',
                       training_ready=False,
                       warning='Variable timing/length. Resample and resolve official real-data action shift before training.')
        f.create_dataset('observations/qpos', data=qpos)
        f.create_dataset('observations/qvel', data=qvel)
        f.create_dataset('action', data=np.deg2rad(actions_deg))
        f.create_dataset('raw/qpos_deg', data=qpos_deg)
        f.create_dataset('raw/firmware_target_deg', data=target_deg)
        for key in ('frame_sequence', 'frame_captured_ns', 'state_start_ns', 'state_end_ns', 'tick'):
            f.create_dataset('timestamps/' + key, data=[r[key] for r in rows])
        first = cv2.imread(str(folder / rows[0]['image']))
        h, w = first.shape[:2]
        images = f.create_dataset('observations/images/top', (len(rows), h, w, 3),
                                  dtype='uint8', chunks=(1, h, w, 3), compression='lzf')
        for i, row in enumerate(rows):
            bgr = cv2.imread(str(folder / row['image']))
            if bgr is None or bgr.shape != first.shape:
                raise ValueError('保存画像のサイズ不一致または欠損があります。')
            images[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    viewer_rows = [{'image': r['image'], 'time_s': (r['state_end_ns']-rows[0]['state_start_ns'])/1e9,
                    'angles_deg': qpos_deg[i].tolist()} for i, r in enumerate(rows)]
    viewer = """<!doctype html><meta charset="utf-8"><title>実演の生記録</title>
<style>body{font:18px system-ui;max-width:1000px;margin:30px auto}img{max-width:100%}input{width:100%}</style>
<h1>実演の画像・指令角度</h1><p>角度は実測値ではありません。スライダーで時系列を確認します。</p>
<img id="frame"><input id="slider" type="range" min="0" value="0"><pre id="info"></pre>
<script>const rows=__ROWS__;const slider=document.getElementById('slider');slider.max=rows.length-1;
function show(){const r=rows[Number(slider.value)];document.getElementById('frame').src=r.image;
document.getElementById('info').textContent=JSON.stringify({sample:Number(slider.value),...r},null,2);}
slider.oninput=show;show();</script>""".replace('__ROWS__', json.dumps(viewer_rows))
    (folder/'viewer.html').write_text(viewer)
    gaps = np.diff([r['state_start_ns'] for r in rows]) / 1e9
    unique = len(set(r['frame_sequence'] for r in rows))
    return {'samples': len(rows), 'duration_s': (rows[-1]['state_end_ns']-rows[0]['state_start_ns'])/1e9,
            'mean_interval_ms': float(gaps.mean()*1000), 'max_interval_ms': float(gaps.max()*1000),
            'unique_frames': unique, 'reused_frames': len(rows)-unique,
            'skipped_ticks': rows[-1]['tick']+1-len(rows), 'training_ready': False}


class Recorder:
    def __init__(self, camera, send, config, root, hz=50):
        self.camera, self.send, self.config, self.root = camera, send, config, Path(root)
        self.hz = hz
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.folder = None
        self.samples = 0
        self.error = ''
        self.result = None
        self.recording = False

    def start(self):
        with self.lock:
            if self.recording or self.folder is not None:
                raise ValueError('現在の試行を保存または破棄してください。')
            state = self.send({'action': 'status'})
            validate_initial(state, self.config)
            self.camera.snapshot()
            folder = self.root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            (folder / 'frames').mkdir(parents=True)
            self.folder = folder
            self.samples, self.error, self.result = 0, '', None
            metadata = {'started_at': datetime.now(timezone.utc).isoformat(), 'config': self.config,
                        'initial_state': state, 'requested_hz': self.hz, 'status': 'recording',
                        'clock': 'monotonic_ns; frame time is read-return time, not hardware exposure time'}
            (folder / 'episode.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            self.stop_event.clear()
            self.recording = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def run(self):
        origin = time.monotonic()
        tick = 0
        seen = set()
        try:
            with (self.folder / 'samples.jsonl').open('w') as log:
                while not self.stop_event.is_set():
                    self.stop_event.wait(max(0, origin + tick/self.hz - time.monotonic()))
                    if self.stop_event.is_set():
                        break
                    # Snapshot before requesting status; timestamps bracket serial observation.
                    sequence, captured, jpg = self.camera.snapshot()
                    start = time.monotonic_ns()
                    state = self.send({'action': 'status'})
                    end = time.monotonic_ns()
                    joints = sorted(state.get('joints', []), key=lambda x: x['joint'])
                    if not state.get('connected') or len(joints) != 5 or not all(j['active'] for j in joints):
                        raise RuntimeError('関節の接続・保持が失われたため記録を停止しました。')
                    state['joints'] = joints
                    name = f'frames/{sequence:09d}.jpg'
                    if sequence not in seen:
                        (self.folder / name).write_bytes(jpg)
                        seen.add(sequence)
                    row = {'index': self.samples, 'tick': tick, 'image': name,
                           'frame_sequence': sequence, 'frame_captured_ns': captured,
                           'state_start_ns': start, 'state_end_ns': end, 'state': state}
                    log.write(json.dumps(row, ensure_ascii=False)+'\n')
                    log.flush()
                    self.samples += 1
                    tick = max(tick+1, int((time.monotonic()-origin)*self.hz)+1)
        except Exception as error:
            self.error = str(error)
            self.stop_event.set()

    def event(self, request, response, start, end):
        with self.lock:
            if self.recording and self.folder:
                with (self.folder / 'commands.jsonl').open('a') as f:
                    f.write(json.dumps({'request': request, 'response': response,
                                        'sent_ns': start, 'acknowledged_ns': end}, ensure_ascii=False)+'\n')

    def finish(self, success):
        with self.lock:
            if self.folder is None:
                raise ValueError('記録中の試行がありません。')
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=8)
                if self.thread.is_alive():
                    raise RuntimeError('記録処理の停止を待っています。もう一度試してください。')
            self.recording = False
            if not success:
                shutil.rmtree(self.folder)
                self.result = {'discarded': True}
            else:
                if self.error:
                    raise ValueError('記録エラーがあります。保存せず破棄してください：'+self.error)
                self.result = export_hdf5(self.folder)
                self.result['folder'] = str(self.folder.relative_to(ROOT)) if self.folder.is_relative_to(ROOT) else str(self.folder)
                metadata = json.loads((self.folder / 'episode.json').read_text())
                metadata.update(status='success', summary=self.result)
                (self.folder / 'episode.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            self.folder = None
            return self.result

    def status(self):
        return {'recording': self.recording, 'samples': self.samples, 'error': self.error,
                'folder': str(self.folder) if self.folder else None, 'result': self.result}


def finish_recording(recorder, send, success):
    """A disconnected board must not prevent discarding a failed trial."""
    warning = None
    try:
        send({'action': 'stop'})
    except Exception as error:
        if success:
            raise
        warning = str(error)
    result = recorder.finish(success)
    if warning:
        result['stop_warning'] = '基板の停止応答は確認できません：' + warning
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--http-port', type=int, default=8766)
    parser.add_argument('--arm-url', default='http://127.0.0.1:8765')
    parser.add_argument('--hz', type=float, default=None)
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/demonstration.json').read_text())
    args.hz = args.hz if args.hz is not None else config.get('recording_hz', 20)
    if not 1 <= args.hz <= 50:
        parser.error('--hz must be 1..50')
    cam_config = json.loads((ROOT/'configs/camera.json').read_text())
    arm = ArmProxy(args.arm_url)
    camera = Camera(cam_config['camera'], cam_config['image_size'])
    recorder = Recorder(camera, arm.send, config, ROOT/'data/act_demonstrations_raw', args.hz)
    returner = ReturnToInitial(arm.send, config['initial_angles_deg'], arm.lock)
    transition_lock = threading.RLock()
    token = secrets.token_urlsafe(32)
    page = (ROOT/'scripts/arm_panel.html').read_text().replace('__TOKEN__', token)
    page = page.replace('const defaultAngles=[90,0,0,80,90];', 'const defaultAngles='+json.dumps(config['initial_angles_deg'])+';')
    extra = (ROOT/'scripts/collect_demonstrations.html').read_text().replace('__TOKEN__', token).replace('__HZ__', str(args.hz))
    page = page.replace('<div class="grid-wrap">', extra+'\n<div class="grid-wrap">')
    page += "<script>request({action:'status'}).catch(e=>show(e.message,true));</script>"

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, mime='application/json', code=200):
            body = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False).encode() if mime=='application/json' else value.encode())
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == '/': self.reply(page, 'text/html; charset=utf-8')
                elif self.path.startswith('/live.jpg'): self.reply(camera.snapshot()[2], 'image/jpeg')
                elif self.path == '/record/status':
                    self.reply({**recorder.status(), 'home': returner.status()})
                else: self.reply({}, code=404)
            except Exception as error:
                self.reply({'error': str(error)}, code=400)

        def do_POST(self):
            if self.headers.get('X-Arm-Token') != token:
                self.reply({}, code=403);return
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 2048: raise ValueError('不正な入力です。')
                data = json.loads(self.rfile.read(size))
                if self.path == '/home':
                    with transition_lock:
                        if data.get('action') == 'start':
                            if recorder.folder is not None:
                                raise ValueError('試行を保存または破棄してから開始姿勢へ戻してください。')
                            returner.start()
                        elif data.get('action') == 'stop':
                            returner.cancel()
                            arm.send({'action':'stop'})
                        else:
                            raise ValueError('未知の復帰操作です。')
                    self.reply(returner.status())
                elif self.path == '/api':
                    with transition_lock:
                        if data.get('action') == 'stop':
                            returner.cancel()
                        elif returner.status()['active'] and data.get('action') != 'status':
                            raise ValueError('開始姿勢への移動中です。停止してから手動操作してください。')
                        if recorder.recording and data.get('action') in ('connect','disconnect','start','off'):
                            raise ValueError('記録中は再接続・有効化・保持解除を行えません。先に記録を終了してください。')
                        start = time.monotonic_ns()
                        try:
                            result = arm.send(data)
                        except Exception as error:
                            recorder.event(data, {'error': str(error)}, start, time.monotonic_ns())
                            raise
                        if data.get('action') != 'status': recorder.event(data,result,start,time.monotonic_ns())
                        self.reply(result)
                elif self.path == '/record':
                    if data['action'] == 'start':
                        with transition_lock:
                            if returner.status()['active']:
                                raise ValueError('開始姿勢への移動が終わってから記録してください。')
                            recorder.start()
                        self.reply(recorder.status())
                    elif data['action'] in ('save','discard'):
                        # Stop JOG without detaching joints before ending the record.
                        self.reply(finish_recording(recorder, arm.send, data['action']=='save'))
                    else: raise ValueError('未知の記録操作です。')
                else: self.reply({}, code=404)
            except Exception as error: self.reply({'error':str(error)}, code=400)

        def log_message(self, *args): pass

    server = ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
    print(f'収集画面 http://127.0.0.1:{args.http_port} / 目標{args.hz:g}Hz。起動時のアーム操作なし。',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        returner.cancel()
        if returner.thread: returner.thread.join(timeout=5)
        recorder.stop_event.set()
        if recorder.thread: recorder.thread.join(timeout=8)
        camera.close()
        server.server_close()


if __name__ == '__main__': main()
