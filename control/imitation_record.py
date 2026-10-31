"""Action-aligned demonstrations. Commands are not measured joint positions."""
from datetime import datetime, timezone
import html
import json
import shutil
import queue
import threading
from pathlib import Path
import time

import cv2

from control.fine_step import fine_step
from vision.arm_position_test import stationary_commands

PHASES = ['つかむまで', 'はなすまで', '退避']
ACTIONS = ['base+', 'base-', 'shoulder+', 'shoulder-', 'elbow+', 'elbow-', 'gripper+', 'gripper-', 'wait']


def now():
    return datetime.now(timezone.utc).isoformat()


class DemonstrationRecorder:
    # Caller serializes operations with the camera panel's fine_lock.
    def __init__(self, root, camera, send):
        self.root = Path(root)
        self.camera = camera
        self.send = send
        self.folder = None
        self.meta = None
        self.count = 0
        self.last_command_time = None
        self.pending_jog = None
        self.writer_error = None
        self.jobs = queue.Queue(maxsize=32)
        self.writer = None

    def _writer_loop(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:return
                record, frame = job
                self.write_image(record['image'], frame)
                self.append(dict(record, event='result'))
            except Exception as error:
                self.writer_error = str(error)
            finally:
                self.jobs.task_done()

    def _queue_sample(self, record, frame):
        if self.writer is None:
            self.writer = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer.start()
        try:self.jobs.put_nowait((record, frame))
        except queue.Full:
            self.writer_error = '保存キューが満杯です。'
            raise RuntimeError('保存が追いつかないため停止します。')

    def _flush(self):
        self.jobs.join()
        if self.writer_error:
            raise RuntimeError('記録の保存に失敗しました: '+self.writer_error)

    def _end_interval(self, state):
        if self.pending_jog is None:return
        record, frame = self.pending_jog
        self.pending_jog = None
        before = stationary_commands(record['state_before'])
        after = stationary_commands(state)
        joint = record['joint']
        record.update(state_after=state, completed_at=now(), interval_end_at=now(),
                      delta_deg=after[joint-1]-before[joint-1], accepted='acknowledged_at' in record)
        if any(after[i]!=before[i] for i in range(5) if i!=joint-1):
            record.update(accepted=False, error='別の関節の変化を検出しました。')
        record['interval_seconds'] = time.monotonic()-record.pop('_started_monotonic')
        self._queue_sample(record, frame)

    def jog(self, joint, direction):
        if not self.active or type(joint) is not int or joint not in (1,2,3,5) or direction not in ('+','-'):
            raise ValueError('記録状態と操作方向を確認してください。')
        try:
            if self.writer_error or self.jobs.full():
                raise RuntimeError('保存が追いつかないか、保存に失敗しています。')
            # The MCU continues JOG during sampling; no per-sample STOP or disk I/O.
            before = self.send({'action':'status'})
            commands = stationary_commands(before)
            self._end_interval(before)
            if commands[3]!=80:raise ValueError('手首を80°にしてください。')
            low, high = (50,150) if joint==5 else (0,180)
            if (direction=='+' and commands[joint-1]>=high) or (direction=='-' and commands[joint-1]<=low):
                raise ValueError('その方向は指令範囲の端です。')
            frame, stamp, seq = self.camera.snapshot(fresh=False)
            tick = time.monotonic()
            record = {'index':self.count, 'phase':self.meta['phase'],
                'image':f'frames/{self.count:06d}.jpg', 'frame_captured_at':stamp,
                'frame_sequence':seq, 'state_before':before, 'state_observed_at':now(),
                'command_sent_at':now(), '_started_monotonic':tick,
                'dt_since_previous_action_s':None if self.last_command_time is None else tick-self.last_command_time,
                'action':ACTIONS[(1,2,3,5).index(joint)*2+(direction=='-')],
                'joint':joint, 'direction':direction, 'control_mode':'jog',
                'delta_deg':None, 'accepted':False}
            self.last_command_time = tick
            self.pending_jog = (record, frame)
            self.count += 1
            self.meta['sample_count'] = self.count
            state = self.send({'action':'jog','joint':joint,'direction':direction})
            record['acknowledged_at'] = now()
            return dict(self.info(), state=state)
        except Exception:
            try:self.stop_jog()
            except Exception:pass
            raise

    def stop_jog(self):
        # Also send STOP when an interval was lost to an I/O error.
        state = self.send({'action':'stop'})
        self._end_interval(state)
        return dict(self.info(), state=state)

    @property
    def active(self):
        return self.meta is not None and self.meta['outcome'] == 'recording'

    def info(self):
        return {'active': self.active, 'path': str(self.folder) if self.folder else None,
                'count': self.count, 'phase': self.meta['phase'] if self.meta else 0,
                'outcome': self.meta['outcome'] if self.meta else None}

    def save_meta(self):
        temp = self.folder / 'episode.json.tmp'
        temp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2) + '\n')
        temp.replace(self.folder / 'episode.json')

    def append(self, record):
        with (self.folder / 'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()

    def start(self, note=''):
        if self.active:
            raise ValueError('すでに記録中です。')
        # Validate camera and enabled/settled state before creating an episode. No motion.
        state = self.send({'action': 'status'})
        commands = stationary_commands(state)
        if commands[3] != 80:
            raise ValueError('記録前に手首を80°に揃えてください。')
        frame, stamp, seq = self.camera.snapshot(fresh=True)
        if stationary_commands(self.send({'action': 'status'})) != commands:
            raise ValueError('動きを止めてから記録を開始してください。')
        self.folder = self.root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.folder.mkdir(parents=True)
        (self.folder / 'frames').mkdir()
        self.write_image('setup.jpg', frame)
        self.meta = {'schema_version': 1, 'started_at': now(), 'outcome': 'recording',
                     'phase': 0, 'phases': PHASES, 'action_classes': ACTIONS,
                     'note': str(note)[:1000], 'initial_state': state,
                     'camera': self.camera.source, 'image_size': [frame.shape[1], frame.shape[0]],
                     'setup_frame_captured_at': stamp, 'setup_frame_sequence': seq,
                     'state_kind': 'commanded_angles_not_measured', 'wrist_fixed_deg': 80,
                     'sampling': 'pre_action_frames; jog_intervals_async_saved; idle_not_labelled',
                     'detections': 'extract_offline_from_saved_pre_action_frames',
                     'sample_count': 0}
        self.count = 0
        self.writer_error = None
        self.last_command_time = None
        self.save_meta()
        self.viewer()
        return self.info()

    def write_image(self, name, frame):
        ok, image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError('記録画像を保存できません。')
        (self.folder / name).write_bytes(image.tobytes())

    def set_phase(self, phase):
        if not self.active or type(phase) is not int or not 0 <= phase < len(PHASES):
            raise ValueError('記録状態・段階を確認してください。')
        if self.pending_jog is not None:self.stop_jog()
        self._flush()
        self.meta['phase'] = phase
        self.save_meta()
        return self.info()

    def action(self, joint=None, direction=None, wait=False, degrees=1):
        if not self.active:
            raise ValueError('先にデモ記録を開始してください。')
        if not wait and (type(joint) is not int or joint not in (1, 2, 3, 5) or direction not in ('+', '-')):
            raise ValueError('台座・肩・肘・グリッパーの方向を指定してください。')
        if self.pending_jog is not None:self.stop_jog()
        self._flush()
        if type(degrees) is not int or degrees not in (1, 3):
            raise ValueError('操作量は1°または3°です。')
        before = self.send({'action': 'status'})
        commands = stationary_commands(before)
        if commands[3] != 80:
            raise ValueError('手首が80°から変わっています。')
        if not wait:
            target = commands[joint-1] + (degrees if direction == '+' else -degrees)
            low, high = (50, 150) if joint == 5 else (0, 180)
            if not low <= target <= high:
                raise ValueError('その方向は指令範囲の端です。')
        frame, stamp, seq = self.camera.snapshot(fresh=True)
        if stationary_commands(self.send({'action': 'status'})) != commands:
            raise ValueError('撮影中に別の操作が入りました。記録用の画面だけを使ってください。')
        image_name = f'frames/{self.count:06d}.jpg'
        self.write_image(image_name, frame)  # Disk failure must prevent sending a command.
        record = {'index': self.count, 'phase': self.meta['phase'], 'image': image_name,
                  'frame_captured_at': stamp, 'frame_sequence': seq, 'state_before': before,
                  'state_observed_at': now(), 'requested_at': now(), 'accepted': False,
                  'action': 'wait' if wait else ACTIONS[(1, 2, 3, 5).index(joint)*2 + (direction == '-')],
                  'joint': joint, 'direction': direction, 'delta_deg': 0 if wait else (degrees if direction == '+' else -degrees),
                  'wait_seconds': .2 if wait else 0, 'wire_commands': []}
        self.append(dict(record, event='intent'))  # Preserve intent even after a crash during transmission.
        def traced_send(command):
            entry = {'command': command, 'sent_at': now()}
            record['wire_commands'].append(entry)
            if command['action'] == 'step':
                tick = time.monotonic()
                record.setdefault('command_sent_at', entry['sent_at'])
                if 'dt_since_previous_action_s' not in record:
                    record['dt_since_previous_action_s'] = None if self.last_command_time is None else tick-self.last_command_time
                    self.last_command_time = tick
            result = self.send(command)
            entry['responded_at'] = now()
            return result
        try:
            if wait:
                record['command_sent_at'] = now()
                tick = time.monotonic()
                record['dt_since_previous_action_s'] = None if self.last_command_time is None else tick-self.last_command_time
                self.last_command_time = tick
                time.sleep(.2)
                after = self.send({'action': 'status'})
                if stationary_commands(after) != commands:
                    raise ValueError('待機中に指令が変わりました。')
            else:
                for _ in range(degrees):
                    after = fine_step(traced_send, {'joint': joint, 'direction': direction})
            record['state_after'] = after
            record['accepted'] = True
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            record['completed_at'] = now()
            self.append(dict(record, event='result'))
            self.count += 1
            self.meta['sample_count'] = self.count
            self.save_meta()
        return dict(self.info(), state=after)

    def finish(self, outcome, note=''):
        if not self.active or outcome not in ('success', 'failure', 'aborted'):
            raise ValueError('記録状態・終了結果を確認してください。')
        if self.pending_jog is not None:self.stop_jog()
        self.jobs.join()
        if outcome == 'success':self._flush()
        if self.writer is not None:
            self.jobs.put(None)
            self.writer.join()
            self.writer = None
        if outcome != 'success':
            # Remove only this recorder's episode; never the root or another location.
            folder = self.folder.resolve()
            if folder.parent != self.root.resolve() or self.folder.is_symlink():
                raise ValueError('破棄対象の保存先が不正です。')
            shutil.rmtree(folder)
            self.folder = None
            self.meta.update(outcome=outcome, ended_at=now())
        else:
            self.meta.update(outcome=outcome, ended_at=now(), result_note=str(note)[:1000])
            self.save_meta()
            self.viewer()
        return self.info()

    def viewer(self):
        rows = []
        path = self.folder / 'samples.jsonl'
        for line in path.read_text().splitlines() if path.exists() else []:
            item = json.loads(line)
            if item['event'] != 'result':
                continue
            label = f"#{item['index']} {PHASES[item['phase']]} / {item['action']} / accepted={item['accepted']}"
            rows.append(f'<figure><img loading="lazy" width="320" src="{html.escape(item["image"])}"><figcaption>{html.escape(label)}</figcaption></figure>')
        (self.folder / 'viewer.html').write_text('<!doctype html><meta charset="utf-8"><title>模倣学習デモ</title>'
            '<h1>操作直前の画像と教師操作</h1><p>関節値は指令値です。学習用検出特徴は別途抽出します。</p>'
            + '<img width="320" src="setup.jpg" alt="開始時">' + ''.join(rows))
