import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from control.imitation_record import DemonstrationRecorder, PHASES
from vision.prepare_imitation import features


class Camera:
    source = 'test'
    def snapshot(self, fresh=False):
        return np.zeros((20, 30, 3), dtype=np.uint8), '2026-10-31T00:00:00+00:00', 1


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = {'connected': True, 'joints': [dict(joint=n, angle=a, target=a, active=True, speed=25)
                      for n, a in enumerate([90, 100, 80, 80, 90], 1)]}
        self.sent = []
        self.rec = DemonstrationRecorder(Path(self.tmp.name), Camera(), self.send)

    def send(self, cmd):
        self.sent.append(cmd)
        if cmd['action'] == 'step':
            j = self.state['joints'][cmd['joint']-1]
            j['angle'] += 1 if cmd['direction'] == '+' else -1
            j['target'] = j['angle']
        return copy.deepcopy(self.state)

    def test_pre_action_alignment_and_finish(self):
        self.rec.start('test')
        self.assertTrue(all(c['action'] == 'status' for c in self.sent))
        self.rec.set_phase(1)
        self.rec.action(2, '+')
        records = [json.loads(s) for s in (self.rec.folder/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in records], ['intent', 'result'])
        row = records[1]
        self.assertEqual(row['state_before']['joints'][1]['angle'], 100)
        self.assertEqual(row['state_after']['joints'][1]['angle'], 101)
        self.assertEqual(row['phase'], 1)
        self.assertEqual(row['action'], 'shoulder+')
        self.assertTrue((self.rec.folder/row['image']).exists())
        self.rec.finish('success')
        self.assertFalse(self.rec.active)
        self.assertIn('shoulder+', (self.rec.folder/'viewer.html').read_text())

    def test_failure_and_abort_delete_only_current_episode(self):
        self.rec.start()
        self.rec.action(1, '+')
        self.rec.finish('success')
        success = self.rec.folder
        for outcome in ('failure', 'aborted'):
            self.rec.start()
            self.rec.action(1, '-')
            discarded = self.rec.folder
            info = self.rec.finish(outcome)
            self.assertFalse(discarded.exists())
            self.assertTrue((success/'episode.json').exists())
            self.assertFalse(info['active'])
            self.assertIsNone(info['path'])
            self.assertEqual(info['outcome'], outcome)

    def test_three_degree_action_records_amount_and_respects_limit(self):
        self.rec.start()
        self.rec.action(1, '+', degrees=3)
        row = json.loads((self.rec.folder/'samples.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['delta_deg'], 3)
        self.assertEqual(row['state_after']['joints'][0]['angle'], 93)
        self.state['joints'][4]['angle'] = self.state['joints'][4]['target'] = 149
        count = len(self.sent)
        with self.assertRaises(ValueError):self.rec.action(5, '+', degrees=3)
        self.assertFalse(any(c['action']=='step' for c in self.sent[count:]))

    def test_jog_keeps_moving_while_disk_writer_is_busy(self):
        self.rec.start()
        blocked, release = threading.Event(), threading.Event()
        original = self.rec.write_image
        def slow(name, frame):
            blocked.set()
            release.wait(3)
            original(name, frame)
        self.rec.write_image = slow
        try:
            self.rec.jog(1, '+')
            self.state['joints'][0]['angle'] = self.state['joints'][0]['target'] = 92
            self.rec.jog(1, '+')
            self.assertTrue(blocked.wait(1))
            # Still accepts another heartbeat even with the previous disk write blocked.
            self.state['joints'][0]['angle'] = self.state['joints'][0]['target'] = 94
            self.rec.jog(1, '+')
            self.assertFalse(any(c['action']=='stop' for c in self.sent))
            self.rec.stop_jog()
            self.assertEqual(self.sent[-1]['action'], 'stop')
        finally:
            release.set()
            self.rec.finish('success')
        rows = [json.loads(s) for s in (self.rec.folder/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['delta_deg'] for r in rows], [2,2,0])
        self.assertTrue(all(r['accepted'] and r['control_mode']=='jog' for r in rows))
        self.assertTrue(all((self.rec.folder/r['image']).exists() for r in rows))

    def test_jog_failure_stops_and_is_not_accepted(self):
        self.rec.start()
        original = self.rec.send
        def fail(cmd):
            if cmd['action']=='jog':raise OSError('serial failed')
            return original(cmd)
        self.rec.send = fail
        with self.assertRaises(OSError):self.rec.jog(1, '+')
        self.assertEqual(self.sent[-1]['action'], 'stop')
        self.rec._flush()
        row = json.loads((self.rec.folder/'samples.jsonl').read_text().splitlines()[-1])
        self.assertFalse(row['accepted'])
        self.rec.finish('aborted')

    def test_camera_failure_prevents_command(self):
        self.rec.start()
        def broken(**kw):raise RuntimeError('camera gone')
        self.rec.camera.snapshot = broken
        with self.assertRaises(RuntimeError):self.rec.action(1, '+')
        self.assertFalse(any(c['action'] == 'step' for c in self.sent))

    def test_bad_wrist_and_limit(self):
        self.state['joints'][3]['angle'] = self.state['joints'][3]['target'] = 90
        with self.assertRaises(ValueError):self.rec.start()
        self.state['joints'][3]['angle'] = self.state['joints'][3]['target'] = 80
        self.rec.start()
        self.state['joints'][4]['angle'] = self.state['joints'][4]['target'] = 150
        with self.assertRaises(ValueError):self.rec.action(5, '+')
        with self.assertRaises(ValueError):self.rec.action(4, '+')
        self.assertFalse(any(c['action'] == 'step' for c in self.sent))

    def test_wire_failure_is_not_successful_label(self):
        self.rec.start()
        original = self.rec.send
        def fail(cmd):
            if cmd['action'] == 'step':raise OSError('serial lost')
            return original(cmd)
        self.rec.send = fail
        with self.assertRaises(OSError):self.rec.action(1, '+')
        row = json.loads((self.rec.folder/'samples.jsonl').read_text().splitlines()[-1])
        self.assertFalse(row['accepted'])
        self.assertEqual(row['error'], 'serial lost')

    def test_wait_no_step_and_feature_order(self):
        self.rec.start()
        self.rec.action(wait=True)
        self.assertFalse(any(c['action'] == 'step' for c in self.sent))
        vector = features({'sponge': {'center_normalized': [.2, .3], 'confidence': .8}}, self.state, 2)
        self.assertEqual(len(vector), 12 + len(PHASES))
        self.assertEqual(vector[:8], [.2, .3, 0, 0, .8, 0, 1, 0])
        self.assertEqual(sum(vector[12:]), 1)
        self.assertEqual(vector[14], 1)


if __name__ == '__main__':unittest.main()
