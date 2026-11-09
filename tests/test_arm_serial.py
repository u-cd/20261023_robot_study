"""Serial framing tests: no USB port or real motion is used."""
import unittest
from pathlib import Path
import re
from ui.arm import ArmConnection, SERIAL_BAUDRATE
from ui.act_test import ACTTrial


class Port:
    def __init__(self,chunks):self.chunks=iter(chunks);self.writes=[]
    def readline(self):return next(self.chunks,b'')
    def write(self,data):self.writes.append(data)


def status_lines():
    return [f'{i} ON commanded=90 target=90 speed=25\r\n'.encode() for i in range(1,6)]


class SerialTests(unittest.TestCase):
    def connection(self,chunks):
        arm=ArmConnection();arm.port=Port(chunks);return arm

    def test_partial_status_lines_and_done_survive_read_timeouts(self):
        chunks=[]
        for line in status_lines()+[b'DONE\r\n']:
            chunks += [line[:3],b'',line[3:-1],b'',line[-1:]]
        arm=self.connection(chunks)
        self.assertEqual([s['joint'] for s in arm.status()],[1,2,3,4,5])
        self.assertEqual(arm.port.writes,[b'STATUS\n'])
        self.assertEqual(arm._rx,b'')

    def test_incomplete_status_has_raw_diagnostics_without_reflash_advice(self):
        arm=self.connection((status_lines()[:4]+[b'DONE\n'])*2)
        with self.assertRaises(RuntimeError) as caught:arm.status()
        self.assertIn('4 ON commanded',str(caught.exception))
        self.assertNotIn('書き込',str(caught.exception))
        self.assertEqual(len(arm.last_exchange['lines']),4)

    def test_duplicate_joint_cannot_masquerade_as_five_valid_states(self):
        arm=self.connection((status_lines()[:4]+status_lines()[:1]+[b'DONE\n'])*2)
        with self.assertRaises(RuntimeError):arm.status()

    def test_batch_move_counts_complete_fragmented_acknowledgements(self):
        arm=self.connection([part for _ in range(5) for part in
                             [b'MOVE accepted: target set\n',b'DO',b'',b'NE\n']])
        trial=ACTTrial.__new__(ACTTrial);trial.arm=arm
        trial.move_batch([70,113,43,80,90])
        self.assertEqual(len(arm.port.writes),1)
        self.assertTrue(arm.port.writes[0].endswith(b'MOVE 5 90\n'))

    def test_firmware_error_in_batch_is_never_accepted(self):
        arm=self.connection([b'ERROR: inactive\n',b'DONE\n']*5)
        trial=ACTTrial.__new__(ACTTrial);trial.arm=arm
        with self.assertRaises(RuntimeError) as caught:trial.move_batch([70,113,43,80,90])
        exchange=caught.exception.serial_exchange
        self.assertEqual(exchange['done_count'],5)
        self.assertEqual(len(exchange['raw_lines_hex']),10)
        arm.last_exchange={'command':'STOP'}
        self.assertEqual(len(exchange['command']),5)

    def test_corrupted_status_is_preserved_then_retried_once_without_motion(self):
        damaged=status_lines();damaged[0]=b'1 ON \xffcommanded=90 target=90 speed=25\n'
        arm=self.connection(damaged+[b'DONE\n']+status_lines()+[b'DONE\n'])
        self.assertEqual(len(arm.status()),5)
        self.assertEqual(arm.port.writes,[b'STATUS\n',b'STATUS\n'])
        self.assertEqual(arm.status_retries,1)
        self.assertIn('ff',arm.status_issues[0]['exchange']['raw_lines_hex'][0])
        self.assertIn('\ufffd',arm.status_issues[0]['exchange']['lines'][0])

    def test_repeated_corruption_stops_after_two_read_only_requests(self):
        arm=self.connection([b'broken\n',b'DONE\n']*2)
        with self.assertRaises(RuntimeError):arm.status()
        self.assertEqual(arm.port.writes,[b'STATUS\n',b'STATUS\n'])
        self.assertEqual(len(arm.status_issues),2)

    def test_pc_and_firmware_baudrates_match(self):
        firmware=(Path(__file__).resolve().parents[1]/'arm_controller.ino').read_text()
        self.assertEqual(int(re.search(r'Serial.begin\((\d+)\)',firmware)[1]),SERIAL_BAUDRATE)
