"""Firmware-name compatibility without connecting to a board."""
import unittest
from unittest.mock import patch

from control.arm_panel import ArmConnection


class FakeSerial:
    def __init__(self,banner):
        self.pending=[banner+b'\n'];self.writes=[];self.closed=False

    def write(self,data):
        self.writes.append(data)
        if data==b'STOP\n':self.pending += [b'STOP: holding\n',b'DONE\n']
        elif data==b'STATUS\n':
            self.pending += [f'{i} OFF last-commanded=90 target=90 speed=25\n'.encode() for i in range(1,6)]+[b'DONE\n']
        else:raise AssertionError('Unexpected command: '+repr(data))

    def readline(self):return self.pending.pop(0) if self.pending else b''
    def close(self):self.closed=True


class ArmConnectionTests(unittest.TestCase):
    def test_new_and_existing_firmware_connect_without_starting_servos(self):
        for banner in (b'ARM_CONTROLLER_V3',b'ARM_TEST_V3'):
            with self.subTest(banner=banner):
                port=FakeSerial(banner);arm=ArmConnection()
                with patch('control.arm_panel.serial.Serial',return_value=port):
                    state=arm.action({'action':'connect','port':'/dev/ttyUSB0'})
                self.assertTrue(state['connected']);self.assertEqual(len(state['joints']),5)
                self.assertFalse(any(j['active'] for j in state['joints']))
                self.assertEqual(port.writes,[b'STOP\n',b'STATUS\n'])
                arm.close();self.assertTrue(port.closed)

    def test_unknown_firmware_is_rejected_without_motion_commands(self):
        port=FakeSerial(b'UNRELATED_DEVICE');arm=ArmConnection()
        with patch('control.arm_panel.serial.Serial',return_value=port),patch('control.arm_panel.time.monotonic',side_effect=[0,0,9]):
            with self.assertRaisesRegex(RuntimeError,'arm_controller'):
                arm.action({'action':'connect','port':'/dev/ttyUSB0'})
        self.assertEqual(port.writes,[]);self.assertTrue(port.closed)


if __name__=='__main__':unittest.main()
