import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from control.arm_calibration import CalibrationRecorder, apply_hover_condition


class FakeArm:
    def __init__(self):
        self.lock=threading.Lock()
        self.last_motion_at=time.monotonic()-10
        self.joints=[{'joint':i,'active':True,'angle':90,'target':90,'speed':25} for i in range(1,6)]

    def status(self):
        return [dict(j) for j in self.joints]


class ArmCalibrationRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        source=self.root/'calibration.json'
        source.write_text(json.dumps({'schema_version':2,'status':'passed','points':[
            {'number':1,'expected_mm':[0,0]},{'number':9,'expected_mm':[30,30]}]}))
        self.arm=FakeArm()
        self.recorder=CalibrationRecorder(self.arm,source,self.root/'records')
        self.data={'point':9,'confirmed':True,'approach':'same_direction','note':'test'}

    @staticmethod
    def capture(command,**kwargs):
        folder=Path(command[command.index('--output')+1])
        (folder/'capture.json').write_text(json.dumps({'captured_at':'test','camera':'test'}))
        (folder/'raw.png').write_bytes(b'test-image')
        (folder/'marked.png').write_bytes(b'test-image')

    def test_rejects_unconfirmed_or_unavailable_reference(self):
        with patch('control.arm_calibration.subprocess.run') as run:
            for data in [dict(self.data,confirmed=False),dict(self.data,point=99),dict(self.data,point=True)]:
                with self.assertRaises(ValueError):
                    self.recorder.record(data)
            run.assert_not_called()

    def test_requires_active_settled_and_quiet_commands(self):
        with patch('control.arm_calibration.subprocess.run') as run:
            self.arm.joints[0]['active']=False
            with self.assertRaises(ValueError):self.recorder.record(self.data)
            self.arm.joints[0]['active']=True
            self.arm.joints[0]['target']=100
            with self.assertRaises(ValueError):self.recorder.record(self.data)
            self.arm.joints[0]['target']=90
            self.arm.last_motion_at=time.monotonic()
            with self.assertRaises(ValueError):self.recorder.record(self.data)
            run.assert_not_called()

    def test_saves_reference_commands_and_repeats_without_sending_motion(self):
        with patch('control.arm_calibration.subprocess.run',side_effect=self.capture):
            first=self.recorder.record(self.data)['calibration_sample']
            second=self.recorder.record(self.data)['calibration_sample']
            self.assertEqual(first['repeat'],1)
            self.assertEqual(second['repeat'],2)
            saved=json.loads((Path(first['path'])/'sample.json').read_text())
            self.assertEqual(saved['reference_xyz_mm'],[30,30,None])
            self.assertEqual(saved['height_condition'],'hover_unmeasured')
            self.assertFalse(saved['paper_contact'])
            self.assertFalse(saved['joint_angles_measured'])
            self.assertEqual(saved['joint_command_degrees'],[90]*5)
            self.assertEqual(saved['tip'],'center of closed gripper tip')
            self.arm.joints[4]['angle']=self.arm.joints[4]['target']=100
            with self.assertRaises(ValueError):self.recorder.record(self.data)

    def test_old_contact_assumption_is_corrected_with_original_preserved(self):
        folder=self.recorder.session/'old_sample'
        folder.mkdir()
        path=folder/'sample.json'
        original={'reference_xyz_mm':[0,30,0],'contact_and_stillness_confirmed_by_user':True}
        path.write_text(json.dumps(original))
        apply_hover_condition(path)
        changed=json.loads(path.read_text())
        self.assertEqual(changed['reference_xyz_mm'],[0,30,None])
        self.assertFalse(changed['paper_contact'])
        self.assertEqual(json.loads((folder/'sample_before_height_correction.json').read_text()),original)
        apply_hover_condition(path)
        self.assertEqual(json.loads((folder/'sample_before_height_correction.json').read_text()),original)

    def test_changed_commands_during_capture_are_not_accepted(self):
        def changed(command,**kwargs):
            self.capture(command,**kwargs)
            self.arm.joints[0]['angle']=self.arm.joints[0]['target']=91
        with patch('control.arm_calibration.subprocess.run',side_effect=changed):
            with self.assertRaises(RuntimeError):self.recorder.record(self.data)
        self.assertEqual(self.recorder.samples,{})
        self.assertEqual(len(list(self.recorder.session.glob('*/rejected.json'))),1)
        self.assertEqual(list(self.recorder.session.glob('*/sample.json')),[])

    def test_camera_failure_is_not_a_successful_measurement(self):
        with patch('control.arm_calibration.subprocess.run',side_effect=subprocess.CalledProcessError(1,['test'],stderr='camera busy')):
            with self.assertRaisesRegex(RuntimeError,'camera busy'):self.recorder.record(self.data)
        self.assertEqual(self.recorder.samples,{})


if __name__=='__main__':unittest.main()
