import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from vision.arm_camera_panel import LiveCamera


class CameraReconnectTests(unittest.TestCase):
    def make_camera(self):
        camera=LiveCamera.__new__(LiveCamera)
        camera.source='/test/camera'
        camera.expected_size=[640,480]
        camera.condition=threading.Condition()
        camera.frame=None
        camera.sequence=0
        camera.error='previous failure'
        camera.closed=False
        camera.captured_at=None
        camera.captured_monotonic=0
        camera.stop_event=Mock()
        return camera

    def test_read_failure_reopens_device_and_clears_stale_error(self):
        camera=self.make_camera()
        first,second=Mock(),Mock()
        first.isOpened.return_value=second.isOpened.return_value=True
        first.read.return_value=(False,None)
        frame=np.zeros((480,640,3),dtype=np.uint8)
        observed=[]
        def second_read():
            if camera.sequence==0:return True,frame
            observed.append((camera.error,camera.frame.copy()))
            camera.closed=True
            return False,None
        second.read.side_effect=second_read
        with patch('vision.arm_camera_panel.Path.exists',return_value=True),patch('vision.arm_camera_panel.cv2.VideoCapture',side_effect=[first,second]) as opened:
            camera.run()
        self.assertEqual(opened.call_count,2)
        first.release.assert_called_once()
        second.release.assert_called_once()
        self.assertIsNone(observed[0][0])
        np.testing.assert_array_equal(observed[0][1],frame)
        self.assertIsNone(camera.frame)  # No stale frame is retained after another failure.

    def test_wrong_resolution_is_not_published(self):
        camera=self.make_camera()
        device=Mock()
        device.isOpened.return_value=True
        device.read.return_value=(True,np.zeros((720,1280,3),dtype=np.uint8))
        camera.stop_event.wait.side_effect=lambda seconds:setattr(camera,'closed',True)
        with patch('vision.arm_camera_panel.Path.exists',return_value=True),patch('vision.arm_camera_panel.cv2.VideoCapture',return_value=device):
            camera.run()
        self.assertIsNone(camera.frame)
        self.assertEqual(camera.sequence,0)
        self.assertIn('解像度',camera.error)


if __name__=='__main__':unittest.main()
