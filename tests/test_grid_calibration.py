import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from vision.table_geometry import grid_reference, fit_grid, project, load_calibration, plane_estimate
from vision.calibrate_grid import save_record


class GridCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.world,self.checks=grid_reference()
        self.forward=np.array([[1.8,.15,100],[.1,-1.5,400],[.0003,.0002,1]])
        self.pixels=project(self.forward,self.world)

    def test_grid_dimensions_order_and_split(self):
        self.assertEqual(self.world.shape,(64,2))
        np.testing.assert_array_equal(self.world[[0,7,8,56,63]],[[0,0],[210,0],[0,30],[0,210],[210,210]])
        self.assertEqual(int(self.checks.sum()),9)
        self.assertFalse(self.checks[[0,7,56,63]].any())

    def test_42_point_grid_roundtrip(self):
        world,checks=grid_reference(7,6,30)
        self.assertEqual(world.shape,(42,2))
        self.assertEqual(int(checks.sum()),6)
        np.testing.assert_array_equal(world[[0,6,7,35,41]],[[0,0],[180,0],[0,30],[0,150],[180,150]])
        pixels=project(self.forward,world)
        matrix,stats=fit_grid(pixels,7,6,30)
        self.assertEqual(stats['fit']['count'],36)
        self.assertEqual(stats['check']['count'],6)
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            frame=np.full((480,640,3),240,dtype=np.uint8)
            record=save_record(folder,frame,pixels.tolist(),'test42','passed',matrix,stats,7,6,30)
            loaded=load_calibration(folder/'record.json','test42',(640,480))
            self.assertEqual(record['rectangle_mm'],[180,150])
            self.assertEqual(len(record['points']),42)
            self.assertEqual(record['points'][-1]['number'],42)
            np.testing.assert_allclose(project(loaded['matrix'],pixels),world,atol=.001)

    def test_held_out_points_do_not_influence_fit(self):
        matrix,stats=fit_grid(self.pixels)
        bad=self.pixels.copy()
        bad[self.checks]+=np.array([20,0])
        unchanged,bad_stats=fit_grid(bad)
        np.testing.assert_allclose(matrix,unchanged,atol=1e-10)
        self.assertLess(stats['check']['max_error_mm'],.001)
        self.assertGreater(bad_stats['check']['max_error_mm'],5)

    def test_noisy_fit_and_unseen_position(self):
        pixels=self.pixels+np.random.default_rng(42).normal(0,.3,self.pixels.shape)
        matrix,stats=fit_grid(pixels)
        position=np.array([[72,119]])
        np.testing.assert_allclose(project(matrix,project(self.forward,position)),position,atol=.4)
        self.assertLess(stats['check']['max_error_mm'],2)
        pixels[20]+=[100,-100]
        _,bad_stats=fit_grid(pixels)
        self.assertGreater(bad_stats['fit']['max_error_mm'],5)

    def test_evidence_roundtrip_partial_and_failed_rejection(self):
        matrix,stats=fit_grid(self.pixels)
        frame=np.full((480,640,3),240,dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            record=save_record(folder,frame,self.pixels.tolist(),'synthetic','passed',matrix,stats)
            loaded=load_calibration(folder/'record.json','synthetic',(640,480))
            np.testing.assert_allclose(plane_estimate(loaded,self.pixels[27]),self.world[27],atol=.001)
            np.testing.assert_array_equal(cv2.imread(record['reference_image']),frame)
            self.assertEqual(cv2.imread(record['annotated_image']).shape,(1060,1280,3))
            with (folder/'points.csv').open() as f:
                self.assertEqual(len(list(csv.reader(f))),65)
            self.assertIn('points.png',(folder/'report.html').read_text())
            self.assertEqual(len(json.loads((folder/'record.json').read_text())['points']),64)
            save_record(folder/'partial',frame,self.pixels[:10].tolist(),'synthetic','partial')
            self.assertEqual(len(json.loads((folder/'partial/record.json').read_text())['points']),10)
            with self.assertRaises(ValueError):
                load_calibration(folder/'partial/record.json','synthetic',(640,480))
            record['grid_points_px'][9][0]+=50
            (folder/'record.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_calibration(folder/'record.json','synthetic',(640,480))


if __name__=='__main__':
    unittest.main()
