import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from vision.table_geometry import (reference_points, fit_plane, project, check_errors,
                                   load_calibration, plane_estimate)


class TableGeometryTests(unittest.TestCase):
    def setUp(self):
        self.corners_mm, self.checks_mm = reference_points(200, 150)
        # Known synthetic pinhole projection with perspective, not merely scaling.
        self.to_camera = np.array([[2, .3, 100], [.1, 1.8, 50], [.001, .0004, 1]])
        self.corners_px = project(self.to_camera, self.corners_mm)
        self.checks_px = project(self.to_camera, self.checks_mm)
        self.matrix = fit_plane(self.corners_px, 200, 150)

    def test_recovers_independent_points_under_perspective(self):
        points = np.array([[25,30],[175,120],[110,65]])
        np.testing.assert_allclose(project(self.matrix, project(self.to_camera, points)), points, atol=1e-4)
        self.assertLess(check_errors(self.matrix, self.checks_px, 200,150)['max_error_mm'], 1e-4)

    def test_wrong_check_click_is_detected(self):
        self.checks_px[0] += [40,30]
        self.assertGreater(check_errors(self.matrix,self.checks_px,200,150)['max_error_mm'],5)

    def test_crossed_duplicate_collinear_and_nonfinite_corners_rejected(self):
        for corners in [self.corners_px[[0,2,1,3]], [[1,1]]*4,
                        [[0,0],[20,0],[40,0],[60,0]],
                        [[0,0],[200,0],[200,150],[np.nan,150]]]:
            with self.assertRaises(ValueError):
                fit_plane(corners,200,150)

    def test_outside_points_are_not_extrapolated(self):
        record = {'matrix': self.matrix, 'corners_px': self.corners_px}
        self.assertIsNone(plane_estimate(record, project(self.to_camera, [[-20,20]])[0]))
        np.testing.assert_allclose(plane_estimate(record,self.checks_px[0]),[100,75],atol=1e-4)

    def test_camera_resolution_and_validation_are_checked_on_load(self):
        record = {'schema_version':1,'coordinate_frame':'calibration_rectangle',
                  'camera':'test-camera','image_size':[640,480], 'rectangle_mm':[200,150],
                  'corners_px':self.corners_px.tolist(),'check_points_px':self.checks_px.tolist(),
                  'max_allowed_error_mm':5}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json'
            path.write_text(json.dumps(record))
            loaded=load_calibration(path,'test-camera',(640,480))
            np.testing.assert_allclose(project(loaded['matrix'],self.checks_px),self.checks_mm,atol=1e-4)
            for camera,size in [('other-camera',(640,480)),('test-camera',(1280,720))]:
                with self.assertRaises(ValueError):
                    load_calibration(path,camera,size)
            record['check_points_px'][0][0]+=50
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_calibration(path,'test-camera',(640,480))


if __name__ == '__main__':
    unittest.main()
