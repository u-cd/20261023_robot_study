import json
from pathlib import Path
import unittest

import numpy as np

from control.pick_place_plan import build_transfer


class PickPlacePlanTests(unittest.TestCase):
    model=Path(__file__).resolve().parents[1]/'runs/arm_kinematics/model.json'

    def setUp(self):
        if not self.model.exists():self.skipTest('11点モデルが未生成')

    def build(self,**kwargs):
        return build_transfer(self.model,[15,135],[30,90],[93,144,63,80,150],**kwargs)

    def test_complete_sequence_keeps_positive_grasp_height_and_fixed_wrist(self):
        plan=self.build()
        self.assertEqual(len(plan['stages']),10)
        self.assertFalse(plan['motion_executed'])
        self.assertFalse(plan['height_generalization_verified'])
        stages=plan['stages']
        self.assertEqual(stages[3]['target_xyz_mm'],[15,135,15])
        self.assertEqual(stages[3]['all_commands_deg'][-1],60)
        self.assertEqual(stages[4]['all_commands_deg'][-1],150)
        self.assertEqual(stages[5]['target_xyz_mm'][-1],40)
        self.assertEqual(stages[6]['target_xyz_mm'],[30,90,40])
        self.assertEqual(stages[8]['all_commands_deg'][-1],60)
        for stage in stages:
            self.assertEqual(stage['all_commands_deg'][3],80)
            previous=None
            for waypoint in stage['waypoints']:
                self.assertGreaterEqual(waypoint['xyz_mm'][2],0)
                if previous is not None:self.assertLessEqual(np.linalg.norm(np.array(previous)-waypoint['xyz_mm']),5.001)
                previous=waypoint['xyz_mm']

    def test_rejects_zero_grasp_height_wrong_wrist_and_outside_xy(self):
        with self.assertRaises(ValueError):self.build(pick_z=0)
        with self.assertRaises(ValueError):self.build(travel_z=10)
        with self.assertRaises(ValueError):self.build(open_angle=160)
        with self.assertRaises(ValueError):build_transfer(self.model,[100,100],[30,90],[93,144,63,80,150])
        with self.assertRaises(ValueError):build_transfer(self.model,[15,135],[30,90],[93,144,63,90,150])


if __name__=='__main__':unittest.main()
