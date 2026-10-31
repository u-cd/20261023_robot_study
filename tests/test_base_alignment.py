import copy
import json
from pathlib import Path
import tempfile
import unittest
from control.base_alignment import BaseAlignment


class BaseAlignmentTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        model={'object':'sponge','mean':[.3,.5],'scale':[.1,.1], 'weights':[50,0,0],
               'quadratic':False,'uv_min':[.2,.3],'uv_max':[.5,.7]}
        path=Path(tmp.name)/'guide.json';path.write_text(json.dumps({'tolerance_deg':4,'models':{'0':model}}))
        self.guide=BaseAlignment(path)
        self.p=dict(action='base+',label=0,joint=1,direction='+',confidence=.6,
                    probabilities=[.6,.2,.13,.02,.04,.005,.003,.001,.001])
        self.d={'sponge':{'center_normalized':[.3,.5]}}

    def apply(self, angle, phase=0):
        return self.guide.apply(copy.deepcopy(self.p),self.d,{'joints':[{'joint':1,'angle':angle}]},phase)

    def test_near_target_switches_from_base_to_highest_remaining(self):
        for _ in range(30):
            p=self.apply(50)
            self.assertEqual(p['action'],'shoulder+')
            self.assertEqual(p['raw_action'],'base+')
            self.assertEqual(p['masked_labels'],[0,1])

    def test_toward_target_allowed_and_away_target_masked(self):
        self.assertEqual(self.apply(40)['action'],'base+')
        p=self.apply(60)
        self.assertNotEqual(p['action'],'base+')
        self.assertEqual(p['masked_labels'],[0])

    def test_untrained_phase_and_outside_position_do_not_fabricate_goal(self):
        self.assertEqual(self.apply(50,2)['action'],'base+')
        self.d['sponge']['center_normalized']=[.9,.9]
        p=self.apply(50)
        self.assertEqual(p['action'],'base+')
        self.assertNotIn('base_goal_deg',p)

    def test_phase_change_does_not_reuse_old_goal_when_detection_missing(self):
        self.apply(50);self.apply(50,2);self.d={}
        p=self.apply(50,0)
        self.assertEqual(p['action'],'base+')
        self.assertNotIn('base_goal_deg',p)


if __name__=='__main__':unittest.main()
