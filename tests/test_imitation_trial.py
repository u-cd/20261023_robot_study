import copy
import unittest
from control.imitation_trial import PolicyTrial


class Policy:
    def predict(self, detections, state, phase):
        return dict(action='base+',label=0,joint=1,direction='+',confidence=.9,
            predicted_amount_deg=3,within_joint_training_bounds=True)


class Detector:
    def observation(self):
        return {'detections':{'sponge':{'center_normalized':[.3,.4],'confidence':.9}},
                'sequence':10,'captured_at':'test'}


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.state={'connected':True,'joints':[dict(joint=i,angle=a,target=a,active=True,speed=25) for i,a in enumerate([90,110,40,80,100],1)]}
        self.sent=[]
        def send(cmd):
            self.sent.append(cmd)
            if cmd['action']=='step':
                item=self.state['joints'][cmd['joint']-1];item['angle']+=1;item['target']=item['angle']
            return copy.deepcopy(self.state)
        self.trial=PolicyTrial(Policy(),Detector(),send)

    def test_preview_read_only_then_one_degree_once(self):
        p=self.trial.preview(0)
        self.assertTrue(all(c['action']=='status' for c in self.sent))
        result=self.trial.execute(p['ticket'])
        self.assertTrue(result['moved'])
        self.assertEqual(self.state['joints'][0]['angle'],91)
        self.assertEqual(sum(c['action']=='step' for c in self.sent),1)
        with self.assertRaises(ValueError):self.trial.execute(p['ticket'])

    def test_low_confidence_maximum_is_still_executable(self):
        original=self.trial.policy.predict
        def low(*args):
            result=original(*args);result['confidence']=.27;return result
        self.trial.policy.predict=low
        p=self.trial.preview(0)
        self.assertTrue(p['can_execute'])
        self.assertTrue(self.trial.execute(p['ticket'])['moved'])

    def test_changed_state_rejects_without_motion(self):
        p=self.trial.preview(0)
        self.state['joints'][0]['angle']=self.state['joints'][0]['target']=89
        with self.assertRaises(ValueError):self.trial.execute(p['ticket'])
        self.assertFalse(any(c['action']=='step' for c in self.sent))

    def test_missing_or_stale_detection_blocks(self):
        self.trial.detector.observation=lambda:dict(detections={},sequence=1,captured_at='time')
        p=self.trial.preview(0)
        self.assertFalse(p['can_execute'])
        with self.assertRaises(ValueError):self.trial.execute(p['ticket'])
        self.assertFalse(any(c['action']=='step' for c in self.sent))

    def test_manual_order_advance_does_not_move_and_invalidates_old_ticket(self):
        from control.motion_order import MotionOrder
        self.trial.motion_order=MotionOrder()
        original=self.trial.policy.predict
        def full(*args):
            p=original(*args);p['probabilities']=[.1,.1,.1,.4,.1,.1,.03,.03,.04];return p
        self.trial.policy.predict=full
        prediction=self.trial.preview(1)
        self.assertEqual(prediction['order_stage'],'lift')
        result=self.trial.advance_order()
        self.assertEqual((result['phase'],result['order_stage']),(1,'base'))
        self.assertTrue(all(c['action']=='status' for c in self.sent))
        with self.assertRaises(ValueError):self.trial.execute(prediction['ticket'])

    def test_invalid_phase_does_not_send(self):
        with self.assertRaises(ValueError):self.trial.preview(3)
        self.assertFalse(self.sent)


if __name__=='__main__':unittest.main()
