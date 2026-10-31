import unittest
from control.motion_order import MotionOrder


class OrderTests(unittest.TestCase):
    def setUp(self):self.order=MotionOrder()

    def prediction(self,error=10):
        return dict(action='gripper-',label=7,joint=5,direction='-',confidence=.8,
            probabilities=[.04,.03,.04,.02,.03,.02,.02,.8,0],
            base_goal_deg=50,base_error_deg=error,selection_note='')

    def test_pick_base_then_arm_latches_and_blocks_base(self):
        p=self.order.apply(self.prediction(10),0)
        self.assertEqual((p['order_stage'],p['joint'],p['direction']),('base',1,'+'))
        p=self.order.apply(self.prediction(2),0)
        self.assertEqual(p['order_stage'],'arm')
        p=self.order.apply(self.prediction(15),0)
        self.assertEqual(p['order_stage'],'arm')
        self.assertNotEqual(p['joint'],1)

    def test_place_lift_blocks_base_and_gripper_until_confirmed(self):
        p=self.order.apply(self.prediction(10),1)
        self.assertEqual(p['order_stage'],'lift')
        self.assertIn(p['joint'],(2,3))
        self.assertTrue(p['lift_confirmation_required'])
        self.assertEqual(p['direction'],'-')
        self.assertEqual(p['order_allowed_labels'],[3,5,8])
        # Repeated preview cannot promote itself based on elapsed steps.
        for _ in range(20):self.assertEqual(self.order.apply(self.prediction(1),1)['order_stage'],'lift')
        self.order.confirm_lift()
        p=self.order.apply(self.prediction(10),1)
        self.assertEqual((p['order_stage'],p['joint']),('base',1))
        p=self.order.apply(self.prediction(2),1)
        self.assertEqual(p['order_stage'],'arm')
        self.assertIn(p['joint'],(2,3))

    def test_full_eight_step_sequence_and_gripper_isolation(self):
        p=self.order.apply(self.prediction(10),0)
        self.assertEqual(p['order_step'],1)
        self.assertEqual(p['order_allowed_labels'],[0])
        p=self.order.apply(self.prediction(0),0)
        self.assertEqual(p['order_step'],2)
        self.assertEqual(p['order_allowed_labels'],[2,3,4,5,8])
        self.order.advance()
        with self.assertRaises(ValueError):self.order.advance()  # No double-click skipping grasp.
        p=self.order.apply(self.prediction(0),0)
        self.assertEqual((p['order_step'],p['action']),(3,'gripper+'))
        self.assertEqual(p['order_allowed_labels'],[6])
        state=self.order.advance()
        self.assertEqual((state['phase'],state['order_stage']),(1,'lift'))
        p=self.order.apply(self.prediction(10),1)
        self.assertEqual((p['order_step'],p['order_allowed_labels']),(4,[3,5,8]))
        self.order.advance()
        p=self.order.apply(self.prediction(10),1)
        self.assertEqual((p['order_step'],p['joint']),(5,1))
        p=self.order.apply(self.prediction(0),1)
        self.assertEqual(p['order_step'],6)
        self.assertNotIn(5,[None if i==8 else (1,2,3,5)[i//2] for i in p['order_allowed_labels']])
        self.order.advance()
        p=self.order.apply(self.prediction(0),1)
        self.assertEqual((p['order_step'],p['action']),(7,'gripper-'))
        self.order.advance()
        p=self.order.apply(self.prediction(0),2)
        self.assertEqual(p['order_step'],8)
        self.assertEqual(p['order_allowed_labels'],[2,3,4,5,8])
        self.order.advance()
        p=self.order.apply(self.prediction(0),2)
        self.assertEqual(p['order_stage'],'done')
        self.assertEqual(p['order_allowed_labels'],[8])
        self.assertTrue(p['order_blocked_reason'])

    def test_no_goal_blocks_base_stage_instead_of_unbounded_rotation(self):
        p=self.prediction();del p['base_goal_deg'];del p['base_error_deg']
        result=self.order.apply(p,0)
        self.assertIsNone(result['joint'])
        self.assertTrue(result['order_blocked_reason'])

    def test_reset_and_phase_change_restart_order(self):
        self.order.apply(self.prediction(0),0)
        self.order.apply(self.prediction(),1)
        self.order.confirm_lift()
        self.order.reset()
        self.assertEqual(self.order.apply(self.prediction(),1)['order_stage'],'lift')
        self.assertEqual(self.order.apply(self.prediction(),0)['order_stage'],'base')
        self.assertEqual(self.order.apply(self.prediction(),2)['order_stage'],'retreat')

    def test_confirm_lift_outside_lift_stage_is_rejected(self):
        with self.assertRaises(ValueError):self.order.confirm_lift()
        self.order.apply(self.prediction(),0)
        with self.assertRaises(ValueError):self.order.confirm_lift()


if __name__=='__main__':unittest.main()
