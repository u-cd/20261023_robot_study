import threading
import unittest
from scripts.return_to_initial import ReturnToInitial


class FakeArm:
    def __init__(self):
        self.values=[73,116,46,82,93]
        self.requests=[]
        self.active=True
        self.on_step=None
        self.ignore_steps=False
    def send(self,request):
        self.requests.append(dict(request))
        if request['action']=='step' and not self.ignore_steps:
            self.values[request['joint']-1] += 1 if request['direction']=='+' else -1
            if self.on_step:self.on_step()
        return {'connected':True,'joints':[dict(joint=i+1,active=self.active,angle=a,target=a,speed=100) for i,a in enumerate(self.values)]}


class ReturnTests(unittest.TestCase):
    def test_sequential_exact_targets_and_no_detach(self):
        arm=FakeArm();targets=[70,113,43,80,90]
        home=ReturnToInitial(arm.send,targets)
        self.assertEqual(arm.requests,[]) # construction never touches the robot
        home.start();home.thread.join(3)
        self.assertFalse(home.active)
        self.assertEqual(arm.values,targets)
        self.assertTrue(all(r['action'] in ('status','stop','step') for r in arm.requests))
        axes=[r['joint'] for r in arm.requests if r['action']=='step']
        self.assertEqual(list(dict.fromkeys(axes)),[2,3,1,4,5])

    def test_stop_prevents_remaining_steps(self):
        arm=FakeArm();home=ReturnToInitial(arm.send,[70,113,43,80,90])
        arm.on_step=home.cancel
        home.start();home.thread.join(2)
        self.assertFalse(home.active)
        self.assertEqual(sum(r['action']=='step' for r in arm.requests),1)
        self.assertEqual(arm.requests[-1],{'action':'stop'})
        self.assertIn('停止',home.message)

    def test_inactive_joint_rejected_without_motion(self):
        arm=FakeArm();arm.active=False
        home=ReturnToInitial(arm.send,[70,113,43,80,90])
        with self.assertRaises(ValueError):home.start()
        self.assertFalse(home.active)
        self.assertEqual(arm.requests,[{'action':'status'}])

    def test_detects_unexpected_other_joint_change(self):
        arm=FakeArm();home=ReturnToInitial(arm.send,[70,113,43,80,90])
        arm.on_step=lambda:arm.values.__setitem__(0,arm.values[0]+1)
        home.start();home.thread.join(2)
        self.assertIn('想定外',home.message)
        self.assertEqual(sum(r['action']=='step' for r in arm.requests),1)
        self.assertEqual(arm.requests[-1],{'action':'stop'})


if __name__=='__main__':unittest.main()
