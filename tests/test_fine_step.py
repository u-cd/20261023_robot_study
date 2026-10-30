import copy
import unittest
from control.fine_step import fine_step


class Controller:
    def __init__(self,skip_first=False):
        self.joints=[{'joint':i,'active':True,'angle':90,'target':90,'speed':25} for i in range(1,6)]
        self.calls=[]
        self.skip_first=skip_first

    def send(self,data):
        self.calls.append(data)
        if data['action']=='step':
            if self.skip_first:self.skip_first=False
            else:
                joint=self.joints[data['joint']-1]
                joint['angle']+=1 if data['direction']=='+' else -1
                joint['target']=joint['angle']
        return {'connected':True,'joints':copy.deepcopy(self.joints),'message':''}


class FineStepTests(unittest.TestCase):
    def test_exactly_one_degree_even_when_first_step_is_ignored(self):
        controller=Controller(skip_first=True)
        waits=[]
        state=fine_step(controller.send,{'joint':2,'direction':'+'},sleep=waits.append)
        self.assertEqual([j['angle'] for j in state['joints']],[90,91,90,90,90])
        self.assertEqual(sum(c['action']=='step' for c in controller.calls),2)
        self.assertEqual(len(waits),1)

    def test_successful_step_is_not_repeated(self):
        controller=Controller()
        fine_step(controller.send,{'joint':1,'direction':'-'})
        self.assertEqual(controller.joints[0]['angle'],89)
        self.assertEqual(sum(c['action']=='step' for c in controller.calls),1)

    def test_limits_inactive_and_pending_motion_reject_steps(self):
        for field,value in [('angle',180),('active',False),('target',100)]:
            controller=Controller()
            controller.joints[0][field]=value
            if field=='angle':controller.joints[0]['target']=value
            with self.assertRaises(ValueError):fine_step(controller.send,{'joint':1,'direction':'+'})
            self.assertFalse(any(c['action']=='step' for c in controller.calls))
        controller=Controller()
        controller.joints[4]['angle']=controller.joints[4]['target']=50
        with self.assertRaises(ValueError):fine_step(controller.send,{'joint':5,'direction':'-'})

    def test_invalid_request_never_reaches_controller(self):
        controller=Controller()
        with self.assertRaises(ValueError):fine_step(controller.send,{'joint':True,'direction':'+'})
        self.assertEqual(controller.calls,[])

    def test_unexpected_motion_is_not_followed_by_retry(self):
        controller=Controller()
        def send(data):
            state=controller.send(data)
            if data['action']=='step':state['joints'][1]['angle']+=1
            return state
        with self.assertRaises(ValueError):fine_step(send,{'joint':1,'direction':'+'})
        self.assertEqual(sum(c['action']=='step' for c in controller.calls),1)


if __name__=='__main__':unittest.main()
