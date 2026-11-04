"""Read-only startup/page and fake-device recording tests; no sockets or hardware."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vision.arm_camera_panel import make_handler, page_html


class CollectionPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.calls=[]
        self.state={'connected':True,'joints':[dict(joint=i,active=True,angle=a,target=a,speed=25)
            for i,a in enumerate((90,100,80,80,90),1)]}
        class Camera:
            source='fake-camera'
            def snapshot(self,fresh=False):
                return np.zeros((20,30,3),dtype=np.uint8),'2026-11-05T00:00:00+00:00',1
        def send(data):
            self.calls.append(data)
            if data['action']=='step':
                j=self.state['joints'][data['joint']-1]
                j['angle']+=1 if data['direction']=='+' else -1;j['target']=j['angle']
            return copy.deepcopy(self.state)
        self.handler,self.rec,self.lock=make_handler(Camera(),send,'test-token',Path(self.tmp.name))

    def request(self,path,data=None,token='test-token'):
        handler=self.handler.__new__(self.handler);handler.path=path
        body=json.dumps(data).encode() if data is not None else b''
        handler.headers={'Content-Length':str(len(body)),'X-Arm-Token':token}
        handler.rfile=io.BytesIO(body);result=[]
        handler.send=lambda code,content,mime='application/json':result.append((code,content,mime))
        if data is None:handler.do_GET()
        else:handler.do_POST()
        return result[0]

    def test_creation_and_page_do_not_access_arm_or_calibration(self):
        self.assertEqual(self.calls,[])
        code,page,_=self.request('/')
        self.assertEqual(code,200)
        self.assertIn('demo-section',page);self.assertIn('live-camera',page)
        self.assertNotIn('calibration-section',page)
        self.assertNotIn('__TOKEN__',page)
        self.assertEqual(self.calls,[])
        self.assertEqual(self.request('/live.jpg')[0],200)
        self.assertEqual(self.calls,[])

    def test_recording_uses_only_fake_devices_and_keeps_raw_files(self):
        self.assertEqual(self.request('/demo',{'action':'start'})[0],200)
        self.assertEqual(self.request('/api',{'action':'step','joint':1,'direction':'+'})[0],400)
        self.assertEqual(self.request('/demo',{'action':'step','joint':1,'direction':'+'})[0],200)
        self.assertEqual(self.request('/demo',{'action':'finish','outcome':'success'})[0],200)
        folder=self.rec.folder
        self.assertTrue((folder/'setup.jpg').exists())
        logs=[json.loads(x) for x in (folder/'samples.jsonl').read_text().splitlines()]
        result=[x for x in logs if x['event']=='result'][0]
        self.assertEqual(result['delta_deg'],1)
        self.assertTrue((folder/result['image']).exists())

    def test_auth_invalid_json_and_retired_routes(self):
        self.assertEqual(self.request('/demo',{'action':'start'},token='bad')[0],403)
        self.assertEqual(self.request('/demo',[])[0],400)
        self.assertEqual(self.request('/policy',{'action':'execute'})[0],404)
        self.assertEqual(self.request('/position-test/preview',{})[0],404)
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
