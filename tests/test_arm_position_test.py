import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vision.arm_position_test import position_preview, save_position_test, stationary_commands

ROOT=Path(__file__).resolve().parents[1]


class PositionTestTests(unittest.TestCase):
    def preview(self):
        return {'target_xyz_mm':[15,135,0],'all_commands_deg':[93,144,63,80,150]}

    def state(self):
        return {'connected':True,'joints':[{'joint':i+1,'active':True,'angle':q,'target':q}
            for i,q in enumerate(self.preview()['all_commands_deg'])]}

    def test_candidate_from_real_model_and_extrapolation_rejected(self):
        model=ROOT/'runs/arm_kinematics/model.json'
        if not model.exists():self.skipTest('ローカルの11点モデルが未生成')
        result=position_preview(model,15,135)
        self.assertEqual(result['all_commands_deg'],[93,144,63,80,150])
        self.assertFalse(result['motion_executed'])
        with self.assertRaises(ValueError):position_preview(model,100,100)

    def test_inactive_or_moving_status_rejected(self):
        state=self.state();state['joints'][1]['active']=False
        with self.assertRaises(ValueError):stationary_commands(state)
        state=self.state();state['joints'][1]['target']+=1
        with self.assertRaises(ValueError):stationary_commands(state)

    def test_record_contains_images_and_no_claim_of_measured_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            calibration={'matrix':np.eye(3),'corners_px':[[0,0],[50,0],[50,50],[0,50]]}
            path=save_position_test(tmp,self.preview(),self.state(),self.state(),
                np.zeros((160,100,3),dtype=np.uint8),'time',42,calibration,'test')
            record=json.loads((path/'record.json').read_text())
            self.assertIsNone(record['actual_error_mm'])
            self.assertTrue((path/'raw.png').is_file())
            self.assertTrue((path/'marked.png').is_file())

    def test_mismatch_or_changed_commands_do_not_create_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=self.state();state['joints'][0]['angle']=92;state['joints'][0]['target']=92
            for before,after in [(state,state),(self.state(),state)]:
                with self.assertRaises(ValueError):
                    save_position_test(tmp,self.preview(),before,after,None,'time',42,{})
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_panel_routes_and_html_without_hardware(self):
        from vision import arm_camera_panel as panel
        original_page=(ROOT/'control/arm_panel.html').read_text().replace('__TOKEN__','test-token')
        class Response(io.BytesIO):
            status=200
            headers={'Content-Type':'text/html'}
        class Camera:
            def __init__(self,*args):pass
            def close(self):pass
        class Server:
            def __init__(server,address,handler):server.handler=handler
            def server_close(server):pass
            def serve_forever(server):
                handler=server.handler.__new__(server.handler)
                responses=[]
                handler.send=lambda code,content,mime,headers=None:responses.append((code,content))
                handler.path='/'
                handler.do_GET()
                self.assertEqual(responses[-1][0],200)
                html=responses[-1][1].decode()
                self.assertIn('position-calculate',html)
                self.assertNotIn('__TOKEN__',html)
                self.assertLess(html.index('function enqueue('),html.index('let positionCandidate='))
                for token,xy,expected in [('wrong',[15,135],403),('test-token',[15,135],200),('test-token',[100,100],400)]:
                    body=json.dumps(dict(zip(('x','y'),xy))).encode()
                    handler.path='/position-test/preview';handler.rfile=io.BytesIO(body)
                    handler.headers={'Content-Length':str(len(body)),'X-Arm-Token':token}
                    handler.do_POST()
                    self.assertEqual(responses[-1][0],expected)
                handler.path='/position-test/record';body=b'{"x":15,"y":135}'
                handler.rfile=io.BytesIO(body);handler.headers={'Content-Length':str(len(body)),'X-Arm-Token':'test-token'}
                handler.do_POST();self.assertEqual(responses[-1][0],400)
        with patch.object(panel,'LiveCamera',Camera),patch.object(panel,'ThreadingHTTPServer',Server),patch.object(panel,'urlopen',side_effect=lambda *a,**k:Response(original_page.encode())) as upstream,patch('sys.argv',['arm_camera_panel.py']):
            panel.main()
            # Startup and HTML load only; preview must never send a serial action.
            self.assertEqual(upstream.call_count,2)


if __name__=='__main__':unittest.main()
