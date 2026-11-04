"""Temporal and CVAE checks for the offline ACT experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from vision.act_smoke import CompactACT, make_samples, objective


class ACTSmokeTests(unittest.TestCase):
    def test_masked_targets_do_not_change_loss_or_posterior(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        model=CompactACT(chunk=3).eval()
        image=torch.randn(2,20,512);q=torch.randn(2,4)
        target=torch.randn(2,3,4);mask=torch.tensor([[True,True,False],[True,False,False]])
        altered=target.clone();altered[~mask]=10000
        torch.manual_seed(5);p,mu,lv=model(image,q,target,mask)
        torch.manual_seed(5);pp,mm,ll=model(image,q,altered,mask)
        torch.testing.assert_close(mu,mm);torch.testing.assert_close(p,pp)
        loss=objective(p,target,mask,mu,lv)[0]
        other=objective(pp,altered,mask,mm,ll)[0]
        torch.testing.assert_close(loss,other)
        loss.backward()
        self.assertGreater(float(model.z_stats.weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with torch.no_grad():
            torch.testing.assert_close(model(image,q)[0],model(image,q)[0])

    def test_command_interpolation_and_long_gap_truncation(self):
        from vision.act_smoke import ROOT
        def stamp(seconds):return datetime.fromtimestamp(1700000000+seconds,timezone.utc).isoformat()
        def state(angle):return {'joints':[{'joint':j,'angle':angle if j==1 else 80} for j in (1,2,3,4,5)]}
        records=[]
        for i,start in enumerate((0,.2,5,5.2)):
            records.append({'index':i,'event':'result','accepted':True,'control_mode':'jog',
                            'command_sent_at':stamp(start),'interval_end_at':stamp(start+.2),
                            'frame_captured_at':stamp(start),'image':f'frames/{i}.jpg',
                            'state_before':state(i*10),'state_after':state(i*10+10)})
        # Use a disposable derived directory under the repository, never raw data.
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            folder=Path(tmp);(folder/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            rows=make_samples(folder,chunk=8,hz=5)
        self.assertEqual(len(rows),2)
        self.assertEqual([r['index'] for r in rows],[0,2])
        self.assertEqual(rows[0]['valid'].sum(),2)
        np.testing.assert_allclose(rows[0]['target'][:2,0],[10,20],atol=1e-4)
        np.testing.assert_allclose(rows[1]['target'][:2,0],[30,40],atol=1e-4)


if __name__=='__main__':unittest.main()
