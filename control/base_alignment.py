"""Demonstration-derived base angle guide, separate from the BC action classifier."""
import json
from pathlib import Path

import numpy as np

from control.imitation_record import ACTIONS


class BaseAlignment:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text())
        self.cached_phase = None
        self.cached_goal = None

    def apply(self, prediction, detections, state, phase):
        p = dict(prediction)
        p['raw_action'] = p['action']
        p['selection_note'] = ''
        if self.cached_phase != phase:
            self.cached_goal = None
            self.cached_phase = phase
        model = self.config['models'].get(str(phase))
        if model is None:return p
        item = detections.get(model['object'])
        if item:
            uv = np.asarray(item['center_normalized'],dtype=float)
            low,high=np.asarray(model['uv_min']),np.asarray(model['uv_max'])
            if np.any(uv<low-.03) or np.any(uv>high+.03):
                p['selection_note']='台座補助：対象位置が学習範囲外のため適用なし'
                return p
            z=(uv-np.asarray(model['mean']))/np.asarray(model['scale'])
            row=[1.,z[0],z[1]]
            if model['quadratic']:row += [z[0]*z[1],z[0]**2,z[1]**2]
            self.cached_goal=float(np.dot(row,model['weights']))
        if self.cached_goal is None:
            p['selection_note']='台座補助：対象の検出位置を待っています'
            return p
        current=next(j['angle'] for j in state['joints'] if j['joint']==1)
        error=self.cached_goal-current
        if abs(error)<=self.config['tolerance_deg']:
            masked=[0,1]
            reason='台座が推定目標の許容範囲に入ったため、台座以外の最大予測を選択'
        else:
            masked=[1 if error>0 else 0]
            reason='推定目標から遠ざかる台座方向を除き、残りの最大予測を選択'
        scores=np.asarray(p['probabilities'],dtype=float).copy()
        scores[masked]=-1
        label=int(scores.argmax())
        p.update(base_goal_deg=self.cached_goal,base_error_deg=error,masked_labels=masked,
            label=label,action=ACTIONS[label],joint=None if label==8 else (1,2,3,5)[label//2],
            direction=None if label==8 else ('+' if label%2==0 else '-'),
            confidence=float(p['probabilities'][label]))
        if p['action']!=p['raw_action']:p['selection_note']=reason
        return p
