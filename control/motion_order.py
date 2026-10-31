"""Eight joint-group steps, with operator-confirmed arm/grip/lift completion."""
import numpy as np

from control.imitation_record import ACTIONS

FLOW = ['台座', '肩・肘', 'つかむ', '持ち上げ', '台座', '肩・肘', '離す', '退避']


class MotionOrder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = None
        self.stage = None
        self.reviewed_stage = None

    def status(self):
        labels = {'base': '台座の位置合わせ', 'lift': '持ち上げ：肩・肘のみ',
                  'arm': '位置合わせ：肩・肘のみ', 'grip': 'つかむ：グリップを閉じる' if self.phase==0 else '離す：グリップを開く',
                  'retreat': '退避：肩・肘のみ', 'done': '退避完了'}
        steps = {(0,'base'):1,(0,'arm'):2,(0,'grip'):3,(1,'lift'):4,(1,'base'):5,
                 (1,'arm'):6,(1,'grip'):7,(2,'retreat'):8,(2,'done'):8}
        next_labels = {'arm':'位置合わせ完了 → '+('つかむ' if self.phase==0 else '離す'),
                       'grip':'つかめた → 持ち上げ' if self.phase==0 else '離せた → 退避',
                       'lift':'持ち上げ完了 → 台座へ','retreat':'退避完了 → 終了'}
        return {'phase':self.phase,'order_stage':self.stage,'order_label':labels.get(self.stage,'未開始'),
                'order_step':steps.get((self.phase,self.stage)), 'order_flow':FLOW,
                'order_can_advance':self.stage in next_labels and self.reviewed_stage==(self.phase,self.stage),'order_next_label':next_labels.get(self.stage,'完了' if self.stage=='done' else '台座は目標到達で自動切替'),
                'lift_confirmation_required':self.stage=='lift'}

    def advance(self):
        if self.reviewed_stage!=(self.phase,self.stage):
            raise ValueError('新しい操作区分の予測を確認してから、完了を確認してください。')
        if self.stage=='arm':self.stage='grip'
        elif self.stage=='grip':
            self.phase += 1
            self.stage='lift' if self.phase==1 else 'retreat'
        elif self.stage=='lift':self.stage='base'
        elif self.stage=='retreat':self.stage='done'
        else:raise ValueError('予測を更新し、現在の肩・肘またはグリップの操作を確認してください。台座は目標到達で自動切替します。')
        self.reviewed_stage=None
        return self.status()

    def confirm_lift(self):
        if self.phase!=1 or self.stage!='lift':
            raise ValueError('「はなすまで」の持ち上げ中に確認してください。')
        self.advance()

    def apply(self, prediction, phase):
        p = dict(prediction)
        if self.phase != phase:
            self.phase = phase
            self.stage = 'base' if phase == 0 else ('lift' if phase == 1 else 'retreat')
        if self.stage == 'base' and 'base_goal_deg' in p and abs(p['base_error_deg']) <= 4:
            self.stage = 'arm'
        self.reviewed_stage=(self.phase,self.stage)
        p.update(self.status(),order_blocked_reason='')
        if self.stage == 'base':
            if 'base_goal_deg' not in p:
                allowed=[8]
                p['order_blocked_reason']='台座の目標角度を推定できません。対象の検出位置を確認してください。'
            else:allowed=[0 if p['base_error_deg']>0 else 1]
        elif self.stage == 'lift':
            allowed=[3,5,8]  # Demonstrations used shoulder- and elbow- for initial lifting.
        elif self.stage in ('arm','retreat'):
            allowed=[2,3,4,5,8]  # Gripper cannot open/close during arm motion.
        elif self.stage == 'grip':
            allowed=[6] if phase==0 else [7]
        else:
            allowed=[8]
            p['order_blocked_reason']='搬送と退避の確認が完了しました。新しい試行は操作順をリセットしてください。'
        scores = np.asarray(p['probabilities'], dtype=float)
        label = max(allowed, key=lambda n: scores[n])
        p.update(label=label,action=ACTIONS[label],joint=None if label==8 else (1,2,3,5)[label//2],
                 direction=None if label==8 else ('+' if label%2==0 else '-'),
                 confidence=float(scores[label]),order_allowed_labels=allowed)
        p['selection_note']=(p.get('selection_note','')+' ｜ 順番：'+p['order_label']).strip(' ｜')
        return p
