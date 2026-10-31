"""Single bounded model-action trials; preview never sends motion commands."""
import secrets
import time

from control.fine_step import fine_step
from vision.arm_position_test import stationary_commands


class PolicyTrial:
    def __init__(self, policy, detector, send, base_alignment=None, motion_order=None):
        self.policy, self.detector, self.send = policy, detector, send
        self.ticket = None
        self.base_alignment = base_alignment
        self.motion_order = motion_order

    def preview(self, phase, use_base_alignment=True, use_motion_order=True):
        if type(phase) is not int or not 0<=phase<3:
            raise ValueError('段階を選択してください。')
        state = self.send({'action':'status'})
        commands = stationary_commands(state)
        if commands[3]!=80:raise ValueError('手首を80°にしてください。')
        obs = self.detector.observation()
        prediction = self.policy.predict(obs['detections'],state,phase)
        if (use_base_alignment or (use_motion_order and self.motion_order is not None)) and self.base_alignment is not None:
            prediction=self.base_alignment.apply(prediction,obs['detections'],state,phase)
        if self.motion_order is not None:
            if use_motion_order:prediction=self.motion_order.apply(prediction,phase)
            else:self.motion_order.reset()
        reasons=[]
        if prediction.get('order_blocked_reason'):reasons.append(prediction['order_blocked_reason'])
        if not obs['detections']:reasons.append('スポンジ・coinがどちらも検出されていません。')
        if not prediction['within_joint_training_bounds']:reasons.append('学習時の関節指令範囲から外れています。')
        joint=prediction['joint'];direction=prediction['direction']
        if joint:
            low,high=(50,150) if joint==5 else (0,180)
            if not low<=commands[joint-1]+(1 if direction=='+' else -1)<=high:
                reasons.append('関節の指令範囲の端です。')
        prediction.update(can_execute=not reasons,blocked_reasons=reasons,
            frame_sequence=obs['sequence'],frame_captured_at=obs['captured_at'],
            detections=obs['detections'],commanded_angles=commands,
            execute_degrees=0 if joint is None else 1, ticket=secrets.token_urlsafe(24),
            note='台座補助が有効な場合は候補を絞り、残りの最大予測を選択。1要求1°。段階切替は手動。')
        self.ticket={'id':prediction['ticket'],'at':time.monotonic(),'prediction':prediction}
        return prediction

    def confirm_lift(self):
        if self.motion_order is None:raise ValueError('順番制御が無効です。')
        stationary_commands(self.send({'action':'status'}))
        self.motion_order.confirm_lift()
        self.ticket=None
        return {'order_stage':'base','message':'目視の持ち上げ完了を受け付けました。次は台座の位置合わせです。'}

    def advance_order(self):
        if self.motion_order is None:raise ValueError('順番制御が無効です。')
        stationary_commands(self.send({'action':'status'}))
        result=self.motion_order.advance()
        self.ticket=None
        result['message']='確認しました。次は「'+result['order_label']+'」です。長押しで再開してください。'
        return result

    def reset_order(self):
        self.ticket=None
        if self.motion_order is not None:self.motion_order.reset()
        if self.base_alignment is not None:
            self.base_alignment.cached_goal=None
            self.base_alignment.cached_phase=None
        return {'message':'操作順をリセットしました。予測を更新してください。'}

    def execute(self, ticket):
        saved=self.ticket
        self.ticket=None  # One-shot even after a failed check.
        if not saved or ticket!=saved['id'] or time.monotonic()-saved['at']>15:
            raise ValueError('予測を更新してから実行してください（15秒以内）。')
        p=saved['prediction']
        if not p['can_execute']:raise ValueError('この予測は実行できません。')
        obs=self.detector.observation()  # Prove camera/inference are still fresh.
        if not obs['detections']:raise ValueError('現在の検出がありません。予測を更新してください。')
        for name, item in p['detections'].items():
            current=obs['detections'].get(name)
            if not current or max(abs(a-b) for a,b in zip(item['center_normalized'],current['center_normalized']))>.05:
                raise ValueError('物体の検出位置が変わりました。予測を更新してください。')
        state=self.send({'action':'status'})
        if stationary_commands(state)!=p['commanded_angles']:
            raise ValueError('指令姿勢が変わりました。予測を更新してください。')
        if p['joint'] is None:return {'prediction':p,'state':state,'moved':False}
        state=fine_step(self.send,{'joint':p['joint'],'direction':p['direction']})
        return {'prediction':p,'state':state,'moved':True}
