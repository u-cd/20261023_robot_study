"""Return commanded positions sequentially via existing STEP API; never detach/START."""
import threading
import time


class ReturnToInitial:
    ORDER = (2, 3, 1, 4, 5) # shoulder, elbow, base, wrist, gripper

    def __init__(self, send, targets, send_lock=None):
        if len(targets)!=5 or any(type(a) is not int or not (50 if i==4 else 0)<=a<=(150 if i==4 else 180) for i,a in enumerate(targets)):
            raise ValueError('開始姿勢の設定が不正です。')
        self.send = send
        self.targets = list(targets)
        self.send_lock = send_lock or threading.RLock()
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.thread = None
        self.active = False
        self.message = '待機中'
        self.last_state = None

    @staticmethod
    def angles(state):
        joints=sorted(state.get('joints',[]),key=lambda j:j['joint'])
        if not state.get('connected') or [j['joint'] for j in joints]!=[1,2,3,4,5] or not all(j['active'] for j in joints):
            raise ValueError('全5関節を有効・保持にしてから戻してください。保持解除は行いません。')
        return [j['angle'] for j in joints]

    def start(self):
        with self.lock:
            if self.active:
                raise ValueError('開始姿勢へ移動中です。')
            state=self.send({'action':'status'})
            self.angles(state)
            self.cancelled.clear()
            self.last_state=state
            self.active=True
            self.message='開始姿勢へ戻しています。周囲と姿勢を確認してください。'
            self.thread=threading.Thread(target=self.run,daemon=True)
            self.thread.start()

    def cancel(self):
        self.cancelled.set()

    def status(self):
        with self.lock:
            return {'active':self.active,'message':self.message,'state':self.last_state,
                    'targets_deg':self.targets,'order':list(self.ORDER)}

    def run(self):
        try:
            # Halt any residual JOG while leaving all joints enabled.
            with self.send_lock:
                state=self.send({'action':'stop'})
            current=self.angles(state)
            for joint in self.ORDER:
                deadline=time.monotonic()+max(10,abs(current[joint-1]-self.targets[joint-1])*.6+5)
                while current[joint-1]!=self.targets[joint-1]:
                    if self.cancelled.is_set():
                        raise InterruptedError('開始姿勢への移動を停止しました。保持は継続します。')
                    if time.monotonic()>deadline:
                        raise TimeoutError('指令角度の変化を確認できないため停止しました。')
                    direction='+' if current[joint-1]<self.targets[joint-1] else '-'
                    before=current[:]
                    started=time.monotonic()
                    # Cancellation and actual STEP transmission are serialized with STOP.
                    with self.send_lock:
                        if self.cancelled.is_set():
                            raise InterruptedError('開始姿勢への移動を停止しました。保持は継続します。')
                        state=self.send({'action':'step','joint':joint,'direction':direction})
                    current=self.angles(state)
                    delta=current[joint-1]-before[joint-1]
                    if delta not in (0,1 if direction=='+' else -1) or any(current[i]!=before[i] for i in range(5) if i!=joint-1):
                        raise RuntimeError('別の操作または想定外の指令変化を検出したため停止しました。')
                    with self.lock:
                        self.last_state=state
                        self.message=f'関節{joint}：{current[joint-1]}° → {self.targets[joint-1]}°'
                    speed=next(j['speed'] for j in state['joints'] if j['joint']==joint)
                    self.cancelled.wait(max(0,1/max(5,speed)-(time.monotonic()-started)))
            if self.cancelled.is_set():
                raise InterruptedError('開始姿勢への移動を停止しました。保持は継続します。')
            with self.lock:
                self.message='開始姿勢の指令値に戻りました。実際の姿勢も目視で確認してください。'
        except Exception as error:
            with self.lock:
                self.message=str(error)
        finally:
            try:
                with self.send_lock:
                    state=self.send({'action':'stop'})
                with self.lock:
                    self.last_state=state
            except Exception as error:
                with self.lock:
                    self.message += ' / 停止応答を確認できません：'+str(error)
            with self.lock:
                self.active=False
