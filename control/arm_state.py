"""Shared commanded-joint state validation; no geometry or device I/O."""

def stationary_commands(state):
    joints=sorted(state.get('joints',[]),key=lambda j:j['joint'])
    if not state.get('connected') or [j['joint'] for j in joints]!=[1,2,3,4,5]:
        raise ValueError('接続して全5関節の状態を取得してください。')
    if not all(j['active'] and j['angle']==j['target'] for j in joints):
        raise ValueError('全関節を保持・静止させてください。')
    return [j['angle'] for j in joints]
