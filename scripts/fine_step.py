"""指令値がちょうど1度変わったことを確認する微調整。実角度の測定ではない。"""
import time


def fine_step(send, data, sleep=time.sleep):
    joint=data.get('joint')
    direction=data.get('direction')
    if type(joint) is not int or not 1<=joint<=5 or direction not in ('+','-'):
        raise ValueError('関節番号と微調整方向を確認してください。')
    state=send({'action':'status'})
    item=next(j for j in state['joints'] if j['joint']==joint)
    if not item['active'] or item['angle']!=item['target']:
        raise ValueError('関節を有効にし、移動を止めてから微調整してください。')
    low,high=(50,150) if joint==5 else (0,180)
    target=item['angle']+(1 if direction=='+' else -1)
    if not low<=target<=high:
        raise ValueError(f'指令範囲{low}〜{high}度の端です。')
    # Stop any residual JOG, keeping the enabled joints holding.
    state=send({'action':'stop'})
    start={j['joint']:j for j in state['joints']}
    if start[joint]['angle']!=item['angle'] or not start[joint]['active']:
        raise ValueError('指令状態が変わりました。状態を確認してやり直してください。')
    for attempt in range(3):
        state=send({'action':'step','joint':joint,'direction':direction})
        current={j['joint']:j for j in state['joints']}
        if any(current[n]['angle']!=start[n]['angle'] or current[n]['active']!=start[n]['active']
               for n in start if n!=joint) or not current[joint]['active']:
            raise ValueError('ほかの操作による指令の変化を検出しました。状態を更新してください。')
        value=current[joint]['angle']
        if value==target and current[joint]['target']==target:
            state['message']=f'関節{joint}の指令を{item["angle"]}° → {target}°に変更しました。'
            return state
        if value!=item['angle'] or current[joint]['target']!=item['angle']:
            raise ValueError('想定外の指令値です。状態を更新してください。')
        if attempt<2:
            # Old firmware ignores STEP within its speed interval. Retry only
            # after proving that the command value has not changed.
            sleep(1/max(5,start[joint]['speed'])+.02)
    raise RuntimeError('1度の指令変更を確認できません。状態を更新してください。')
