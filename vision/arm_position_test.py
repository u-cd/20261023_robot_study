"""紙面座標テストの候補計算と写真保存。移動指令は送らない。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from control.arm_kinematics import inverse_preview


def position_preview(model_path, x, y):
    content=Path(model_path).read_bytes()
    model=json.loads(content)
    result=inverse_preview(model,[float(x),float(y),0])
    result['all_commands_deg']=result['command_degrees_rounded']+[
        model['wrist_fixed_deg'],model['calibration_gripper_deg']]
    result['model_sha256']=hashlib.sha256(content).hexdigest()
    return result


def stationary_commands(state):
    joints=sorted(state.get('joints',[]),key=lambda j:j['joint'])
    if not state.get('connected') or [j['joint'] for j in joints]!=[1,2,3,4,5]:
        raise ValueError('接続して全5関節の状態を取得してください。')
    if not all(j['active'] and j['angle']==j['target'] for j in joints):
        raise ValueError('全関節を保持・静止させてください。')
    return [j['angle'] for j in joints]


def save_position_test(root, preview, before, after, frame, stamp, sequence, calibration, note=''):
    commands=stationary_commands(before)
    if commands!=stationary_commands(after):
        raise ValueError('撮影中に指令値が変わりました。静止して再記録してください。')
    if commands!=preview['all_commands_deg']:
        raise ValueError('まだ計算候補の角度に達していません。')
    marked=frame.copy()
    corners=np.rint(calibration['corners_px']).astype(np.int32)
    cv2.polylines(marked,[corners],True,(220,150,40),2,cv2.LINE_AA)
    xy=preview['target_xyz_mm'][:2]
    p=np.linalg.inv(calibration['matrix'])@np.array([*xy,1.])
    uv=tuple(np.rint(p[:2]/p[2]).astype(int))
    cv2.drawMarker(marked,uv,(0,170,255),cv2.MARKER_CROSS,24,2)
    z=preview['target_xyz_mm'][2]
    cv2.putText(marked,f'Target XY={xy} mm / Z={z:g} model',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,170,255),1,cv2.LINE_AA)
    cv2.putText(marked,f'Commanded: {commands}',(8,44),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,170,255),1,cv2.LINE_AA)
    ok_raw,raw=cv2.imencode('.png',frame)
    ok_marked,encoded=cv2.imencode('.png',marked)
    if not ok_raw or not ok_marked:raise RuntimeError('写真の変換に失敗しました。')
    folder=Path(root)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'raw.png').write_bytes(raw.tobytes())
    (folder/'marked.png').write_bytes(encoded.tobytes())
    record={'recorded_at':datetime.now(timezone.utc).isoformat(),'preview':preview,
            'commanded_angles_deg':commands,'frame_captured_at':stamp,'frame_sequence':sequence,
            'status_before':before,'status_after':after,'note':str(note)[:200],
            'actual_tip_position_mm':None,'actual_error_mm':None,
            'description':'候補指令と撮影時状態の記録。実測位置・実測誤差ではない。'}
    (folder/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return folder
