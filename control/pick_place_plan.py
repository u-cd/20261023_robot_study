"""固定手首での初回搬送候補。角度計算のみで実機には接続しない。"""
import hashlib
import json
from pathlib import Path

import numpy as np

from control.arm_kinematics import forward, inverse_preview


def build_transfer(model_path, pick_xy, place_xy, current_commands, pick_z=15,
                   travel_z=40, release_z=15, open_angle=60, closed_angle=150):
    content=Path(model_path).read_bytes();model=json.loads(content)
    q=np.asarray(current_commands,dtype=float)
    if q.shape!=(5,) or not np.isfinite(q).all() or np.any(q!=np.rint(q)):
        raise ValueError('現在の5関節の整数指令値が必要です。')
    if np.any(q[:4]<0) or np.any(q[:4]>180) or not 50<=q[4]<=150:
        raise ValueError('現在指令が範囲外です。')
    if q[3]!=model['wrist_fixed_deg']:
        raise ValueError('手首を80度に合わせてください。')
    if not np.isfinite([pick_z,travel_z,release_z]).all() or not 0<pick_z<travel_z<=80 or not 0<release_z<travel_z:
        raise ValueError('把持・解放高さは0より大きく、搬送高さより低くしてください。搬送高さは80mm以下です。')
    if type(open_angle) is not int or type(closed_angle) is not int or not 50<=open_angle<closed_angle<=150:
        raise ValueError('グリップは50〜150度で、開く角度を閉じる角度より小さくしてください。')
    current_xyz=forward(model['parameters'],q[:3],model['signs'])[0]
    if current_xyz[2]>travel_z+2:
        raise ValueError('現在位置が搬送高さより高い候補です。姿勢・高さを確認してください。')
    stages=[]
    last_xyz=current_xyz.tolist()
    for label,xy,z,grip in [
        ('現在位置の上へ上げる',current_xyz[:2],travel_z,int(q[4])),
        ('スポンジの上へ移動',pick_xy,travel_z,int(q[4])),
        ('グリッパーを開く',pick_xy,travel_z,open_angle),
        ('把持高さへ下げる',pick_xy,pick_z,open_angle),
        ('スポンジをつかむ',pick_xy,pick_z,closed_angle),
        ('持ち上げる',pick_xy,travel_z,closed_angle),
        ('ターゲットの上へ移動',place_xy,travel_z,closed_angle),
        ('解放高さへ下げる',place_xy,release_z,closed_angle),
        ('グリッパーを開いて放す',place_xy,release_z,open_angle),
        ('上へ退避する',place_xy,travel_z,open_angle)]:
        xyz=[*np.asarray(xy,dtype=float).tolist(),float(z)]
        if len(xyz)!=3:raise ValueError('把持位置・置き先はXYの2値にしてください。')
        candidate=inverse_preview(model,xyz,allow_unvalidated_height=True)
        steps=max(1,int(np.ceil(np.linalg.norm(np.array(xyz)-last_xyz)/5)))
        waypoints=[]
        # Solve at intervals <=5mm rather than taking a single joint-space jump.
        for t in np.linspace(0,1,steps+1)[1:]:
            p=(1-t)*np.array(last_xyz)+t*np.array(xyz)
            c=inverse_preview(model,p,allow_unvalidated_height=True)
            waypoints.append({'xyz_mm':p.tolist(),'all_commands_deg':c['command_degrees_rounded']+[int(q[3]),grip]})
        stages.append({'name':label,'target_xyz_mm':xyz,
                       'all_commands_deg':candidate['command_degrees_rounded']+[int(q[3]),grip],
                       'waypoints':waypoints,'height_verified':False})
        last_xyz=xyz
    return {'model_sha256':hashlib.sha256(content).hexdigest(),'start_commands_deg':q.astype(int).tolist(),
            'pick_xy_mm':list(pick_xy),'place_xy_mm':list(place_xy),
            'pick_height_mm':pick_z,'travel_height_mm':travel_z,'release_height_mm':release_z,
            'stages':stages,'motion_executed':False,'physical_path_verified':False,
            'grasp_verified':False,'height_generalization_verified':False}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('runs/arm_kinematics/model.json'))
    parser.add_argument('--pick',nargs=2,type=float,required=True,metavar=('X','Y'))
    parser.add_argument('--place',nargs=2,type=float,required=True,metavar=('X','Y'))
    parser.add_argument('--current',nargs=5,type=int,required=True)
    parser.add_argument('--pick-z',type=float,default=15)
    parser.add_argument('--travel-z',type=float,default=40)
    parser.add_argument('--release-z',type=float,default=15)
    parser.add_argument('--output',type=Path,default=Path('runs/pick_place/preview.json'))
    args=parser.parse_args()
    try:plan=build_transfer(args.model,args.pick,args.place,args.current,args.pick_z,args.travel_z,args.release_z)
    except ValueError as error:parser.error(str(error))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    for i,stage in enumerate(plan['stages'],1):print(i,stage['name'],stage['target_xyz_mm'],stage['all_commands_deg'])
    print('候補のみ保存。アームは動かしていません：',args.output)


if __name__=='__main__':main()
