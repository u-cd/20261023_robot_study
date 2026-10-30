"""手動で合わせた基準点と指令値・写真を記録する。移動指令は生成しない。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]


def apply_hover_condition(path):
    """接続中の旧記録サーバーが作るZ=0のデータも、セッション条件に従い訂正する。"""
    path=Path(path)
    condition_path=path.parent.parent/'measurement_conditions.json'
    if not condition_path.is_file():
        return
    conditions=json.loads(condition_path.read_text())
    if conditions.get('height_mode')!='hover_unmeasured':
        return
    record=json.loads(path.read_text())
    if record.get('height_condition')=='hover_unmeasured':
        return
    backup=path.with_name('sample_before_height_correction.json')
    if not backup.exists():shutil.copy2(path,backup)
    record['reference_xyz_mm']=[*record['reference_xyz_mm'][:2],None]
    record['reference_xy_mm']=record['reference_xyz_mm'][:2]
    record['height_condition']='hover_unmeasured'
    record['paper_contact']=False
    record['contact_and_stillness_confirmed_by_user']=False
    record['hover_and_stillness_confirmed_by_user']=True
    record['height_condition_source']='User clarified: tip is just above paper, without contact; clearance not measured.'
    record['height_note']='Z is unknown, not zero; equal height across samples has not been measured or verified.'
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')



class CalibrationRecorder:
    def __init__(self, arm, calibration_path, output_root=None):
        self.arm=arm
        self.source=Path(calibration_path)
        raw=self.source.read_bytes()
        self.calibration=json.loads(raw)
        if self.calibration.get('schema_version')!=2 or self.calibration.get('status')!='passed':
            raise ValueError('合格した格子校正JSONを指定してください。')
        self.points={p['number']:p['expected_mm'] for p in self.calibration['points']}
        self.session=(Path(output_root) if output_root else ROOT/'runs/arm_calibration')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.session.mkdir(parents=True)
        self.calibration_path=self.session/'table_calibration.json'
        self.calibration_path.write_bytes(raw)
        self.calibration_hash=hashlib.sha256(raw).hexdigest()
        (self.session/'measurement_conditions.json').write_text(json.dumps({'height_mode':'hover_unmeasured','paper_contact':False,'clearance_mm':None,'tip':'center of closed gripper tip'},ensure_ascii=False,indent=2)+'\n')
        self.samples={}
        self.gripper_angle=None

    def point_options(self):
        return [{'number':n,'xy':xy} for n,xy in sorted(self.points.items())]

    @staticmethod
    def check_status(joints):
        if len(joints)!=5 or sorted(j['joint'] for j in joints)!=[1,2,3,4,5] or not all(j['active'] for j in joints):
            raise ValueError('全5関節を有効にし、グリッパーを閉じてから記録してください。')
        if any(j['angle']!=j['target'] for j in joints):
            raise ValueError('指令が移動中です。停止・静止を確認してから記録してください。')
        return [j['angle'] for j in sorted(joints,key=lambda j:j['joint'])]

    def record(self,data):
        point=data.get('point')
        if type(point) is not int or point not in self.points:
            raise ValueError('紙上の点番号を選んでください。')
        if data.get('confirmed') is not True:
            raise ValueError('閉じたグリップの先端中心が交点の直上で紙に触れず、静止していることを確認してください。')
        note=data.get('note','')
        approach=data.get('approach','unspecified')
        if not isinstance(note,str) or len(note)>200 or approach not in ('same_direction','opposite_direction','unspecified'):
            raise ValueError('接近方向またはメモが不正です。')
        # Hold the same lock as every manual command while reading / photographing.
        with self.arm.lock:
            if time.monotonic()-self.arm.last_motion_at<1:
                raise ValueError('最後の操作から1秒以上待ち、先端の静止を確認してください。')
            before=self.arm.status()
            angles=self.check_status(before)
            if self.gripper_angle is not None and angles[4]!=self.gripper_angle:
                raise ValueError(f'基準点を一定にするため、グリッパーは最初の記録と同じ{self.gripper_angle}度にしてください。')
            sample_id=datetime.now().strftime('%H%M%S_')+uuid.uuid4().hex[:8]
            folder=self.session/sample_id
            folder.mkdir()
            try:
                subprocess.run([sys.executable,str(ROOT/'vision/capture_arm_sample.py'),
                    '--calibration',str(self.calibration_path),'--point',str(point),
                    '--angles',','.join(map(str,angles)),'--output',str(folder)],
                    capture_output=True,text=True,timeout=15,check=True)
                after=self.arm.status()
                if self.check_status(after)!=angles:
                    raise ValueError('撮影中に指令値が変化したため、測定として保存しません。')
                capture=json.loads((folder/'capture.json').read_text())
                repeat=1+sum(s['point_number']==point for s in self.samples.values())
                record={'sample_id':sample_id,'recorded_at':datetime.now(timezone.utc).isoformat(),
                    'point_number':point,'reference_xyz_mm':[*self.points[point],None],
                    'reference_xy_mm':self.points[point], 'height_condition':'hover_unmeasured',
                    'paper_contact':False,'height_note':'Z is unknown, not zero; equal height across samples has not been measured or verified.',
                    'reference_frame':'calibration_rectangle','tip':'center of closed gripper tip',
                    'joint_command_degrees':angles,'joint_status_before':before,'joint_status_after':after,
                    'joint_angles_measured':False,'hover_and_stillness_confirmed_by_user':True,
                    'approach':approach,'note':note,'repeat_for_point':repeat,
                    'calibration_sha256':self.calibration_hash,'capture':capture,
                    'raw_image':'raw.png','marked_image':'marked.png',
                    'note_on_position':'Reference grid location manually aligned by user; actual tip position not automatically measured.'}
                (folder/'sample.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
                self.samples[sample_id]=record
                self.gripper_angle=angles[4]
                return {'connected':True,'joints':after,'message':f'点{point}・{repeat}回目を保存しました。',
                    'calibration_sample':{'id':sample_id,'point':point,'repeat':repeat,
                        'path':str(folder),'angles':angles,'count':len(self.samples)}}
            except Exception as error:
                detail=error.stderr.strip() if isinstance(error,subprocess.CalledProcessError) else str(error)
                (folder/'rejected.json').write_text(json.dumps({'error':detail,'accepted':False},ensure_ascii=False,indent=2)+'\n')
                raise RuntimeError(f'記録できませんでした: {detail}') from error
