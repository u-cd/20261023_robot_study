"""静止させたアームの校正写真を取得する。シリアル通信・移動指令は行わない。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np

if __package__:
    from .table_geometry import load_calibration, project
else:
    from table_geometry import load_calibration, project


def shared_frame(base_url, source):
    try:
        with urlopen(base_url+'/camera-info',timeout=2) as response:
            info=json.loads(response.read())
    except URLError:
        return None
    if info.get('service')!='arm-calibration-live-camera-v1' or info.get('camera')!=source:
        raise RuntimeError('ライブ映像と校正のカメラが一致しません。')
    with urlopen(base_url+'/snapshot.png',timeout=4) as response:
        encoded=response.read()
        stamp=response.headers.get('X-Frame-Captured-At')
        if response.headers.get('X-Camera-Source')!=source or not stamp:
            raise RuntimeError('ライブ映像の撮影情報が不正です。')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(stamp)).total_seconds()
        if not -.25<=age<=1:
            raise RuntimeError('ライブカメラの画像が古いため記録しません。')
        frame=cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError('ライブ映像から写真を取得できません。')
        return frame,stamp,{'transport':'shared_live_camera','sequence':int(response.headers['X-Frame-Sequence'])}


def direct_frame(source):
    camera=cv2.VideoCapture(int(source) if source.isdigit() else source,cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError('カメラを開けません。ライブ映像付き画面を起動するか、他のカメラアプリを終了してください。')
        camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
        for _ in range(8):
            ok,frame=camera.read()
            if not ok:raise RuntimeError('カメラ画像を取得できません。')
        return frame,datetime.now(timezone.utc).isoformat(),{'transport':'direct_camera'}
    finally:
        camera.release()


def annotate_sample(frame,calibration,point_number,xy,uv,angles):
    marked=frame.copy()
    corners=np.rint(calibration['corners_px']).astype(np.int32)
    cv2.polylines(marked,[corners],True,(220,150,40),2,cv2.LINE_AA)
    cv2.drawMarker(marked,tuple(map(round,uv)),(0,170,255),cv2.MARKER_CROSS,22,2)
    marked=cv2.copyMakeBorder(marked,100,0,0,0,cv2.BORDER_CONSTANT,value=(25,30,35))
    lines=[f"Point #{point_number}: XY ({xy[0]:g}, {xy[1]:g}) mm | tip Z unknown (hover)",
           'Cross = paper target; NOT an automatically measured tool position',
           f'Command degrees (not measured): {angles}',
           'Tip hovers just above paper; no contact; clearance not measured.']
    for i,text in enumerate(lines):
        cv2.putText(marked,text,(8,21+i*24),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,240,245),1,cv2.LINE_AA)
    return marked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--point',type=int,required=True)
    parser.add_argument('--angles',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live-camera-url',default='http://127.0.0.1:8766')
    args=parser.parse_args()
    saved=json.loads(args.calibration.read_text())
    camera_source=saved['camera']
    live=shared_frame(args.live_camera_url,camera_source)
    frame,captured,transport=live if live is not None else direct_frame(camera_source)
    calibration=load_calibration(args.calibration,camera_source,(frame.shape[1],frame.shape[0]))
    reference=next(p for p in calibration['points'] if p['number']==args.point)
    xy=reference['expected_mm']
    uv=project(np.linalg.inv(calibration['matrix']),[xy])[0]
    marked=annotate_sample(frame,calibration,args.point,xy,uv,args.angles)
    for name,image in [('raw.png',frame),('marked.png',marked)]:
        if not cv2.imwrite(str(args.output/name),image):
            raise OSError(f'画像を保存できません: {name}')
    (args.output/'capture.json').write_text(json.dumps({'captured_at':captured,'camera':camera_source,
        'image_size':[frame.shape[1],frame.shape[0]],'reference_target_px':uv.tolist(),**transport},indent=2)+'\n')


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,ValueError,OSError,KeyError,cv2.error) as error:
        raise SystemExit(f'撮影エラー: {error}')
