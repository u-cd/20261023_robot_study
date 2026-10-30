"""2枚の同じ先端点を手動指定し、画像上のずれを比較する。アームは操作しない。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from table_geometry import load_calibration, project, plane_estimate

WINDOW='Reproduction displacement - click SAME tip point in BOTH images / R reset / Enter save / Q quit'
ZOOM=4
HALF=60


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir',type=Path,required=True)
    args=parser.parse_args()
    folder=args.test_dir
    test=json.loads((folder/'test.json').read_text())
    saved=json.loads((folder/'table_calibration.json').read_text())
    cal=load_calibration(folder/'table_calibration.json',saved['camera'],tuple(saved['image_size']))
    frames=[cv2.imread(str(folder/name)) for name in ['baseline_raw.png','raw.png']]
    if any(f is None for f in frames) or frames[0].shape!=frames[1].shape:
        raise ValueError('同じ解像度の比較画像2枚が必要です。')
    h,w=frames[0].shape[:2]
    center=project(np.linalg.inv(cal['matrix']),[test['reference_xy_mm']])[0]
    x0=max(0,min(round(center[0])-HALF,w-2*HALF))
    y0=max(0,min(round(center[1])-HALF,h-2*HALF))
    size=2*HALF*ZOOM
    selections=[None,None]
    display=None

    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and 0<=x<2*size and 30<=y<size+30:
            index=x//size
            selections[index]=[x0+(x%size)/ZOOM,y0+(y-30)/ZOOM]

    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW,click)
    print('左：元の記録、右：今回。両方で「同じグリッパー先端中心」を1点ずつクリックしてください。',flush=True)
    print('4倍拡大（新しい画素情報を増やすものではありません）。Enterで比較数値・画像を保存します。',flush=True)
    try:
        while True:
            display=np.full((size+190,2*size,3),30,dtype=np.uint8)
            for i,frame in enumerate(frames):
                crop=cv2.resize(frame[y0:y0+2*HALF,x0:x0+2*HALF],(size,size),interpolation=cv2.INTER_NEAREST)
                if selections[i] is not None:
                    u,v=selections[i]
                    pixel=(round((u-x0)*ZOOM),round((v-y0)*ZOOM))
                    cv2.drawMarker(crop,pixel,(0,200,255),cv2.MARKER_CROSS,18,1)
                display[30:30+size,i*size:(i+1)*size]=crop
                title='Original record' if i==0 else 'Returned pose'
                cv2.putText(display,title,(i*size+10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
            lines=['Click SAME physical tip landmark in each image (4x zoom).',
                   'R: reset / click again: correct / Enter: save / Q: quit']
            result=None
            if all(p is not None for p in selections):
                delta=np.subtract(selections[1],selections[0])
                projected=[plane_estimate(cal,p) for p in selections]
                delta_mm=(np.subtract(projected[1],projected[0]).tolist()
                          if all(p is not None for p in projected) else None)
                result={'measured_at':datetime.now(timezone.utc).isoformat(),
                        'measurement_type':'manual_image_landmarks','baseline_tip_pixel':selections[0],
                        'returned_tip_pixel':selections[1], 'delta_pixel':delta.tolist(),
                        'displacement_pixel':float(np.linalg.norm(delta)),
                        'paper_plane_projection_mm':projected,'apparent_delta_mm':delta_mm,
                        'apparent_displacement_mm':float(np.linalg.norm(delta_mm)) if delta_mm is not None else None,
                        'actual_3d_position_error_mm':None,'height_condition':'hover_unmeasured',
                        'baseline_image_sha256':hashlib.sha256((folder/'baseline_raw.png').read_bytes()).hexdigest(),
                        'returned_image_sha256':hashlib.sha256((folder/'raw.png').read_bytes()).hexdigest(),
                        'limitations':'Manual landmark selection uncertainty not estimated. Paper-plane projection only; actual tip height / change in height unknown. Not a measured 3D error.'}
                lines += [f'Delta image: dx={delta[0]:+.2f}, dy={delta[1]:+.2f} px; distance={np.linalg.norm(delta):.2f} px']
                if delta_mm is not None:
                    lines += [f'APPARENT paper-plane delta: X={delta_mm[0]:+.1f}, Y={delta_mm[1]:+.1f} mm; distance={np.linalg.norm(delta_mm):.1f} mm']
                else:
                    lines += ['Outside calibrated area: no mm conversion. Image displacement only.']
            lines += ['Actual 3D error UNKNOWN. Height / landmark selection affect this estimate.']
            for i,line in enumerate(lines):
                cv2.putText(display,line,(8,size+54+i*25),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)
            cv2.imshow(WINDOW,display)
            key=cv2.waitKey(30)&0xff
            if key==ord('r'):selections=[None,None]
            if key in (ord('q'),27) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
            if key in (10,13) and result is not None:
                stem=datetime.now().strftime('landmark_%H%M%S_%f')
                image_path=folder/f'{stem}.png'
                if not cv2.imwrite(str(image_path),display):raise OSError('画像を保存できません。')
                result['comparison_image']=image_path.name
                path=folder/f'{stem}.json'
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                lines=['# 37番の画像上の先端位置の比較','',
                       f'画像上のずれ：{result["displacement_pixel"]:.2f}px。',
                       f'差（今回−元）：dx={delta[0]:+.2f}px、dy={delta[1]:+.2f}px。','']
                if delta_mm is not None:
                    lines += [f'紙面換算の見かけのずれ：約{result["apparent_displacement_mm"]:.1f}mm。',
                              f'差：X={delta_mm[0]:+.1f}mm、Y={delta_mm[1]:+.1f}mm。','']
                lines += ['実際の3D位置誤差は未測定。先端の高さや、手動で同じ点を選ぶ誤差が含まれる。',
                          f'![同じ先端点の指定結果]({image_path.name})']
                (folder/f'{stem}.md').write_text('\n'.join(lines)+'\n')
                print(f'保存: {path}\n画像上のずれ: {result["displacement_pixel"]:.2f}px\n紙面換算の目安: {result["apparent_displacement_mm"]}mm',flush=True)
                break
    finally:
        cv2.destroyAllWindows()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
    except (ValueError,RuntimeError,OSError,cv2.error) as error:raise SystemExit(str(error))
