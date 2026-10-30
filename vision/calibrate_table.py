"""長方形の四隅と独立した5点をクリックし、机面の座標変換を保存する。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .table_geometry import reference_points, fit_plane, check_errors, project
else:
    from table_geometry import reference_points, fit_plane, check_errors, project

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 'Table calibration - SPACE freeze / click points / U undo / R retake / Enter save / Q quit'
NAMES = ['A origin', 'B +X', 'C opposite', 'D +Y', 'V center',
         'V bottom midpoint (AB)', 'V right midpoint (BC)',
         'V top midpoint (CD)', 'V left midpoint (DA)']


def save_evidence(frame, points, width, height, source, status, matrix=None, stats=None, folder=None):
    """元画像・クリック点付き画像・数値記録を同じ名前で保存する。"""
    folder = Path(folder) if folder is not None else ROOT/'runs/table_calibration'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    raw_path = folder/f'{stamp}_raw.png'
    overlay_path = folder/f'{stamp}_points.png'
    record_path = folder/f'{stamp}.json'
    corners, checks = reference_points(width, height)
    expected = np.vstack([corners, checks])
    overlay = cv2.copyMakeBorder(frame, 0, 365, 0, 0, cv2.BORDER_CONSTANT, value=(25,30,35))
    h, w = frame.shape[:2]
    point_records = []
    for i, (x, y) in enumerate(points):
        color = (60,220,70) if i<4 else (0,180,255)
        cv2.drawMarker(overlay, (x,y), color, cv2.MARKER_CROSS, 18, 2)
        label = f'{i+1}:{"ABCD"[i] if i<4 else "V"+str(i-3)}'
        tx, ty = max(0,min(x+8,w-65)), max(18,y-9)
        cv2.putText(overlay,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(overlay,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
        entry = {'number':i+1,'name':NAMES[i], 'role':'fit' if i<4 else 'check',
                 'image_px':[x,y], 'expected_mm':expected[i].tolist()}
        if matrix is not None:
            estimated = project(matrix, [[x,y]])[0]
            entry['estimated_mm'] = estimated.tolist()
            entry['error_mm'] = float(np.linalg.norm(estimated-expected[i]))
        point_records.append(entry)
    if len(points)>=4:
        cv2.polylines(overlay,[np.array(points[:4],dtype=np.int32)],True,(60,220,70),1)
    lines = [f'TABLE CALIBRATION | {status} | {width:g} x {height:g} mm',
             'Origin A; +X A->B; +Y A->D | green: fit / orange: check',
             'No. name                    pixel (u,v)     expected mm     error mm']
    for entry in point_records:
        x,y = entry['image_px']
        ex,ey = entry['expected_mm']
        error = f"{entry['error_mm']:.2f}" if 'error_mm' in entry else '--'
        lines.append(f"{entry['number']} {entry['name'][:23]:23s} ({x:3d},{y:3d})    ({ex:5.1f},{ey:5.1f})    {error}")
    if stats is not None:
        lines.append(f"Check points only: RMSE {stats['rmse_mm']:.2f} mm / max {stats['max_error_mm']:.2f} mm; limit 5 mm")
    lines.append('Paper plane only; height uncorrected; not robot coordinates.')
    for i,line in enumerate(lines):
        cv2.putText(overlay,line,(8,h+23+i*24),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,240,245),1,cv2.LINE_AA)
    record = {'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'camera':source, 'image_size':[w,h], 'coordinate_frame':'calibration_rectangle',
        'units':'mm','axes':'origin A; +X A to B; +Y A to D; plane of paper only',
        'rectangle_mm':[width,height], 'corners_px':points[:4], 'check_points_px':points[4:],
        'homography_px_to_mm':matrix.tolist() if matrix is not None else None,
        'validation':stats, 'max_allowed_error_mm':5, 'status':status, 'points':point_records,
        'reference_image':str(raw_path.resolve()), 'annotated_image':str(overlay_path.resolve()),
        'evidence_json':str(record_path.resolve()),
        'limitations':'No lens distortion or object height correction; not robot coordinates.'}
    for path,image in [(raw_path,frame),(overlay_path,overlay)]:
        if not cv2.imwrite(str(path),image):
            raise OSError(f'画像を保存できません: {path}')
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(f'記録: {overlay_path}\n数値: {record_path}',flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width-mm', type=float, required=True, help='AからBまでの実測距離')
    parser.add_argument('--height-mm', type=float, required=True, help='AからDまでの実測距離')
    parser.add_argument('--camera', default='auto')
    parser.add_argument('--output', type=Path, default=ROOT/'configs/table_calibration.json')
    args = parser.parse_args()
    reference_points(args.width_mm, args.height_mm)
    if args.output.exists():
        parser.error('保存先は既に存在します。別の --output を指定するか、既存ファイルを退避してください。')
    if args.camera == 'auto':
        choices = sorted(Path('/dev/v4l/by-id').glob('*EMEET*video-index0'))
        if not choices:
            parser.error('EMEETカメラが見つかりません。接続または --camera を確認してください。')
        source = str(choices[0])
    else:
        source = args.camera
    camera = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_V4L2)
    frozen, points, message = None, [], ''

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None and len(points) < 9:
            if 0 <= x < frozen.shape[1] and 0 <= y < frozen.shape[0]:
                points.append([x, y])

    try:
        if not camera.isOpened():
            raise RuntimeError('カメラを開けません。live_detect等のカメラ画面を終了してください。')
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Autosize keeps click coordinates in original image pixels (information below image).
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, click)
        print('紙のA/B/C/Dと、中央・AB/BC/CD/DAの中点を用意してください。')
        print('SPACEで静止→9点クリック→Enterで校正・記録。Sで途中記録、Uで戻る、Rで撮り直す。')
        while True:
            if frozen is None:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError('カメラ画像を取得できません。')
            else:
                frame = frozen
            canvas = cv2.copyMakeBorder(frame, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=(25, 30, 35))
            h, w = frame.shape[:2]
            for i, (x, y) in enumerate(points):
                color = (60, 220, 70) if i < 4 else (0, 180, 255)
                cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_CROSS, 12, 1)
                cv2.putText(canvas, str(i+1), (x+5, y-6), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            if len(points) >= 4:
                cv2.polylines(canvas, [np.array(points[:4], dtype=np.int32)], True, (60,220,70), 1)
            instruction = 'SPACE: freeze image' if frozen is None else (
                f'Click {len(points)+1}/9: {NAMES[len(points)]}' if len(points)<9 else 'All points selected. Enter: validate and save')
            lines = [f'Rectangle: {args.width_mm:g} x {args.height_mm:g} mm | {instruction}',
                     'A(0,0) -> B(W,0) -> C(W,H) -> D(0,H)',
                     'U: undo / R: retake / S: save evidence / Q: quit', message]
            for i, text in enumerate(lines):
                cv2.putText(canvas, text, (8, h+23+i*25), cv2.FONT_HERSHEY_SIMPLEX, .45, (230,235,240), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20)&0xff
            if key in (ord('q'),27) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                break
            if key == 32 and frozen is None:
                frozen = frame.copy()
            if key == ord('s') and frozen is not None:
                save_evidence(frozen,points,args.width_mm,args.height_mm,source,'partial')
                message = 'Evidence saved (not an accepted calibration).'
            if key == ord('u') and points:
                points.pop()
                message = ''
            if key == ord('r'):
                frozen, points, message = None, [], ''
            if key in (10,13) and frozen is not None and len(points)==9:
                try:
                    matrix = fit_plane(points[:4],args.width_mm,args.height_mm)
                    stats = check_errors(matrix,points[4:],args.width_mm,args.height_mm)
                    message = f"Check RMSE: {stats['rmse_mm']:.1f} mm / max: {stats['max_error_mm']:.1f} mm (limit 5 mm)"
                    print(message, flush=True)
                    passed = stats['max_error_mm'] <= 5
                    record = save_evidence(frozen,points,args.width_mm,args.height_mm,source,
                                           'passed' if passed else 'failed',matrix,stats)
                    if not passed:
                        message += ' FAILED (recorded): U/R to retry'
                        continue
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    # Exclusive creation avoids accidentally replacing a calibration from another run.
                    with args.output.open('x') as f:
                        json.dump(record,f,ensure_ascii=False,indent=2)
                        f.write('\n')
                    print(f'保存: {args.output}\nアーム座標との対応と物体高さの補正は未実施です。',flush=True)
                    break
                except ValueError as e:
                    save_evidence(frozen,points,args.width_mm,args.height_mm,source,'invalid_geometry')
                    message = str(e)
                    print(message,flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n終了しました。')
    except (ValueError,RuntimeError,OSError,cv2.error) as e:
        raise SystemExit(f'エラー: {e}')
