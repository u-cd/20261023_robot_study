"""格子の全交点をクリックし、平面校正と画像記録を保存する。"""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .table_geometry import grid_reference, fit_grid, project
else:
    from table_geometry import grid_reference, fit_grid, project

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 'Grid calibration - all intersections'
SCALE = 1.5


def save_record(folder, frame, points, source, status, matrix=None, stats=None, columns=8, rows=8, spacing_mm=30):
    """draftは随時更新、Enterでの評価記録は別フォルダで残す。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected, checks = grid_reference(columns, rows, spacing_mm)
    total = columns*rows
    width, height = (columns-1)*spacing_mm, (rows-1)*spacing_mm
    estimated = project(matrix, points) if matrix is not None else None
    details = []
    marked = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    for i, (u,v) in enumerate(points):
        error = float(np.linalg.norm(estimated[i]-expected[i])) if estimated is not None else None
        entry = {'number':i+1, 'row_from_bottom':i//columns, 'column_from_left':i%columns,
                 'role':'check' if checks[i] else 'fit', 'image_px':[u,v],
                 'expected_mm':expected[i].tolist(),
                 'estimated_mm':estimated[i].tolist() if estimated is not None else None,
                 'error_mm':error}
        details.append(entry)
        color = (0,180,255) if checks[i] else (60,220,70)
        if error is not None and error>5:
            color = (50,50,255)
        x,y = round(u*2),round(v*2)
        cv2.drawMarker(marked,(x,y),color,cv2.MARKER_CROSS,16,2)
        label=f'{i+1}'
        tx,ty=max(0,min(x+6,marked.shape[1]-36)),max(18,y-6)
        for thickness,c in [(3,(0,0,0)),(1,color)]:
            cv2.putText(marked,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.55,c,thickness,cv2.LINE_AA)
    marked=cv2.copyMakeBorder(marked,0,100,0,0,cv2.BORDER_CONSTANT,value=(25,30,35))
    lines=[f'{status} | {columns}x{rows} intersections / spacing {spacing_mm:g} mm / {width:g}x{height:g} mm',
           'Origin #1 bottom-left; +X right along paper; +Y up along paper. Green fit / orange check / red >5mm.',
           f'{len(points)}/{total} points. Full coordinates and errors: record.json / points.csv / report.html']
    if stats:
        lines.append(f"Fit RMSE/max: {stats['fit']['rmse_mm']:.2f}/{stats['fit']['max_error_mm']:.2f} mm | Check RMSE/max: {stats['check']['rmse_mm']:.2f}/{stats['check']['max_error_mm']:.2f} mm")
    for i,line in enumerate(lines):
        cv2.putText(marked,line,(8,marked.shape[0]-77+i*22),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,240,245),1,cv2.LINE_AA)
    corners=[points[i] for i in [0,columns-1,total-1,total-columns]] if len(points)==total else []
    record={'schema_version':2,'created_at':datetime.now(timezone.utc).isoformat(),
            'camera':source,'image_size':[frame.shape[1],frame.shape[0]],
            'coordinate_frame':'calibration_rectangle','units':'mm',
            'axes':'origin #1 bottom-left; +X first row; +Y first column',
            'grid':{'columns':columns,'rows':rows,'spacing_mm':spacing_mm},'rectangle_mm':[width,height],
            'grid_points_px':points,'corners_px':corners,'points':details,
            'homography_px_to_mm':matrix.tolist() if matrix is not None else None,
            'validation':stats,'max_allowed_error_mm':5,'status':status,
            'method':f'{total-int(checks.sum())}-point least squares homography; {int(checks.sum())} interior held-out points; no outlier removal',
            'reference_image':str((folder/'raw.png').resolve()),
            'annotated_image':str((folder/'points.png').resolve()),
            'limitations':'Paper plane only. No lens distortion or object height correction. Not robot coordinates.'}
    for name,data in [('raw.png',frame),('points.png',marked)]:
        if not cv2.imwrite(str(folder/name),data):
            raise OSError(f'画像を保存できません: {folder/name}')
    (folder/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    with (folder/'points.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['number','row_from_bottom_0based','column_from_left_0based','role',
                         'u_px','v_px','expected_x_mm','expected_y_mm','estimated_x_mm','estimated_y_mm','error_mm'])
        for d in details:
            writer.writerow([d['number'],d['row_from_bottom'],d['column_from_left'],d['role'],
                             *d['image_px'],*d['expected_mm'],*(d['estimated_mm'] or ['', '']),
                             d['error_mm'] if d['error_mm'] is not None else ''])
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in
                  [d['number'],d['role'],d['image_px'],d['expected_mm'],d['estimated_mm'],
                   f"{d['error_mm']:.3f}" if d['error_mm'] is not None else '未計算'])+'</tr>' for d in details)
    summary=html.escape(json.dumps(stats,ensure_ascii=False,indent=2)) if stats else '未計算'
    (folder/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>格子校正の記録</title>
<style>body{{font:16px sans-serif;max-width:1300px;margin:24px auto;padding:16px}}img{{max-width:100%}}td,th{{border:1px solid #ccc;padding:8px}}table{{border-collapse:collapse}}pre{{white-space:pre-wrap}}</style>
<h1>格子校正：{html.escape(status)}</h1><p>{spacing_mm:g}mm間隔、横{columns}×縦{rows}交点、{width:g}×{height:g}mm。左下が原点、紙に沿って右が+X、上が+Y。</p>
<p>緑：計算用{total-int(checks.sum())}点／橙：計算に使わない確認用{int(checks.sum())}点／赤：誤差5mm超。番号は左下から右へ、段ごとに上がる順。</p>
<p>{html.escape(record['created_at'])} / {html.escape(source)}</p><img src="points.png" alt="全クリック点の番号付き画像">
<p><a href="raw.png">元画像</a> / <a href="record.json">JSON</a> / <a href="points.csv">CSV</a></p>
<p>紙面の座標であり、物体高さ・レンズ歪みは未補正。アーム座標との対応は未実施。</p><pre>{summary}</pre>
<table><tr><th>番号</th><th>用途</th><th>画像座標 px</th><th>実寸座標 mm</th><th>変換結果 mm</th><th>誤差 mm</th></tr>{table}</table>''')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--columns',type=int,default=7,help='横方向の交点数（マス数ではない）')
    parser.add_argument('--rows',type=int,default=6,help='縦方向の交点数（マス数ではない）')
    parser.add_argument('--spacing-mm',type=float,default=30)
    parser.add_argument('--camera',default='auto')
    parser.add_argument('--output',type=Path,default=ROOT/'configs/table_calibration.json')
    parser.add_argument('--resume',type=Path,help='途中保存のrecord.jsonから再開')
    args=parser.parse_args()
    expected,checks=grid_reference(args.columns,args.rows,args.spacing_mm)
    total=len(expected)
    if args.output.exists():
        parser.error('校正ファイルは既に存在します。--outputで別名を指定してください。')
    if args.camera=='auto':
        choices=sorted(Path('/dev/v4l/by-id').glob('*EMEET*video-index0'))
        if not choices:
            parser.error('EMEETカメラが見つかりません。')
        source=str(choices[0])
    else:
        source=args.camera
    frozen,points,session,selected=None,[],None,None
    if args.resume:
        saved=json.loads(args.resume.read_text())
        if saved['camera']!=source or saved['grid']!={'columns':args.columns,'rows':args.rows,'spacing_mm':args.spacing_mm}:
            parser.error('再開するカメラまたは格子設定が異なります。')
        frozen=cv2.imread(saved['reference_image'])
        if frozen is None or [frozen.shape[1],frozen.shape[0]]!=saved['image_size']:
            parser.error('再開用の元画像を読み込めません。')
        points=saved['grid_points_px']
        if len(points)>total or (points and (np.asarray(points).shape!=(len(points),2) or not np.isfinite(points).all())):
            parser.error('再開用の点データが不正です。')
        # Keep the old session intact; subsequent saves go to a new session.
        session=ROOT/'runs/table_calibration'/datetime.now().strftime('grid_%Y%m%d_%H%M%S_%f')
    camera=cv2.VideoCapture(int(source) if source.isdigit() else source,cv2.CAP_V4L2)
    message=''
    def checkpoint():
        save_record(session/'draft',frozen,points,source,'partial',columns=args.columns,rows=args.rows,spacing_mm=args.spacing_mm)
    def click(event,x,y,flags,param):
        nonlocal selected
        if frozen is None or not (0<=x<frozen.shape[1]*SCALE and 0<=y<frozen.shape[0]*SCALE):
            return
        point=[x/SCALE,y/SCALE]
        if event==cv2.EVENT_RBUTTONDOWN and points:
            distances=np.linalg.norm(np.asarray(points)-point,axis=1)
            selected=int(np.argmin(distances)) if distances.min()<15 else None
        if event==cv2.EVENT_LBUTTONDOWN:
            if selected is not None:
                points[selected]=point
                selected=None
            elif len(points)<total:
                points.append(point)
            else:
                return
            checkpoint()
    try:
        if not camera.isOpened():
            raise RuntimeError('カメラを開けません。他のカメラ画面を終了してください。')
        for prop,value in [(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),
                           (cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),(cv2.CAP_PROP_BUFFERSIZE,1)]:
            camera.set(prop,value)
        cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW,click)
        print(f'横{args.columns}×縦{args.rows}交点、計{total}点。間隔{args.spacing_mm:g}mm。SPACEで静止。')
        print(f'左下から右へ{args.columns}点、1段上の左端へ戻り、上段まで{total}点。Enterで評価・保存。')
        print('各クリックを自動保存。Uで戻る。右クリックで点を選択→左クリックで修正。Qで終了。',flush=True)
        if session is not None:
            checkpoint()
        while True:
            if frozen is None:
                ok,frame=camera.read()
                if not ok:
                    raise RuntimeError('カメラ画像を取得できません。')
            else:
                frame=frozen
            display=cv2.resize(frame,None,fx=SCALE,fy=SCALE)
            for i,(u,v) in enumerate(points):
                color=(0,180,255) if checks[i] else (60,220,70)
                if i==selected:
                    color=(255,100,255)
                x,y=round(u*SCALE),round(v*SCALE)
                cv2.drawMarker(display,(x,y),color,cv2.MARKER_CROSS,12,1)
                for thickness,c in [(3,(0,0,0)),(1,color)]:
                    cv2.putText(display,str(i+1),(max(0,min(x+5,display.shape[1]-30)),max(16,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.43,c,thickness,cv2.LINE_AA)
            h=display.shape[0]
            display=cv2.copyMakeBorder(display,0,120,0,0,cv2.BORDER_CONSTANT,value=(25,30,35))
            if frozen is None:
                instruction=f'SPACE: freeze grid ({args.columns} columns x {args.rows} rows of intersections, {args.spacing_mm:g} mm)'
            elif selected is not None:
                instruction=f'Editing #{selected+1}: left-click its correct intersection'
            elif len(points)<total:
                i=len(points)
                instruction=f'NEXT #{i+1}/{total}: row {i//args.columns+1} from bottom, col {i%args.columns+1} from left; ({expected[i,0]:g},{expected[i,1]:g}) mm'
            else:
                instruction=f'{total}/{total} selected. Enter: fit, validate and save'
            for i,line in enumerate([instruction,'Order: left -> right, then next row UP (return to left edge).',
                                    'U undo / right-click point then left-click to edit / S record / R retake / Q quit',message]):
                cv2.putText(display,line,(8,h+24+i*25),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,235,240),1,cv2.LINE_AA)
            cv2.imshow(WINDOW,display)
            key=cv2.waitKey(20)&0xff
            if key in (ord('q'),27) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                if session:
                    print(f'途中記録: {session/"draft/record.json"}',flush=True)
                break
            if key==32 and frozen is None:
                frozen=frame.copy()
                session=ROOT/'runs/table_calibration'/datetime.now().strftime('grid_%Y%m%d_%H%M%S_%f')
                checkpoint()
                print(f'記録先: {session}',flush=True)
            if key==ord('u') and points:
                points.pop()
                selected=None
                checkpoint()
                message=''
            if key==ord('r'):
                frozen,points,session,selected=None,[],None,None
                message='Previous session records kept. SPACE to freeze a new image.'
            if key==ord('s') and frozen is not None:
                checkpoint()
                message='Draft image, points and report saved.'
            if key in (10,13) and len(points)==total and selected is None:
                matrix=stats=None
                try:
                    matrix,stats=fit_grid(points,args.columns,args.rows,args.spacing_mm)
                    passed=max(stats['fit']['max_error_mm'],stats['check']['max_error_mm'])<=5
                    status='passed' if passed else 'failed'
                except ValueError as error:
                    passed,status=False,'invalid_geometry'
                    message=str(error)
                attempt=session/datetime.now().strftime('attempt_%H%M%S_%f')
                record=save_record(attempt,frozen,points,source,status,matrix,stats,args.columns,args.rows,args.spacing_mm)
                print(f'{status}: {attempt/"report.html"}',flush=True)
                if not passed:
                    message='FAILED (recorded). Edit misplaced points; see report.html for point errors.'
                    continue
                args.output.parent.mkdir(parents=True,exist_ok=True)
                with args.output.open('x') as f:
                    json.dump(record,f,ensure_ascii=False,indent=2)
                    f.write('\n')
                print(f'校正保存: {args.output}\n確認用画像: {attempt/"points.png"}',flush=True)
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n終了しました。途中記録はruns/table_calibration/内にあります。')
    except (ValueError,RuntimeError,OSError,cv2.error,KeyError) as error:
        raise SystemExit(f'エラー: {error}')
