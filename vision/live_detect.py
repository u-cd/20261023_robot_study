"""学習済みYOLOでWebカメラを表示する。アームやシリアル通信は操作しない。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

if __package__:
    from .table_geometry import load_calibration, plane_estimate
else:
    from table_geometry import load_calibration, plane_estimate

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.tools/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib'))
os.environ.setdefault('YOLO_OFFLINE', 'true')

import cv2
import torch
from ultralytics import YOLO, settings
from ultralytics.utils import torch_utils

CLASSES = {0: 'sponge', 1: 'target'}
COLORS = {0: (60, 220, 70), 1: (0, 170, 255)}
WINDOW = 'Robot vision - pixel coordinates (Q: quit, S: snapshot)'


def camera_source(value):
    if value == 'auto':
        choices = sorted(Path('/dev/v4l/by-id').glob('*EMEET*video-index0'))
        if not choices:
            raise ValueError('EMEETカメラが見つかりません。USB接続を確認するか --camera /dev/video2 を指定してください。')
        return str(choices[0])
    return int(value) if value.isdigit() else value


def detections_from_result(result):
    """Ultralyticsの枠は元画像のピクセル座標へ復元済み。"""
    detections = []
    for row in result.boxes.data.cpu().tolist():
        x1, y1, x2, y2, score, class_id = row
        class_id = int(class_id)
        if class_id not in CLASSES:
            raise ValueError(f'想定外のクラス: {class_id}')
        detections.append({'class_id': class_id, 'name': CLASSES[class_id],
            'confidence': score, 'box_xyxy_px': [x1, y1, x2, y2],
            'center_px': [(x1+x2)/2, (y1+y2)/2]})
    return detections


def pair_status(detections):
    counts = {name: sum(d['name'] == name for d in detections) for name in CLASSES.values()}
    if any(n > 1 for n in counts.values()):
        status = 'MULTIPLE: check extra detections'
    elif any(n == 0 for n in counts.values()):
        status = 'MISSING: ' + ', '.join(name for name, n in counts.items() if not n)
    else:
        status = 'ONE EACH: image positions only'
    return status, counts


def annotate(frame, detections, fps, inference_ms, calibration=None):
    image = frame.copy()
    if calibration is not None:
        import numpy as np
        corners = np.array(calibration['corners_px'], dtype=np.int32)
        cv2.polylines(image, [corners], True, (220, 150, 40), 1)
        for name, corner in zip('ABCD', corners):
            cv2.putText(image, name, tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 150, 40), 1)
    for d in detections:
        color = COLORS[d['class_id']]
        x1, y1, x2, y2 = map(round, d['box_xyxy_px'])
        cx, cy = map(round, d['center_px'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(image, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        text = f"{d['name']} {d['confidence']:.2f} ({cx}, {cy}) px"
        if 'plane_estimate_mm' in d:
            point = d['plane_estimate_mm']
            label = f'plane~ ({point[0]:.0f}, {point[1]:.0f}) mm' if point is not None else 'outside calibrated area'
            cv2.putText(image, label, (max(0, x1), min(image.shape[0]-5, y2+18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]
        tx = max(0, min(x1, image.shape[1]-tw-4))
        ty = max(18, min(y1-6, image.shape[0]-5))
        cv2.rectangle(image, (tx, ty-16), (tx+tw+4, ty+4), (20, 30, 25), -1)
        cv2.putText(image, text, (tx+2, ty), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    # Put status outside the camera image so targets near the upper edge stay visible.
    height, width = frame.shape[:2]
    canvas = cv2.copyMakeBorder(image, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 36, 43))
    status, counts = pair_status(detections)
    lines = [f'YOLO11n | {width}x{height} | {fps:.1f} FPS | inference {inference_ms:.0f} ms',
             f"sponge: {counts['sponge']}  target: {counts['target']} | {status}",
             ('Plane~ mm: origin A, +X A->B, +Y A->D | height uncorrected | Q/S'
              if calibration is not None else
              'Pixels: origin top-left, x right, y down | Q: quit / S: snapshot')]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 23+25*i), cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 240, 245), 1, cv2.LINE_AA)
    return canvas


def save_snapshot(frame, overlay, detections, args, source, captured_at):
    args.output.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    raw = args.output / f'{stem}_raw.jpg'
    marked = args.output / f'{stem}_overlay.jpg'
    for path, data in [(raw, frame), (marked, overlay)]:
        if not cv2.imwrite(str(path), data):
            raise OSError(f'保存できません: {path}')
    status, counts = pair_status(detections)
    record = {'captured_at': captured_at, 'camera': source, 'weights': str(args.weights.resolve()),
        'coordinate_system': 'original image pixels: origin top-left, x right, y down',
        'image_size': {'width': frame.shape[1], 'height': frame.shape[0]},
        'confidence_threshold': args.confidence, 'inference_image_size': args.imgsz,
        'calibration_file': str(args.calibration.resolve()) if args.calibration else None,
        'plane_estimate_note': 'rectangle frame; assumes image point is on paper plane; height uncorrected',
        'detections': detections, 'counts': counts, 'status': status,
        'raw_image': raw.name, 'overlay_image': marked.name}
    path = args.output / f'{stem}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(f'保存: {marked}\n座標: {path}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='auto', help='auto: EMEETの固定IDを優先、または/dev/video2・番号')
    parser.add_argument('--weights', type=Path, default=ROOT / 'runs/sponge_comparison/yolo11n/weights/best.pt')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--frames', type=int, default=0, help='指定枚数で終了。0はQを押すまで継続')
    parser.add_argument('--headless', action='store_true', help='画面を表示せずカメラと推論を確認')
    parser.add_argument('--save-last', action='store_true', help='終了時の最新1枚を保存')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/live_vision')
    parser.add_argument('--calibration', type=Path, help='校正JSON。紙面上への推定mm座標を追加表示する')
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.threads < 1 or args.imgsz < 32 or args.imgsz % 32 or args.frames < 0:
        parser.error('confidenceは0より大きく1以下、threadsは1以上、imgszは32以上の32の倍数、framesは0以上です。')
    if args.headless and args.frames == 0:
        parser.error('--headlessでは終了枚数を --frames で指定してください。')
    if not args.weights.is_file():
        parser.error(f'学習済みモデルがありません: {args.weights}')
    source = camera_source(args.camera)
    torch.set_num_threads(args.threads)
    torch_utils.NUM_THREADS = args.threads
    settings.update({'sync': False})
    model = YOLO(str(args.weights))
    if model.names != CLASSES:
        raise ValueError(f'2クラスの学習済みモデルを指定してください: {model.names}')
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError(f'カメラを開けません: {source}。他のカメラアプリを終了して接続を確認してください。')
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        mode = '画像座標と紙面上の推定mm座標を表示します（高さ未補正）。' if args.calibration else '画像上の座標だけを表示します。'
        print(f'カメラ: {source}\nモデル: {args.weights}\n{mode}Qで終了、Sで画像と座標を保存。', flush=True)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 960, 855)
        frames, fps = 0, 0.0
        frame = overlay = detections = captured_at = None
        calibration = None
        while True:
            start = time.perf_counter()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('カメラ画像を取得できません。USB接続・他アプリでの使用を確認してください。')
            captured_at = datetime.now(timezone.utc).isoformat()
            if args.calibration and calibration is None:
                calibration = load_calibration(args.calibration, str(source), (frame.shape[1], frame.shape[0]))
            inference_start = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.confidence,
                iou=.7, max_det=100, rect=False, device='cpu', verbose=False)[0]
            detections = detections_from_result(result)
            if calibration is not None:
                for detection in detections:
                    detection['plane_estimate_mm'] = plane_estimate(calibration, detection['center_px'])
            inference_ms = 1000*(time.perf_counter()-inference_start)
            elapsed = time.perf_counter()-start
            instant_fps = 1/max(elapsed, 1e-9)
            fps = instant_fps if frames == 0 else .85*fps+.15*instant_fps
            overlay = annotate(frame, detections, fps, inference_ms, calibration)
            frames += 1
            if not args.headless:
                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(1) & 0xff
                if key == ord('s'):
                    save_snapshot(frame, overlay, detections, args, source, captured_at)
                if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.frames and frames >= args.frames:
                break
        if args.save_last and frame is not None:
            save_snapshot(frame, overlay, detections, args, source, captured_at)
        print(f'終了: {frames}フレーム / 最新状態: {pair_status(detections)[0]}', flush=True)
    finally:
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n終了しました。')
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f'エラー: {error}')
