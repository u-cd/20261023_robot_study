"""Evaluate saved best checkpoints with a shared COCO evaluator and batch-one CPU timing."""
import contextlib
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.tools/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib'))
os.environ.setdefault('YOLO_OFFLINE', 'true')
import cv2
import numpy as np
import torch
from PIL import Image
from faster_coco_eval import COCO, COCOeval_faster

NAMES = ['sponge', 'target']


def detector(name, cfg):
    output = ROOT / cfg['output'] / name
    if name.startswith('yolo'):
        from ultralytics import YOLO, settings
        from ultralytics.utils import torch_utils
        torch_utils.NUM_THREADS = cfg['threads']
        settings.update({'sync': False})
        path = output / 'weights/best.pt'
        model = YOLO(str(path))
        def predict(frame, confidence=0.001):
            result = model.predict(frame, imgsz=cfg['image_size'], device='cpu',
                                   conf=confidence, iou=0.7, max_det=100, rect=False, verbose=False)[0]
            return [(int(c), float(s), b.tolist()) for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]
        params = sum(p.numel() for p in model.model.parameters())
    else:
        v2 = name.startswith('deimv2_')
        sys.path.insert(0, str(ROOT / ('reference/DEIMv2' if v2 else 'reference/DEIM')))
        from engine.core import YAMLConfig
        import torchvision.transforms.v2 as T
        config = YAMLConfig(str(output / 'deim.yml'))
        model = config.model
        path = output / 'best_stg1.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        model = model.deploy()  # Official inference path: fold reparameterizable blocks.
        post = config.postprocessor.eval()
        transforms = [T.Resize((cfg['image_size'], cfg['image_size'])), T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        if v2:
            transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transform = T.Compose(transforms)
        def predict(frame, confidence=0.001):
            h, w = frame.shape[:2]
            inputs = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            result = post(model(inputs), torch.tensor([[w, h]]))[0]
            rows = [(int(c), float(s), b.tolist()) for c, s, b in zip(result['labels'], result['scores'], result['boxes']) if s >= confidence]
            return sorted(rows, key=lambda r: r[1], reverse=True)[:100]
        params = sum(p.numel() for p in model.parameters())
    return predict, path, params


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    w, h = max(0, min(a[2], b[2])-x), max(0, min(a[3], b[3])-y)
    inter = w*h
    return inter / max(1e-9, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def operational_metrics(predictions, ground_truth, cfg):
    counts = [{'tp': 0, 'fp': 0, 'fn': 0, 'center_errors_px': []} for _ in NAMES]
    both, exact = 0, 0
    per_image = []
    for image_id, gt in ground_truth.items():
        selected = [p for p in predictions[image_id] if p[1] >= cfg['confidence_threshold']]
        matched = set()
        image_fp = 0
        for cls, score, box in sorted(selected, key=lambda r: r[1], reverse=True):
            candidates = [(iou(box, gbox), j) for j, (gcls, gbox) in enumerate(gt) if j not in matched and gcls == cls]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= cfg['iou_threshold']:
                matched.add(index)
                counts[cls]['tp'] += 1
                gbox = gt[index][1]
                counts[cls]['center_errors_px'].append(math.hypot((box[0]+box[2]-gbox[0]-gbox[2])/2, (box[1]+box[3]-gbox[1]-gbox[3])/2))
            else:
                counts[cls]['fp'] += 1
                image_fp += 1
        for j, (cls, _) in enumerate(gt):
            if j not in matched:
                counts[cls]['fn'] += 1
        all_found = len(matched) == len(gt)
        both += all_found
        exact += all_found and image_fp == 0
        per_image.append({'image_id': image_id, 'matched': len(matched), 'gt': len(gt), 'false_positives': image_fp})
    for c in counts:
        c['precision'] = c['tp']/max(1, c['tp']+c['fp'])
        c['recall'] = c['tp']/max(1, c['tp']+c['fn'])
        errors = c.pop('center_errors_px')
        c['mean_center_error_px'] = statistics.mean(errors) if errors else None
    return {'per_class': dict(zip(NAMES, counts)), 'both_found_images': both,
            'both_found_without_extra_detections': exact, 'per_image': per_image}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--model', choices=['yolo11n', 'deim_n', 'yolo11s', 'deim_s', 'deimv2_s', 'deimv2_m'])
    parser.add_argument('--reuse-existing', action='store_true', help='Keep evaluations of previously completed models')
    parser.add_argument('--render-only', action='store_true', help='Rebuild comparison from saved predictions/images')
    args = parser.parse_args()
    split = args.split
    cfg = json.loads((ROOT / 'configs/vision_experiment.json').read_text())
    torch.set_num_threads(cfg['threads'])
    output, dataset = ROOT / cfg['output'], ROOT / cfg['dataset']
    annotation = dataset / f'annotations/{split}.json'
    document = json.loads(annotation.read_text())
    frames = {im['id']: cv2.imread(str(dataset / f'images/{split}' / im['file_name'])) for im in document['images']}
    truth = {key: [] for key in frames}
    for a in document['annotations']:
        x, y, w, h = a['bbox']
        truth[a['image_id']].append((a['category_id'], [x, y, x+w, y+h]))
    coco_gt = COCO(str(annotation))
    report = {'config': cfg, 'split': split, 'images': len(frames), 'objects': len(document['annotations']), 'models': {}}
    drawn = {}
    if not args.model:
        # The official DEIM registry mutates defaults when models are built.
        # Fresh processes prevent Nano defaults leaking into Small's configuration.
        for name in cfg['models']:
            if not args.render_only and not (args.reuse_existing and (output / name / f'{split}_evaluation.json').exists()):
                subprocess.run([sys.executable, __file__, '--split', split, '--model', name], check=True, cwd=ROOT)
            metrics = json.loads((output / name / f'{split}_evaluation.json').read_text())
            report['models'][name] = metrics
            drawn[name] = {im['id']: cv2.imread(str(output / name / f'{split}_images' / im['file_name'])) for im in document['images']}
        (output / f'{split}_evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
        comparison(cfg, split, frames, truth, drawn, output)
        return
    for name in [args.model]:
        print(f'Evaluating {name}', flush=True)
        predict, path, params = detector(name, cfg)
        with torch.inference_mode():
            predictions = {image_id: predict(frame) for image_id, frame in frames.items()}
            for _ in range(3):
                predict(next(iter(frames.values())), cfg['confidence_threshold'])
            timings = []
            for repeat in range(3):
                for image_id, frame in frames.items():
                    start = time.perf_counter()
                    predict(frame, cfg['confidence_threshold'])
                    timings.append(1000*(time.perf_counter()-start))
        coco_rows = []
        for image_id, rows in predictions.items():
            for cls, score, (x1, y1, x2, y2) in rows:
                coco_rows.append({'image_id': image_id, 'category_id': cls, 'score': score, 'bbox': [x1, y1, max(0, x2-x1), max(0, y2-y1)]})
        (output / name / f'{split}_predictions.json').write_text(json.dumps(coco_rows, indent=2)+'\n')
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            evaluator = COCOeval_faster(coco_gt, coco_gt.loadRes(coco_rows), 'bbox')
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        (output / name / f'{split}_coco.txt').write_text(captured.getvalue())
        metrics = {'weights': str(path.relative_to(ROOT)), 'parameters': params,
            'map50_95': float(evaluator.stats[0]), 'map50': float(evaluator.stats[1]),
            'latency_mean_ms': statistics.mean(timings), 'latency_median_ms': statistics.median(timings),
            'latency_p95_ms': float(np.percentile(timings, 95)), 'timed_frames': len(timings),
            'latency_samples_ms': timings,
            'operating_threshold': cfg['confidence_threshold'],
            **operational_metrics(predictions, truth, cfg)}
        filenames = {im['id']: im['file_name'] for im in document['images']}
        for row in metrics['per_image']:
            row['file_name'] = filenames[row['image_id']]
        report['models'][name] = metrics
        folder = output / name / f'{split}_images'
        folder.mkdir(exist_ok=True)
        drawn[name] = {}
        for im in document['images']:
            frame = frames[im['id']].copy()
            for cls, score, box in predictions[im['id']]:
                if score < cfg['confidence_threshold']:
                    continue
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f'{NAMES[cls]} {score:.2f}', (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 0), 1)
            cv2.imwrite(str(folder / im['file_name']), frame)
            drawn[name][im['id']] = frame
        (output / name / f'{split}_evaluation.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print(json.dumps({name: {k: metrics[k] for k in ('map50_95', 'map50', 'latency_mean_ms', 'both_found_images')}}), flush=True)
        del predict


def comparison(cfg, split, frames, truth, drawn, output):
    # A compact visual comparison; all twelve individual frames are also saved.
    ids = list(frames)
    errors = {i: [0, 0] for i in ids}
    for name in cfg['models']:
        metrics = json.loads((output / name / f'{split}_evaluation.json').read_text())
        for row in metrics['per_image']:
            errors[row['image_id']][0] += row['gt'] - row['matched']
            errors[row['image_id']][1] += row['false_positives']
    selected = [ids[0]]
    for image_id in sorted(ids, key=lambda i: tuple(errors[i]), reverse=True):
        if image_id not in selected and any(errors[image_id]):
            selected.append(image_id)
        if len(selected) == 3:
            break
    for image_id in [ids[len(ids)//2], ids[-1], *ids]:
        if len(selected) < 3 and image_id not in selected:
            selected.append(image_id)
    canvas = np.full((3*270, (1+len(cfg['models']))*320, 3), 245, np.uint8)
    for row, image_id in enumerate(selected):
        gt_frame = frames[image_id].copy()
        for cls, box in truth[image_id]:
            x1, y1, x2, y2 = map(round, box)
            cv2.rectangle(gt_frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(gt_frame, NAMES[cls], (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 180, 0), 1)
        for column, name in enumerate(['ground truth', *cfg['models']]):
            frame = gt_frame if column == 0 else drawn[name][image_id]
            x, y = column*320, row*270
            canvas[y+30:y+270, x:x+320] = cv2.resize(frame, (320, 240))
            cv2.putText(canvas, f'{name} / image {image_id}', (x+5, y+21), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1)
    cv2.imwrite(str(output / f'{split}_comparison.jpg'), canvas)


if __name__ == '__main__':
    main()
